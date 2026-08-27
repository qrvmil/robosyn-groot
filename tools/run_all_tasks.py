#!/usr/bin/env python3
"""Resumable single-GPU state machine for all ten RoboSyn tasks."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.pipeline import (
    TASKS,
    Workspace,
    evaluation_episodes_for_task,
    pipeline_environment,
)


ORDERED_TASKS = (
    "drawer_open_place",
    *tuple(task for task in TASKS if task != "drawer_open_place"),
)
ALL_PHASES = ("bootstrap", "smoke", "train", "eval")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def new_state() -> dict[str, object]:
    return {
        "schema_version": 1,
        "updated_at": _now(),
        "active": None,
        "tasks": {
            task: {
                "status": "pending",
                "phases": {phase: "pending" for phase in ALL_PHASES},
                "last_error": None,
                "traceback": None,
            }
            for task in ORDERED_TASKS
        },
    }


def command_for_phase(task: str, phase: str) -> list[str]:
    if phase not in ALL_PHASES:
        raise ValueError(f"unknown orchestrator phase: {phase}")
    return [f"robosyn-{phase}", task]


class Orchestrator:
    def __init__(self, workspace: Workspace, phases: tuple[str, ...]):
        self.workspace = workspace
        self.phases = phases
        self.state_path = workspace.root / "reports/orchestrator_state.json"
        self.state = (
            json.loads(self.state_path.read_text())
            if self.state_path.is_file()
            else new_state()
        )
        self.current_process: subprocess.Popen | None = None
        self.terminate_requested = False
        self._reconcile_completed_evaluations()

    def _reconcile_completed_evaluations(self) -> None:
        """Reopen legacy/incomplete evals instead of trusting stale state alone."""
        try:
            registry = json.loads(
                (self.workspace.root / "configs/tasks/registry.json").read_text()
            )
        except (FileNotFoundError, TypeError, json.JSONDecodeError):
            registry = {"tasks": {}}
        for task, task_state in self.state["tasks"].items():
            if task_state["phases"].get("eval") != "completed":
                continue
            summary_path = self.workspace.root / "eval" / task / "summary.json"
            try:
                summary = json.loads(summary_path.read_text())
                entry = registry.get("tasks", {}).get(task, {})
                target = int(
                    entry.get("evaluation_episodes", evaluation_episodes_for_task(task))
                )
                valid = (
                    int(summary.get("number_of_episodes", 0)) == target
                    and int(summary.get("requested_episodes", 0)) == target
                    and summary.get("seed_list") == list(range(target))
                    and summary.get("inference_latency_scope") == "model_rpc_only"
                )
            except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError):
                valid = False
            if valid:
                continue
            task_state["phases"]["eval"] = "pending"
            task_state["status"] = "pending"
            task_state["last_error"] = (
                "completed evaluation reopened: artifacts do not match the per-task "
                "episode/seed/latency contract"
            )
            task_state["traceback"] = None

    def save(self) -> None:
        self.state["updated_at"] = _now()
        _write_json(self.state_path, self.state)

    def request_termination(self, signum, _frame) -> None:
        self.terminate_requested = True
        self.state["termination_signal"] = int(signum)
        self.save()
        process = self.current_process
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)

    def run_phase(self, task: str, phase: str) -> None:
        task_state = self.state["tasks"][task]
        self.state["active"] = {"task": task, "phase": phase, "started_at": _now()}
        task_state["status"] = "running"
        task_state["phases"][phase] = "running"
        self.save()
        command = command_for_phase(task, phase)
        print(f"[{_now()}] START {' '.join(command)}", flush=True)
        self.current_process = subprocess.Popen(
            command,
            cwd=self.workspace.root,
            env=pipeline_environment(self.workspace),
            start_new_session=True,
        )
        returncode = self.current_process.wait()
        self.current_process = None
        if returncode != 0:
            raise RuntimeError(f"{' '.join(command)} exited {returncode}")
        task_state["phases"][phase] = "completed"
        task_state["last_error"] = None
        task_state["traceback"] = None
        self.state["active"] = None
        self.save()
        print(f"[{_now()}] PASS {' '.join(command)}", flush=True)

    def run(self) -> int:
        failures = 0
        for task in ORDERED_TASKS:
            task_state = self.state["tasks"][task]
            for phase in self.phases:
                if task_state["phases"].get(phase) == "completed":
                    continue
                if self.terminate_requested:
                    task_state["status"] = "interrupted"
                    self.state["active"] = None
                    self.save()
                    return 143
                try:
                    self.run_phase(task, phase)
                except Exception as exc:
                    if self.terminate_requested:
                        task_state["status"] = "interrupted"
                        self.state["active"] = None
                        self.save()
                        return 143
                    failures += 1
                    task_state["status"] = "failed"
                    task_state["phases"][phase] = "failed"
                    task_state["last_error"] = str(exc)
                    task_state["traceback"] = traceback.format_exc()
                    self.state["active"] = None
                    self.save()
                    print(f"[{_now()}] FAILED {task}/{phase}: {exc}", flush=True)
                    break
            else:
                task_state["status"] = "completed"
                self.save()
        self.state["active"] = None
        self.state["status"] = "completed" if failures == 0 else "completed_with_failures"
        self.state["failure_count"] = failures
        self.save()
        return 0 if failures == 0 else 1


def run_continuously(
    orchestrator: Orchestrator,
    *,
    retry_delay: int = 60,
    sleeper=time.sleep,
) -> int:
    """Repeat resumable passes until every requested phase is verified."""
    while True:
        result = orchestrator.run()
        if result == 0 or orchestrator.terminate_requested:
            return result
        print(
            f"[{_now()}] pass completed with failures; retrying failed phases "
            f"in {retry_delay}s",
            flush=True,
        )
        sleeper(retry_delay)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phases", choices=("all", "eval"), default="all")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--retry-delay", type=int, default=60)
    args = parser.parse_args(argv)
    workspace = Workspace(
        Path(os.environ.get("ROBOSYN_WORK_ROOT", ROOT)),
        Path(os.environ.get("ROBOSYN_GROOT_SRC", "/opt/robosyn/Isaac-GR00T")),
    )
    workspace.initialize()
    phases = ALL_PHASES if args.phases == "all" else ("eval",)
    orchestrator = Orchestrator(workspace, phases)
    signal.signal(signal.SIGTERM, orchestrator.request_termination)
    signal.signal(signal.SIGINT, orchestrator.request_termination)
    if args.continuous:
        return run_continuously(orchestrator, retry_delay=args.retry_delay)
    return orchestrator.run()


if __name__ == "__main__":
    raise SystemExit(main())
