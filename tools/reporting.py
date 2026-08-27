"""Generate honest incremental training/evaluation summary artifacts."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.pipeline import (
    TASKS,
    Workspace,
    evaluation_episodes_for_task,
    training_steps_for_task,
)


def _read(path: Path) -> dict[str, object] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def _display(value: object) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def _tree_size(path: Path) -> int | None:
    if not path.is_dir():
        return None
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _has_valid_episode_results(eval_dir: Path, target: int) -> bool:
    results_path = eval_dir / "episode_results.jsonl"
    if not results_path.is_file():
        return False
    try:
        results = [
            json.loads(line)
            for line in results_path.read_text().splitlines()
            if line.strip()
        ]
        return (
            len(results) == target
            and [int(item["episode"]) for item in results] == list(range(target))
            and [int(item["seed"]) for item in results] == list(range(target))
            and all(not item.get("error") for item in results)
            and all(int(item.get("inference_calls", 0)) > 0 for item in results)
            and all(int(item.get("episode_length", 0)) > 0 for item in results)
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def refresh_reports(workspace: Workspace) -> tuple[Path, Path]:
    registry = _read(workspace.root / "configs/tasks/registry.json")
    if registry is None:
        raise FileNotFoundError("task registry is missing")
    training_rows: list[dict[str, object]] = []
    evaluation_rows: list[dict[str, object]] = []
    for task in TASKS:
        entry = registry["tasks"][task]
        training = _read(workspace.run_dir(task) / "training_result.json") or {}
        checkpoint_value = training.get("checkpoint")
        checkpoint = Path(str(checkpoint_value)) if checkpoint_value else None
        training_rows.append(
            {
                "task": task,
                "dataset_revision": entry["resolved_hf_revision"],
                "training_status": training.get("status", "not_started"),
                "target_optimizer_steps": entry.get(
                    "training_max_steps", training_steps_for_task(task)
                ),
                "optimizer_steps": training.get("optimizer_steps"),
                "global_batch_size": training.get("global_batch_size"),
                "gradient_accumulation_steps": training.get("gradient_accumulation_steps"),
                "effective_batch_size": training.get("effective_batch_size"),
                "final_train_loss": training.get("final_loss"),
                "final_checkpoint": checkpoint_value,
                "checkpoint_size_bytes": _tree_size(checkpoint) if checkpoint else None,
                "checkpoint_sha256": training.get("checkpoint_sha256"),
                "checkpoint_reload_status": training.get("reload_status"),
            }
        )
        eval_dir = workspace.root / "eval" / task
        summary = _read(eval_dir / "summary.json")
        evaluation_target = int(
            entry.get("evaluation_episodes", evaluation_episodes_for_task(task))
        )
        if (
            summary is not None
            and summary.get("requested_episodes") is not None
            and summary.get("inference_latency_scope") == "model_rpc_only"
            and int(summary.get("number_of_episodes", 0))
            == evaluation_target
            and int(summary["requested_episodes"]) == evaluation_target
            and summary.get("seed_list") == list(range(evaluation_target))
            and _has_valid_episode_results(eval_dir, evaluation_target)
        ):
            interval = summary.get("wilson_95_confidence_interval") or [None, None]
            evaluation_rows.append(
                {
                    "task": task,
                    "dataset_revision": entry["resolved_hf_revision"],
                    "checkpoint": summary.get("checkpoint_path"),
                    "checkpoint_sha256": summary.get("checkpoint_sha256"),
                    "eval_episodes": summary.get("number_of_episodes"),
                    "successes": summary.get("successes"),
                    "success_rate": summary.get("success_rate"),
                    "wilson_95_low": interval[0],
                    "wilson_95_high": interval[1],
                    "mean_episode_length": summary.get("mean_episode_length"),
                    "mean_inference_latency_s": summary.get("mean_inference_latency_s"),
                    "inference_latency_scope": summary.get("inference_latency_scope"),
                    "rollout_video_directory": str(workspace.root / "eval" / task / "videos"),
                    "protocol_source": summary.get("protocol_source"),
                }
            )
    report_dir = workspace.root / "reports"
    training_json = report_dir / "TRAINING_SUMMARY.json"
    evaluation_json = report_dir / "EVALUATION_SUMMARY.json"
    _atomic_json(training_json, training_rows)
    _atomic_json(evaluation_json, evaluation_rows)
    _atomic_csv(report_dir / "TRAINING_SUMMARY.csv", training_rows, list(training_rows[0]))
    eval_fields = [
        "task", "dataset_revision", "checkpoint", "checkpoint_sha256",
        "eval_episodes", "successes", "success_rate", "wilson_95_low",
        "wilson_95_high", "mean_episode_length", "mean_inference_latency_s",
        "inference_latency_scope",
        "rollout_video_directory", "protocol_source",
    ]
    _atomic_csv(report_dir / "EVALUATION_SUMMARY.csv", evaluation_rows, eval_fields)
    evaluations_by_task = {row["task"]: row for row in evaluation_rows}
    completed = len(evaluation_rows)
    lines = [
        "# RoboSyn × GR00T Final Status",
        "",
        f"Completed full evaluations: {completed}/{len(TASKS)}",
        "",
        "Only validated full closed-loop results are shown; unfinished entries are explicitly marked `not_started`.",
        "",
        "| task | dataset revision | training | steps/target | batch/accum | final loss | checkpoint | size bytes | reload | eval episodes | successes | success rate | Wilson 95% CI | videos |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for training in training_rows:
        evaluation = evaluations_by_task.get(training["task"], {})
        interval = (
            f"[{_display(evaluation.get('wilson_95_low'))}, "
            f"{_display(evaluation.get('wilson_95_high'))}]"
            if evaluation
            else "not_started"
        )
        batch = (
            f"{_display(training['global_batch_size'])}/"
            f"{_display(training['gradient_accumulation_steps'])}"
            if training["global_batch_size"] is not None
            else "—"
        )
        lines.append(
            "| "
            + " | ".join(
                _display(value)
                for value in (
                    training["task"],
                    training["dataset_revision"],
                    training["training_status"],
                    f"{_display(training['optimizer_steps'])}/{_display(training['target_optimizer_steps'])}",
                    batch,
                    training["final_train_loss"],
                    training["final_checkpoint"],
                    training["checkpoint_size_bytes"],
                    training["checkpoint_reload_status"],
                    evaluation.get("eval_episodes", "not_started"),
                    evaluation.get("successes", "not_started"),
                    evaluation.get("success_rate", "not_started"),
                    interval,
                    evaluation.get("rollout_video_directory", "not_started"),
                )
            )
            + " |"
        )
    _atomic_text(report_dir / "FINAL_STATUS.md", "\n".join(lines) + "\n")
    return training_json, evaluation_json


if __name__ == "__main__":
    refresh_reports(Workspace(Path("/workspace/challenge/robosyn-groot")))
