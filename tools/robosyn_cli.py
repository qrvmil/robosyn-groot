#!/usr/bin/env python3
"""Repository-controlled CLI for the ten-task RoboSyn × GR00T pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.generate_task_semantics import generate_semantics
from tools.evaluation import run_closed_loop_evaluation
from tools.cleanup import compact_evaluated_task, compact_verified_training
from tools.reporting import refresh_reports
from tools.pipeline import (
    TASKS,
    Workspace,
    dataset_materialization_missing_paths,
    evaluation_episodes_for_task,
    pipeline_environment,
    prepared_dataset_is_current,
    training_steps_for_task,
    write_train_launcher,
)
from tools.prepare_robosyn_for_groot import (
    policy_feature_allowlist,
    prepare_dataset,
    render_groot_config,
)


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _retry(label: str, operation, attempts: int = 6):
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            if attempt + 1 == attempts:
                raise
            delay = 2**attempt
            print(f"{label} failed ({type(exc).__name__}); retrying in {delay}s", flush=True)
            time.sleep(delay)


def _registry(workspace: Workspace) -> tuple[Path, dict[str, object]]:
    path = workspace.root / "configs/tasks/registry.json"
    registry = _json(path)
    changed = False
    for task, entry in registry.get("tasks", {}).items():
        if task not in TASKS:
            continue
        steps = training_steps_for_task(task)
        evaluation_episodes = evaluation_episodes_for_task(task)
        run_path = str(workspace.run_dir(task))
        if entry.get("training_max_steps") != steps:
            entry["training_max_steps"] = steps
            changed = True
        if entry.get("training_run_path") != run_path:
            entry["training_run_path"] = run_path
            changed = True
        if entry.get("evaluation_episodes") != evaluation_episodes:
            entry["evaluation_episodes"] = evaluation_episodes
            changed = True
    if changed:
        registry["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_json(path, registry)
    return path, registry


def raw_dataset_missing_paths(raw: Path) -> list[Path]:
    """Return required metadata, parquet, or video paths that are absent.

    An old interrupted ``snapshot_download(local_dir=...)`` can leave a valid
    ``meta/info.json`` alongside only a subset of the large media files.  The
    metadata file alone is therefore not a completeness marker.
    """
    missing = dataset_materialization_missing_paths(raw)
    tasks_path = raw / "meta/tasks.jsonl"
    if not tasks_path.is_file():
        missing.append(tasks_path)
    return missing


def _download_inputs(workspace: Workspace, entry: dict[str, object]) -> None:
    from huggingface_hub import snapshot_download

    env_cache = workspace.root / "cache/huggingface/hub"
    if not workspace.base_model_snapshot.is_dir():
        _retry(
            "base model download",
            lambda: snapshot_download(
                repo_id="nvidia/GR00T-N1.7-3B",
                revision="2fc962b973bccdd5d8ce4f67cc63b264d6886495",
                cache_dir=env_cache,
            ),
        )
    else:
        print(f"Base snapshot already present: {workspace.base_model_snapshot}")

    raw = Path(str(entry["raw_dataset_path"]))
    missing_raw_paths = raw_dataset_missing_paths(raw)
    if missing_raw_paths:
        if (raw / "meta/info.json").is_file():
            print(
                f"Raw dataset is incomplete ({len(missing_raw_paths)} missing/invalid paths); "
                "resuming immutable snapshot download.",
                flush=True,
            )
        raw.parent.mkdir(parents=True, exist_ok=True)
        _retry(
            f"dataset download {entry['hf_repository']}",
            lambda: snapshot_download(
                repo_id=str(entry["hf_repository"]),
                repo_type="dataset",
                revision=str(entry["resolved_hf_revision"]),
                local_dir=raw,
            ),
        )
        remaining = raw_dataset_missing_paths(raw)
        if remaining:
            preview = "\n".join(str(path) for path in remaining[:10])
            raise RuntimeError(
                f"immutable dataset snapshot remains incomplete: {len(remaining)} paths\n{preview}"
            )
    else:
        print(f"Raw dataset already present and complete: {raw}")


def bootstrap_task(workspace: Workspace, task: str) -> None:
    registry_path, registry = _registry(workspace)
    entry = registry["tasks"][task]
    cleanup_report = workspace.root / f"reports/cleanup/{task}.json"
    if entry.get("current_status") == "evaluation_complete" and cleanup_report.is_file():
        report = _json(cleanup_report)
        if report.get("status") == "pass":
            print(f"Bootstrap evidence archived after completed evaluation: {cleanup_report}")
            return
    revision = str(entry["resolved_hf_revision"])
    if len(revision) != 40:
        raise ValueError(f"task {task} has no immutable 40-character HF revision")
    _download_inputs(workspace, entry)

    raw = Path(str(entry["raw_dataset_path"]))
    source_manifest = {
        "task": task,
        "repo": entry["hf_repository"],
        "revision": revision,
        "raw_dataset_path": str(raw),
    }
    _write_json(Path(str(entry["source_manifest_path"])), source_manifest)

    semantics = generate_semantics(
        _json(workspace.root / "configs/robosyn_cobotmagic_semantics.json"),
        raw,
        task=task,
        repo=str(entry["hf_repository"]),
        revision=revision,
    )
    semantics_path = Path(str(entry["generated_semantics_path"]))
    config_path = Path(str(entry["generated_modality_config_path"]))
    _write_json(semantics_path, semantics)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(render_groot_config(semantics))

    allowed = policy_feature_allowlist(semantics)
    prepared = Path(str(entry["prepared_dataset_path"]))
    if not prepared_dataset_is_current(
        prepared, semantics, allowed_features=allowed, require_stats=False
    ):
        if prepared.exists():
            print(f"Removing invalid reproducible prepared dataset: {prepared}")
            shutil.rmtree(prepared)
        report = prepare_dataset(raw, prepared, semantics)
        _write_json(Path(str(entry["preparation_manifest_path"])), report)
    else:
        print(f"Prepared dataset is current: {prepared}")

    if not all(
        (prepared / "meta" / name).is_file()
        for name in ("stats.json", "relative_stats.json")
    ):
        command = [
            str(workspace.groot_link / ".venv/bin/python"),
            str(workspace.groot_link / "gr00t/data/stats.py"),
            "--dataset-path",
            str(prepared),
            "--embodiment-tag",
            "NEW_EMBODIMENT",
            "--modality-config-path",
            str(config_path),
        ]
        subprocess.run(
            command,
            cwd=workspace.groot_link,
            env=pipeline_environment(workspace),
            check=True,
        )
    else:
        print("Statistics already present and current.")

    roundtrip_path = workspace.root / f"data/manifests/{task}.roundtrip.json"
    loader_smoke_path = workspace.root / f"data/manifests/{task}.loader_smoke.json"
    subprocess.run(
        [
            str(workspace.groot_link / ".venv/bin/python"),
            str(workspace.root / "tools/dataset_smoke.py"),
            "--dataset",
            str(prepared),
            "--config",
            str(config_path),
            "--output",
            str(loader_smoke_path),
        ],
        cwd=workspace.root,
        env=pipeline_environment(workspace),
        check=True,
    )
    subprocess.run(
        [
            str(workspace.groot_link / ".venv/bin/python"),
            str(workspace.root / "tools/compare_actions.py"),
            "--dataset",
            str(prepared),
            "--config",
            str(config_path),
            "--samples",
            "16",
            "--seed",
            "17",
            "--output",
            str(roundtrip_path),
        ],
        cwd=workspace.root,
        env=pipeline_environment(workspace),
        check=True,
    )

    info = _json(prepared / "meta/info.json")
    entry.update(
        {
            "detected_state_key": semantics["state"]["original_key"],
            "detected_action_key": semantics["action"]["original_key"],
            "detected_camera_keys": {
                key: value["original_key"] for key, value in semantics["video"].items()
            },
            "state_dimension": 14,
            "action_dimension": 14,
            "state_names": semantics["evidence"]["joint_names"],
            "action_names": semantics["evidence"]["joint_names"],
            "current_status": "bootstrap_complete",
            "prepared_features": sorted(info["features"]),
        }
    )
    registry["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_json(registry_path, registry)
    print(f"BOOTSTRAP PASS: {task}")


def _latest_checkpoint(root: Path) -> Path | None:
    candidates = []
    for path in root.rglob("checkpoint-*") if root.exists() else ():
        if not path.is_dir():
            continue
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        candidates.append((step, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _last_loss(log_path: Path) -> float | None:
    text = log_path.read_text(errors="replace")
    # Transformers emits a run-average ``train_loss`` after the last actual
    # optimizer-step ``loss``. Report the latter when present.
    step_losses = re.findall(r"['\"]loss['\"]\s*:\s*([0-9.eE+-]+)", text)
    if step_losses:
        return float(step_losses[-1])
    summary_losses = re.findall(
        r"['\"]train_loss['\"]\s*:\s*([0-9.eE+-]+)", text
    )
    return float(summary_losses[-1]) if summary_losses else None


def _archive_training_evidence(
    workspace: Workspace, task: str, result: dict[str, object]
) -> None:
    checkpoint = Path(str(result["checkpoint"]))
    evidence = checkpoint / "robosyn_evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    registry_path, registry = _registry(workspace)
    entry = registry["tasks"][task]
    evidence_files = [
        workspace.prepared_dataset(task) / "meta/stats.json",
        workspace.prepared_dataset(task) / "meta/relative_stats.json",
        workspace.task_config(task),
        Path(str(entry["generated_semantics_path"])),
        Path(str(entry["source_manifest_path"])),
        Path(str(entry["preparation_manifest_path"])),
    ]
    for source in evidence_files:
        if not source.is_file():
            raise FileNotFoundError(f"checkpoint evidence is missing: {source}")
        shutil.copy2(source, evidence / source.name)
    _write_json(evidence / "task_registry_entry.json", entry)
    entry["current_status"] = "training_complete"
    entry["final_checkpoint"] = str(checkpoint)
    entry["checkpoint_sha256"] = result["checkpoint_sha256"]
    entry["training_run_path"] = str(workspace.run_dir(task))
    registry["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_json(registry_path, registry)


def build_live_pruner_command(
    workspace: Workspace,
    task: str,
    *,
    run_dir: Path,
    target_steps: int,
) -> list[str]:
    """Build the task-scoped disk-safety companion for a full training run."""
    if task not in TASKS:
        raise ValueError(f"unknown task: {task}")
    return [
        str(workspace.groot_link / ".venv/bin/python"),
        str(workspace.root / "tools/prune_live_checkpoints.py"),
        "--run-dir",
        str(run_dir),
        "--log",
        str(workspace.root / f"reports/cleanup/{task}.live.jsonl"),
        "--interval",
        "5",
        "--until-step",
        str(target_steps),
    ]


def wait_for_training_with_pruner(
    training_process: subprocess.Popen,
    pruner_process: subprocess.Popen,
    pruner_log_path: Path,
    *,
    poll_interval: float = 1.0,
) -> int:
    """Wait for training, aborting only when its disk-safety companion fails."""
    while training_process.poll() is None:
        pruner_returncode = pruner_process.poll()
        if pruner_returncode is not None and pruner_returncode != 0:
            training_process.terminate()
            try:
                training_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                training_process.kill()
                training_process.wait(timeout=10)
            raise RuntimeError(
                f"live checkpoint pruner exited {pruner_returncode} before "
                f"training completed; see {pruner_log_path}"
            )
        time.sleep(poll_interval)
    return int(training_process.returncode)


def run_training(workspace: Workspace, task: str, *, smoke: bool) -> None:
    _, registry = _registry(workspace)
    entry = registry["tasks"][task]
    run_root = workspace.run_dir(task)
    run_dir = run_root / "smoke" if smoke else run_root
    result_path = run_dir / ("smoke_result.json" if smoke else "training_result.json")
    if result_path.is_file() and _json(result_path).get("status") == "pass":
        result = _json(result_path)
        if not smoke:
            log_path = run_dir / "stdout.log"
            reload_path = run_dir / "checkpoint_reload.json"
            if log_path.is_file():
                result["final_loss"] = _last_loss(log_path)
            if reload_path.is_file():
                reload_result = _json(reload_path)
                result["checkpoint_sha256"] = reload_result["sha256"]
                result["checkpoint_sha256_scope"] = reload_result.get(
                    "sha256_scope", "model_safetensors"
                )
                result["reload_status"] = reload_result["reload_status"]
            _write_json(result_path, result)
            checkpoint = Path(str(result["checkpoint"]))
            evidence = checkpoint / "robosyn_evidence/task_registry_entry.json"
            if not evidence.is_file():
                _archive_training_evidence(workspace, task, result)
        print(f"{'Smoke' if smoke else 'Training'} already complete: {result_path}")
        return
    if entry["current_status"] not in {
        "bootstrap_complete",
        "training_complete",
        "evaluation_complete",
    }:
        bootstrap_task(workspace, task)

    attempts = [(16, 2), (8, 4)] if smoke else [(16, 2)]
    target_steps = 1 if smoke else training_steps_for_task(task)
    last_error = None
    for batch, accumulation in attempts:
        if smoke and run_dir.exists():
            for child in (run_dir / "checkpoints",):
                if child.exists():
                    shutil.rmtree(child)
        command_path, manifest_path = write_train_launcher(
            workspace,
            task,
            run_dir=run_dir,
            max_steps=target_steps,
            global_batch_size=batch,
            gradient_accumulation_steps=accumulation,
        )
        manifest = _json(manifest_path)
        manifest["status"] = "running"
        _write_json(manifest_path, manifest)
        log_path = run_dir / "stdout.log"
        # Preserve the pre-interruption trace when the launcher resumes from a
        # saved optimizer state; fresh attempts still start a clean log.
        log_mode = "a" if manifest.get("resume_from_checkpoint") else "w"
        pruner_process = None
        pruner_log = None
        with log_path.open(log_mode) as log:
            if log_mode == "a":
                log.write(
                    f"\n=== RESUME {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                    f"from {manifest['resume_checkpoint']} ===\n"
                )
            if not smoke:
                pruner_log = (run_dir / "live_checkpoint_pruner.log").open("a")
                pruner_process = subprocess.Popen(
                    build_live_pruner_command(
                        workspace,
                        task,
                        run_dir=run_dir,
                        target_steps=target_steps,
                    ),
                    cwd=workspace.root,
                    env=pipeline_environment(workspace),
                    stdout=pruner_log,
                    stderr=subprocess.STDOUT,
                )
            training_process = subprocess.Popen(
                [str(command_path)],
                cwd=workspace.root,
                env=pipeline_environment(workspace),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                if pruner_process is None:
                    returncode = training_process.wait()
                else:
                    returncode = wait_for_training_with_pruner(
                        training_process,
                        pruner_process,
                        run_dir / "live_checkpoint_pruner.log",
                    )
                if pruner_process is not None:
                    if returncode == 0:
                        try:
                            pruner_returncode = pruner_process.wait(timeout=30)
                        except subprocess.TimeoutExpired as error:
                            raise RuntimeError(
                                "live checkpoint pruner did not validate the final checkpoint"
                            ) from error
                        if pruner_returncode != 0:
                            raise RuntimeError(
                                f"live checkpoint pruner exited {pruner_returncode}; "
                                f"see {run_dir / 'live_checkpoint_pruner.log'}"
                            )
                    else:
                        pruner_process.terminate()
                        pruner_process.wait(timeout=10)
            finally:
                if training_process.poll() is None:
                    training_process.terminate()
                    try:
                        training_process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        training_process.kill()
                        training_process.wait(timeout=10)
                if pruner_process is not None and pruner_process.poll() is None:
                    pruner_process.terminate()
                    try:
                        pruner_process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pruner_process.kill()
                        pruner_process.wait(timeout=10)
                if pruner_log is not None:
                    pruner_log.close()
        if returncode != 0:
            log_text = log_path.read_text(errors="replace")
            last_error = f"training exit {returncode}"
            if smoke and batch == 16 and (
                "CUDA out of memory" in log_text or "OutOfMemoryError" in log_text
            ):
                print("Smoke OOM at batch=16/accum=2; retrying batch=8/accum=4")
                continue
            manifest["status"] = "failed"
            manifest["error"] = last_error
            _write_json(manifest_path, manifest)
            raise RuntimeError(f"{last_error}; see {log_path}")

        checkpoint = _latest_checkpoint(run_dir / "checkpoints")
        if checkpoint is None:
            raise RuntimeError(f"training exited 0 but produced no checkpoint under {run_dir}")
        if checkpoint.name != f"checkpoint-{target_steps}":
            raise RuntimeError(
                f"training exited 0 but latest checkpoint is {checkpoint.name}; "
                f"expected checkpoint-{target_steps}"
            )
        reload_path = run_dir / "checkpoint_reload.json"
        reload_log = run_dir / "checkpoint_reload.log"
        with reload_log.open("w") as log:
            reload_process = subprocess.run(
                [
                    str(workspace.groot_link / ".venv/bin/python"),
                    str(workspace.root / "tools/reload_checkpoint.py"),
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(reload_path),
                ],
                cwd=workspace.groot_link,
                env=pipeline_environment(workspace),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if reload_process.returncode != 0:
            raise RuntimeError(f"checkpoint reload failed; see {reload_log}")
        reload_result = _json(reload_path)
        result = {
            "status": "pass",
            "task": task,
            "smoke": smoke,
            "global_batch_size": batch,
            "gradient_accumulation_steps": accumulation,
            "effective_batch_size": batch * accumulation,
            "optimizer_steps": target_steps,
            "final_loss": _last_loss(log_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": reload_result["sha256"],
            "checkpoint_sha256_scope": reload_result.get(
                "sha256_scope", "model_safetensors"
            ),
            "reload_status": reload_result["reload_status"],
        }
        _write_json(result_path, result)
        manifest.update(result)
        manifest["status"] = "completed"
        _write_json(manifest_path, manifest)
        if smoke:
            shutil.rmtree(run_dir / "checkpoints")
            result["temporary_checkpoint_removed"] = True
            _write_json(result_path, result)
        else:
            _archive_training_evidence(workspace, task, result)
            compact_verified_training(workspace, task)
            refresh_reports(workspace)
        print(f"{'SMOKE' if smoke else 'TRAINING'} PASS: {task}")
        return
    raise RuntimeError(last_error or "training failed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="robosyn")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list-tasks")
    bootstrap = subparsers.add_parser("bootstrap")
    bootstrap.add_argument("task", choices=TASKS)
    prefetch = subparsers.add_parser("prefetch")
    prefetch.add_argument("task", choices=TASKS)
    for name in ("smoke", "train", "eval"):
        child = subparsers.add_parser(name)
        child.add_argument("task", choices=TASKS)
    for name in ("train-all", "eval-all", "status"):
        subparsers.add_parser(name)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = Workspace(
        Path(os.environ.get("ROBOSYN_WORK_ROOT", ROOT)),
        Path(os.environ.get("ROBOSYN_GROOT_SRC", "/opt/robosyn/Isaac-GR00T")),
    )
    workspace.initialize()
    if args.command == "list-tasks":
        print("\n".join(TASKS))
        return 0
    if args.command == "bootstrap":
        bootstrap_task(workspace, args.task)
        return 0
    if args.command == "prefetch":
        _, registry = _registry(workspace)
        _download_inputs(workspace, registry["tasks"][args.task])
        print(f"PREFETCH PASS: {args.task}")
        return 0
    if args.command == "smoke":
        run_training(workspace, args.task, smoke=True)
        return 0
    if args.command == "train":
        run_training(workspace, args.task, smoke=False)
        return 0
    if args.command == "eval":
        smoke_summary = run_closed_loop_evaluation(workspace, args.task, smoke=True)
        print(f"ROLLOUT SMOKE PASS: {smoke_summary}")
        summary = run_closed_loop_evaluation(workspace, args.task, smoke=False)
        registry_path, registry = _registry(workspace)
        entry = registry["tasks"][args.task]
        entry["current_status"] = "evaluation_complete"
        entry["evaluation_path"] = str(workspace.root / "eval" / args.task)
        registry["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_json(registry_path, registry)
        cleanup_report = compact_evaluated_task(workspace, args.task)
        refresh_reports(workspace)
        print(f"COMPACTION PASS: {cleanup_report}")
        print(f"EVALUATION PASS: {summary}")
        return 0
    if args.command == "status":
        _, registry = _registry(workspace)
        for task in TASKS:
            print(f"{task}\t{registry['tasks'][task]['current_status']}")
        return 0
    if args.command in {"train-all", "eval-all"}:
        phases = "all" if args.command == "train-all" else "eval"
        return subprocess.run(
            [
                str(workspace.groot_link / ".venv/bin/python"),
                str(workspace.root / "tools/run_all_tasks.py"),
                "--phases",
                phases,
                "--continuous",
            ],
            cwd=workspace.root,
            env=pipeline_environment(workspace),
        ).returncode
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
