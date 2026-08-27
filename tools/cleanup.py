"""Validated post-evaluation compaction for the 250 GB workspace."""

from __future__ import annotations

import json
import filecmp
import os
import re
import shutil
import time
from pathlib import Path

from tools.pipeline import Workspace, evaluation_episodes_for_task


_CHECKPOINT_NAME = re.compile(r"checkpoint-(\d+)$")


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _validated_training(
    workspace: Workspace, task: str
) -> tuple[dict[str, object], Path, Path]:
    training_path = workspace.run_dir(task) / "training_result.json"
    training = _read(training_path)
    if training.get("status") != "pass" or training.get("reload_status") != "pass":
        raise RuntimeError(f"refusing cleanup without verified training: {training_path}")

    run_root = workspace.run_dir(task).resolve()
    checkpoint = Path(str(training["checkpoint"])).resolve()
    if not checkpoint.is_dir() or not checkpoint.is_relative_to(run_root):
        raise RuntimeError(f"refusing checkpoint outside task run: {checkpoint}")
    evidence = checkpoint / "robosyn_evidence"
    required_evidence = ("stats.json", "relative_stats.json", "task_registry_entry.json")
    if not all((evidence / name).is_file() for name in required_evidence):
        raise RuntimeError(f"refusing cleanup before checkpoint evidence archive: {evidence}")
    return training, run_root, checkpoint


def _remove_nonfinal_checkpoints(
    run_root: Path, checkpoint: Path
) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = []
    for candidate in sorted(checkpoint.parent.glob("checkpoint-*")):
        if candidate.resolve() == checkpoint:
            continue
        if candidate.is_dir() and candidate.resolve().is_relative_to(run_root):
            size = sum(path.stat().st_size for path in candidate.rglob("*") if path.is_file())
            shutil.rmtree(candidate)
            actions.append(
                {"action": "remove_resume_checkpoint", "path": str(candidate), "bytes": size}
            )
    return actions


def _validate_live_resume_checkpoint(checkpoint: Path) -> int:
    """Prove a trainer checkpoint is complete enough to replace an older resume."""
    match = _CHECKPOINT_NAME.fullmatch(checkpoint.name)
    if match is None:
        raise RuntimeError(f"invalid live checkpoint name: {checkpoint}")
    step = int(match.group(1))
    try:
        trainer_state = _read(checkpoint / "trainer_state.json")
        index = _read(checkpoint / "model.safetensors.index.json")
        shard_names = set(index["weight_map"].values())
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"incomplete live checkpoint: {checkpoint}") from error
    if int(trainer_state.get("global_step", -1)) != step or not shard_names:
        raise RuntimeError(f"incomplete live checkpoint: {checkpoint}")
    required = {
        "config.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        *shard_names,
    }
    for name in required:
        path = checkpoint / str(name)
        if (
            Path(str(name)).name != str(name)
            or not path.is_file()
            or path.stat().st_size <= 0
            or not path.resolve().is_relative_to(checkpoint.resolve())
        ):
            raise RuntimeError(f"incomplete live checkpoint: {checkpoint}")
    return step


def prune_superseded_live_checkpoints(run_root: Path) -> list[dict[str, object]]:
    """Keep only the newest validated resume checkpoint in an active task run.

    If the newest save is incomplete, no prior checkpoint is touched. This is
    intentionally separate from post-training compaction because it runs while
    the trainer still owns the run directory.
    """
    run_root = Path(run_root).resolve()
    if not run_root.is_dir():
        raise RuntimeError(f"live run directory does not exist: {run_root}")
    candidates = [
        path
        for path in (run_root / "checkpoints").glob("*/checkpoint-*")
        if path.is_dir() and not path.is_symlink() and _CHECKPOINT_NAME.fullmatch(path.name)
    ]
    if not candidates:
        return []
    parents = {path.parent.resolve() for path in candidates}
    if len(parents) != 1:
        raise RuntimeError(f"ambiguous live checkpoint roots under {run_root}")
    if any(not path.resolve().is_relative_to(run_root) for path in candidates):
        raise RuntimeError(f"live checkpoint escapes run directory: {run_root}")
    candidates.sort(key=lambda path: int(_CHECKPOINT_NAME.fullmatch(path.name).group(1)))
    newest = candidates[-1]
    preserved_step = _validate_live_resume_checkpoint(newest)
    if len(candidates) == 1:
        return []
    actions: list[dict[str, object]] = []
    for old in candidates[:-1]:
        size = sum(path.stat().st_size for path in old.rglob("*") if path.is_file())
        shutil.rmtree(old)
        actions.append(
            {
                "action": "remove_superseded_live_checkpoint",
                "path": str(old),
                "bytes": size,
                "preserved_checkpoint": str(newest),
                "preserved_step": preserved_step,
            }
        )
    return actions


def _deduplicate_final_export(checkpoint: Path) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = []
    checkpoint_root = checkpoint.parent
    for exported in sorted(checkpoint_root.glob("model*.safetensors")):
        canonical = checkpoint / exported.name
        if not canonical.is_file() or exported.stat().st_ino == canonical.stat().st_ino:
            continue
        if exported.stat().st_size != canonical.stat().st_size or not filecmp.cmp(
            exported, canonical, shallow=False
        ):
            continue
        size = exported.stat().st_size
        temporary = exported.with_name(exported.name + ".hardlink.tmp")
        if temporary.exists():
            temporary.unlink()
        os.link(canonical, temporary)
        os.replace(temporary, exported)
        actions.append(
            {
                "action": "deduplicate_final_export",
                "path": str(exported),
                "canonical": str(canonical),
                "bytes": size,
            }
        )
    return actions


def compact_verified_training(workspace: Workspace, task: str) -> Path:
    """Safely reduce a verified run before evaluation without dropping resume state."""
    training, run_root, checkpoint = _validated_training(workspace, task)
    report_path = workspace.root / f"reports/cleanup/{task}.training.json"
    prior_actions: list[dict[str, object]] = []
    if report_path.is_file():
        prior = _read(report_path)
        if prior.get("status") == "pass" and prior.get("checkpoint") == str(checkpoint):
            prior_actions = list(prior.get("actions", []))

    actions = _remove_nonfinal_checkpoints(run_root, checkpoint)
    actions.extend(_deduplicate_final_export(checkpoint))
    report = {
        "status": "pass",
        "phase": "verified_training",
        "task": task,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": training["checkpoint_sha256"],
        "preserved": [str(checkpoint)],
        "actions": prior_actions + actions,
        "bytes_reclaimed": sum(
            int(action["bytes"]) for action in prior_actions + actions
        ),
    }
    _write(report_path, report)
    return report_path


def compact_evaluated_task(workspace: Workspace, task: str) -> Path:
    """Keep model weights/evidence/videos, remove reproducible bulky state."""
    summary_path = workspace.root / "eval" / task / "summary.json"
    summary = _read(summary_path)
    registry_path = workspace.root / "configs/tasks/registry.json"
    registry = _read(registry_path) if registry_path.is_file() else {"tasks": {}}
    entry = registry.get("tasks", {}).get(task, {})
    target = int(entry.get("evaluation_episodes", evaluation_episodes_for_task(task)))
    if (
        int(summary.get("number_of_episodes", 0)) != target
        or int(summary.get("requested_episodes", 0)) != target
        or summary.get("seed_list") != list(range(target))
    ):
        raise RuntimeError(f"refusing cleanup before full evaluation: {summary_path}")
    if summary.get("inference_latency_scope") != "model_rpc_only":
        raise RuntimeError(
            f"refusing cleanup without model-RPC-only latency evidence: {summary_path}"
        )
    training, run_root, checkpoint = _validated_training(workspace, task)

    report_path = workspace.root / f"reports/cleanup/{task}.json"
    prior_actions: list[dict[str, object]] = []
    if report_path.is_file():
        prior = _read(report_path)
        if prior.get("status") == "pass" and prior.get("checkpoint") == str(checkpoint):
            prior_actions = list(prior.get("actions", []))
    elif (workspace.root / f"reports/cleanup/{task}.training.json").is_file():
        prior = _read(workspace.root / f"reports/cleanup/{task}.training.json")
        if prior.get("status") == "pass" and prior.get("checkpoint") == str(checkpoint):
            prior_actions = list(prior.get("actions", []))
    actions: list[dict[str, object]] = []
    actions.extend(_remove_nonfinal_checkpoints(run_root, checkpoint))

    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        path = checkpoint / name
        if path.is_file():
            size = path.stat().st_size
            path.unlink()
            actions.append({"action": "remove_resume_state", "path": str(path), "bytes": size})

    # Transformers saves the same final model both at the run export root and
    # in checkpoint-N. Keep both usable paths while sharing identical bytes.
    actions.extend(_deduplicate_final_export(checkpoint))

    # Both snapshots are immutable and reproducible from the recorded HF SHA.
    # Their policy metadata/stats are archived with the final model first.
    for dataset in (workspace.raw_dataset(task), workspace.prepared_dataset(task)):
        if dataset.exists():
            resolved = dataset.resolve()
            data_root = (workspace.root / "data").resolve()
            if not resolved.is_relative_to(data_root):
                raise RuntimeError(f"refusing dataset outside workspace data root: {dataset}")
            size = sum(path.stat().st_size for path in dataset.rglob("*") if path.is_file())
            shutil.rmtree(dataset)
            actions.append({"action": "remove_reproducible_dataset", "path": str(dataset), "bytes": size})

    report = {
        "status": "pass",
        "task": task,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": training["checkpoint_sha256"],
        "preserved": [
            str(checkpoint),
            str(workspace.root / "eval" / task),
            str(workspace.root / f"data/manifests/{task}.source.json"),
            str(workspace.root / f"data/manifests/{task}.preparation.json"),
        ],
        "actions": prior_actions + actions,
        "bytes_reclaimed": sum(
            int(action["bytes"]) for action in prior_actions + actions
        ),
    }
    _write(report_path, report)
    return report_path
