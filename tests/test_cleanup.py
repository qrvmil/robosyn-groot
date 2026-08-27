import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.cleanup import (
    compact_evaluated_task,
    compact_verified_training,
    prune_superseded_live_checkpoints,
)
from tools.pipeline import Workspace
from tools.prune_live_checkpoints import main as live_pruner_main


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_completed_task(
    tmp_path: Path,
    *,
    task: str = "drawer_open_place",
    evaluation_episodes: int = 50,
) -> tuple[Workspace, Path, Path]:
    workspace = Workspace(tmp_path, tmp_path / "installed-groot")
    final = workspace.run_dir(task) / "checkpoints/run/checkpoint-4000"
    old = final.parent / "checkpoint-1750"
    for checkpoint in (final, old):
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.safetensors").write_bytes(b"model")
        (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    (final.parent / "model.safetensors").write_bytes(b"model")
    evidence = final / "robosyn_evidence"
    evidence.mkdir()
    for name in ("stats.json", "relative_stats.json", "task_registry_entry.json"):
        (evidence / name).write_text("{}")
    _write(
        workspace.run_dir(task) / "training_result.json",
        {"status": "pass", "reload_status": "pass", "checkpoint": str(final), "checkpoint_sha256": "abc"},
    )
    _write(
        tmp_path / f"eval/{task}/summary.json",
        {
            "number_of_episodes": evaluation_episodes,
            "requested_episodes": evaluation_episodes,
            "seed_list": list(range(evaluation_episodes)),
            "inference_latency_scope": "model_rpc_only",
        },
    )
    _write(
        tmp_path / "configs/tasks/registry.json",
        {"tasks": {task: {"evaluation_episodes": evaluation_episodes}}},
    )
    for dataset in (workspace.raw_dataset(task), workspace.prepared_dataset(task)):
        dataset.mkdir(parents=True)
        (dataset / "payload").write_bytes(b"data")
    return workspace, final, old


def test_cleanup_requires_full_evaluation_and_preserves_final_model(tmp_path: Path):
    workspace, final, old = make_completed_task(tmp_path)
    report = compact_evaluated_task(workspace, "drawer_open_place")
    assert report.is_file()
    assert final.is_dir() and (final / "model.safetensors").is_file()
    assert not (final / "optimizer.pt").exists()
    assert not old.exists()
    assert not workspace.raw_dataset("drawer_open_place").exists()
    assert not workspace.prepared_dataset("drawer_open_place").exists()
    exported = final.parent / "model.safetensors"
    assert exported.is_file()
    assert exported.stat().st_ino == (final / "model.safetensors").stat().st_ino
    first = json.loads(report.read_text())
    compact_evaluated_task(workspace, "drawer_open_place")
    second = json.loads(report.read_text())
    assert second["actions"] == first["actions"]
    assert second["bytes_reclaimed"] == first["bytes_reclaimed"]


def test_cleanup_refuses_partial_evaluation(tmp_path: Path):
    workspace, _, _ = make_completed_task(tmp_path)
    _write(tmp_path / "eval/drawer_open_place/summary.json", {"number_of_episodes": 1})
    with pytest.raises(RuntimeError, match="before full evaluation"):
        compact_evaluated_task(workspace, "drawer_open_place")


def test_cleanup_accepts_approved_twenty_episode_task_target(tmp_path: Path):
    workspace, final, _ = make_completed_task(
        tmp_path, task="table_rearrangement", evaluation_episodes=20
    )

    report = compact_evaluated_task(workspace, "table_rearrangement")

    assert report.is_file()
    assert final.is_dir()


def test_cleanup_refuses_nineteen_of_twenty_episodes(tmp_path: Path):
    workspace, _, _ = make_completed_task(
        tmp_path, task="table_rearrangement", evaluation_episodes=20
    )
    _write(
        tmp_path / "eval/table_rearrangement/summary.json",
        {
            "number_of_episodes": 19,
            "requested_episodes": 20,
            "seed_list": list(range(20)),
            "inference_latency_scope": "model_rpc_only",
        },
    )

    with pytest.raises(RuntimeError, match="before full evaluation"):
        compact_evaluated_task(workspace, "table_rearrangement")


def test_cleanup_refuses_legacy_policy_chunk_latency(tmp_path: Path):
    workspace, _, _ = make_completed_task(tmp_path)
    _write(
        tmp_path / "eval/drawer_open_place/summary.json",
        {
            "number_of_episodes": 50,
            "requested_episodes": 50,
            "seed_list": list(range(50)),
            "inference_latency_scope": "policy_eval_including_environment",
        },
    )
    with pytest.raises(RuntimeError, match="model-RPC-only"):
        compact_evaluated_task(workspace, "drawer_open_place")


def test_verified_training_compaction_preserves_resume_state_and_datasets(tmp_path: Path):
    workspace, final, old = make_completed_task(tmp_path)

    report = compact_verified_training(workspace, "drawer_open_place")

    assert report.is_file()
    assert final.is_dir()
    assert (final / "optimizer.pt").is_file()
    assert not old.exists()
    assert workspace.raw_dataset("drawer_open_place").is_dir()
    assert workspace.prepared_dataset("drawer_open_place").is_dir()
    exported = final.parent / "model.safetensors"
    assert exported.stat().st_ino == (final / "model.safetensors").stat().st_ino


def _make_live_checkpoint(parent: Path, step: int, *, complete: bool = True) -> Path:
    checkpoint = parent / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "max_steps": 4000})
    )
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "one": "model-00001-of-00002.safetensors",
                    "two": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    (checkpoint / "model-00001-of-00002.safetensors").write_bytes(b"one")
    (checkpoint / "model-00002-of-00002.safetensors").write_bytes(b"two")
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
    (checkpoint / "rng_state.pth").write_bytes(b"rng")
    (checkpoint / "config.json").write_text("{}")
    if not complete:
        (checkpoint / "optimizer.pt").unlink()
    return checkpoint


def test_live_pruner_keeps_newest_complete_resume_checkpoint(tmp_path: Path):
    run_root = tmp_path / "runs/task_sim_baseline_4k"
    checkpoint_parent = run_root / "checkpoints/task_sim_baseline_4k"
    old = _make_live_checkpoint(checkpoint_parent, 2000)
    newest = _make_live_checkpoint(checkpoint_parent, 2250)
    old_bytes = sum(path.stat().st_size for path in old.rglob("*") if path.is_file())

    actions = prune_superseded_live_checkpoints(run_root)

    assert newest.is_dir()
    assert not old.exists()
    assert actions == [
        {
            "action": "remove_superseded_live_checkpoint",
            "path": str(old),
            "bytes": old_bytes,
            "preserved_checkpoint": str(newest),
            "preserved_step": 2250,
        }
    ]
    assert prune_superseded_live_checkpoints(run_root) == []


def test_live_pruner_refuses_incomplete_newest_checkpoint(tmp_path: Path):
    run_root = tmp_path / "runs/task_sim_baseline_4k"
    checkpoint_parent = run_root / "checkpoints/task_sim_baseline_4k"
    old = _make_live_checkpoint(checkpoint_parent, 2000)
    newest = _make_live_checkpoint(checkpoint_parent, 2250, complete=False)

    with pytest.raises(RuntimeError, match="incomplete live checkpoint"):
        prune_superseded_live_checkpoints(run_root)

    assert old.is_dir()
    assert newest.is_dir()


def test_live_pruner_cli_once_writes_machine_readable_action_log(tmp_path: Path):
    run_root = tmp_path / "runs/task_sim_baseline_4k"
    checkpoint_parent = run_root / "checkpoints/task_sim_baseline_4k"
    old = _make_live_checkpoint(checkpoint_parent, 2000)
    newest = _make_live_checkpoint(checkpoint_parent, 2250)
    log = tmp_path / "pruner.jsonl"

    result = live_pruner_main(
        ["--run-dir", str(run_root), "--log", str(log), "--once"]
    )

    assert result == 0
    assert newest.is_dir()
    assert not old.exists()
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["status"] == "pruned"
    assert records[0]["actions"][0]["preserved_step"] == 2250


def test_live_pruner_script_runs_directly_from_repository_root():
    result = subprocess.run(
        [sys.executable, "tools/prune_live_checkpoints.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--run-dir" in result.stdout
