import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.pipeline import (
    BASE_MODEL_REVISION,
    Workspace,
    build_train_command,
    evaluation_episodes_for_task,
    prepared_dataset_is_current,
    pipeline_environment,
    training_steps_for_task,
    write_train_launcher,
)
from tools.robosyn_cli import (
    _last_loss,
    _registry,
    build_live_pruner_command,
    build_parser,
    raw_dataset_missing_paths,
    wait_for_training_with_pruner,
)


def test_workspace_initialization_creates_pinned_groot_symlink(tmp_path: Path):
    work = tmp_path / "workspace"
    groot = tmp_path / "Isaac-GR00T"
    (groot / ".venv/bin").mkdir(parents=True)
    (groot / ".venv/bin/python").write_text("python")
    workspace = Workspace(work, groot)

    workspace.initialize()

    link = work / "repos/Isaac-GR00T"
    assert link.is_symlink()
    assert link.resolve() == groot.resolve()


def test_pipeline_environment_uses_resilient_online_hub_timeouts(tmp_path: Path):
    env = pipeline_environment(Workspace(tmp_path, tmp_path / "installed-groot"))

    assert int(env["HF_HUB_ETAG_TIMEOUT"]) >= 30
    assert int(env["HF_HUB_DOWNLOAD_TIMEOUT"]) >= 60
    assert env["HF_HUB_DISABLE_XET"] == "1"
    assert "HF_HUB_OFFLINE" not in env


def test_raw_dataset_completeness_rejects_missing_camera_materialization(tmp_path: Path):
    raw = tmp_path / "raw"
    (raw / "meta").mkdir(parents=True)
    (raw / "meta/info.json").write_text(
        json.dumps(
            {
                "total_episodes": 1,
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": {
                    camera: {"dtype": "video", "shape": [480, 640, 3]}
                    for camera in (
                        "cam_high.color",
                        "cam_left_wrist.color",
                        "cam_right_wrist.color",
                    )
                },
            }
        )
    )
    (raw / "meta/episodes.jsonl").write_text('{"episode_index": 0}\n')
    (raw / "meta/tasks.jsonl").write_text('{"task_index": 0, "task": "test"}\n')
    parquet = raw / "data/chunk-000/episode_000000.parquet"
    parquet.parent.mkdir(parents=True)
    parquet.write_bytes(b"parquet")
    front = raw / "videos/chunk-000/cam_high.color/episode_000000.mp4"
    front.parent.mkdir(parents=True)
    front.write_bytes(b"video")

    missing = raw_dataset_missing_paths(raw)

    assert raw / "videos/chunk-000/cam_left_wrist.color/episode_000000.mp4" in missing
    assert raw / "videos/chunk-000/cam_right_wrist.color/episode_000000.mp4" in missing
    assert front not in missing


def test_each_task_launcher_starts_from_same_immutable_base_snapshot(tmp_path: Path):
    workspace = Workspace(tmp_path, tmp_path / "installed-groot")

    drawer = build_train_command(workspace, "drawer_open_place", max_steps=2000)
    water = build_train_command(workspace, "water_pouring", max_steps=2000)

    for command in (drawer, water):
        base_index = command.index("--base-model-path") + 1
        assert command[base_index] == str(workspace.base_model_snapshot)
        assert BASE_MODEL_REVISION in command[base_index]
        assert command[command.index("--global-batch-size") + 1] == "16"
        assert command[command.index("--gradient-accumulation-steps") + 1] == "2"
        assert command[command.index("--save-total-limit") + 1] == "2"
    assert drawer[drawer.index("--dataset-path") + 1] != water[
        water.index("--dataset-path") + 1
    ]


def test_remaining_tasks_use_4000_steps_but_completed_drawer_stays_2000(tmp_path: Path):
    workspace = Workspace(tmp_path, tmp_path / "installed-groot")

    assert training_steps_for_task("drawer_open_place") == 2000
    assert workspace.run_dir("drawer_open_place").name.endswith("_2k")
    for task in (
        "click_bell",
        "mixer_operating",
        "table_rearrangement",
        "manipulate_pipette",
        "water_pouring",
        "item_assembly",
        "handle_basket",
        "items_handover",
        "sample_loading",
    ):
        assert training_steps_for_task(task) == 4000
        assert workspace.run_dir(task).name.endswith("_4k")
    water = build_train_command(
        workspace,
        "water_pouring",
        max_steps=training_steps_for_task("water_pouring"),
    )
    assert water[water.index("--experiment-name") + 1].endswith("_4k")
    assert water[water.index("--max-steps") + 1] == "4000"


def test_registry_self_heals_training_step_contract(tmp_path: Path):
    workspace = Workspace(tmp_path, tmp_path / "installed-groot")
    path = tmp_path / "configs/tasks/registry.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "tasks": {
                    "drawer_open_place": {"training_run_path": "old"},
                    "mixer_operating": {"training_run_path": "old"},
                }
            }
        )
    )

    _, registry = _registry(workspace)

    assert registry["tasks"]["drawer_open_place"]["training_max_steps"] == 2000
    assert registry["tasks"]["mixer_operating"]["training_max_steps"] == 4000
    assert registry["tasks"]["mixer_operating"]["training_run_path"].endswith(
        "mixer_operating_sim_baseline_4k"
    )
    assert registry["tasks"]["drawer_open_place"]["evaluation_episodes"] == 50
    assert registry["tasks"]["mixer_operating"]["evaluation_episodes"] == 50


def test_evaluation_episode_contract_keeps_completed_runs_and_shortens_future_tasks():
    for task in ("drawer_open_place", "click_bell", "mixer_operating"):
        assert evaluation_episodes_for_task(task) == 50
    for task in (
        "table_rearrangement",
        "manipulate_pipette",
        "water_pouring",
        "item_assembly",
        "handle_basket",
        "items_handover",
        "sample_loading",
    ):
        assert evaluation_episodes_for_task(task) == 20


def test_prefetch_cli_requires_a_known_task():
    parsed = build_parser().parse_args(["prefetch", "table_rearrangement"])
    assert parsed.command == "prefetch"
    assert parsed.task == "table_rearrangement"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["prefetch", "not_a_task"])


def test_prepared_dataset_current_check_is_idempotent(tmp_path: Path):
    prepared = tmp_path / "prepared"
    (prepared / "meta").mkdir(parents=True)
    semantics = {"state": {"original_key": "observation.state"}}
    digest = hashlib.sha256(
        json.dumps(semantics, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (prepared / "meta/preparation_manifest.json").write_text(
        json.dumps({"semantics_sha256": digest})
    )
    (prepared / "meta/info.json").write_text(
        json.dumps(
            {
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [14]},
                    "action": {"dtype": "float32", "shape": [14]},
                }
            }
        )
    )
    (prepared / "meta/stats.json").write_text("{}")
    (prepared / "meta/relative_stats.json").write_text("{}")

    assert prepared_dataset_is_current(
        prepared,
        semantics,
        allowed_features={"observation.state", "action"},
        require_stats=True,
    )


def test_prepared_dataset_current_rejects_missing_materialized_camera(tmp_path: Path):
    prepared = tmp_path / "prepared"
    (prepared / "meta").mkdir(parents=True)
    semantics = {"state": {"original_key": "observation.state"}}
    digest = hashlib.sha256(
        json.dumps(semantics, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (prepared / "meta/preparation_manifest.json").write_text(
        json.dumps({"semantics_sha256": digest})
    )
    (prepared / "meta/info.json").write_text(
        json.dumps(
            {
                "total_episodes": 1,
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [14]},
                    "action": {"dtype": "float32", "shape": [14]},
                    "cam_high.color": {"dtype": "video", "shape": [480, 640, 3]},
                },
            }
        )
    )
    (prepared / "meta/episodes.jsonl").write_text('{"episode_index": 0}\n')
    parquet = prepared / "data/chunk-000/episode_000000.parquet"
    parquet.parent.mkdir(parents=True)
    parquet.write_bytes(b"parquet")

    assert not prepared_dataset_is_current(
        prepared,
        semantics,
        allowed_features={"observation.state", "action", "cam_high.color"},
        require_stats=False,
    )


@pytest.mark.parametrize("task", ["unknown", "", "click-bell"])
def test_train_command_rejects_unknown_task(tmp_path: Path, task: str):
    workspace = Workspace(tmp_path, tmp_path / "installed-groot")
    with pytest.raises(ValueError, match="unknown task"):
        build_train_command(workspace, task, max_steps=1)


def test_launcher_generation_records_base_and_effective_batch(tmp_path: Path):
    workspace = Workspace(tmp_path, tmp_path / "installed-groot")
    run_dir = workspace.run_dir("drawer_open_place") / "smoke"

    command_path, manifest_path = write_train_launcher(
        workspace,
        "drawer_open_place",
        run_dir=run_dir,
        max_steps=1,
        global_batch_size=16,
        gradient_accumulation_steps=2,
    )

    manifest = json.loads(manifest_path.read_text())
    assert command_path.is_file()
    assert command_path.stat().st_mode & 0o111
    assert str(workspace.base_model_snapshot) in command_path.read_text()
    assert manifest["base_model_path"] == str(workspace.base_model_snapshot)
    assert manifest["base_model_revision"] == BASE_MODEL_REVISION
    assert manifest["effective_batch_size"] == 32
    assert manifest["max_steps"] == 1


def test_full_training_live_pruner_launcher_is_task_scoped(tmp_path: Path):
    workspace = Workspace(tmp_path, tmp_path / "installed-groot")
    run_dir = workspace.run_dir("water_pouring")

    command = build_live_pruner_command(
        workspace, "water_pouring", run_dir=run_dir, target_steps=4000
    )

    assert command == [
        str(workspace.groot_link / ".venv/bin/python"),
        str(workspace.root / "tools/prune_live_checkpoints.py"),
        "--run-dir",
        str(run_dir),
        "--log",
        str(workspace.root / "reports/cleanup/water_pouring.live.jsonl"),
        "--interval",
        "5",
        "--until-step",
        "4000",
    ]


def test_training_continues_when_pruner_reaches_final_step_first(tmp_path: Path):
    training = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.2)"]
    )
    pruner = subprocess.Popen([sys.executable, "-c", "raise SystemExit(0)"])

    returncode = wait_for_training_with_pruner(
        training, pruner, tmp_path / "pruner.log", poll_interval=0.01
    )

    assert returncode == 0
    assert training.returncode == 0
    assert pruner.returncode == 0


def test_training_is_stopped_when_live_pruner_crashes(tmp_path: Path):
    training = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    pruner = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"])

    with pytest.raises(RuntimeError, match="live checkpoint pruner exited 7"):
        wait_for_training_with_pruner(
            training, pruner, tmp_path / "pruner.log", poll_interval=0.01
        )

    assert training.poll() is not None


def test_launcher_resumes_existing_optimizer_checkpoint(tmp_path: Path):
    workspace = Workspace(tmp_path, tmp_path / "installed-groot")
    run_dir = workspace.run_dir("drawer_open_place")
    checkpoint = (
        run_dir
        / "checkpoints/drawer_open_place_sim_baseline_2k/checkpoint-250"
    )
    checkpoint.mkdir(parents=True)
    (checkpoint / "optimizer.pt").write_text("resume-state")

    command_path, manifest_path = write_train_launcher(
        workspace,
        "drawer_open_place",
        run_dir=run_dir,
        max_steps=2000,
        global_batch_size=16,
        gradient_accumulation_steps=2,
    )

    manifest = json.loads(manifest_path.read_text())
    assert "--resume-from-checkpoint" in command_path.read_text()
    assert manifest["resume_from_checkpoint"] is True
    assert manifest["resume_checkpoint"] == str(checkpoint)


def test_last_loss_accepts_transformers_train_loss_summary(tmp_path: Path):
    log = tmp_path / "stdout.log"
    log.write_text(
        "{'train_runtime': 77.2, 'train_loss': 1.3747591972351074}\n"
    )

    assert _last_loss(log) == 1.3747591972351074


def test_last_loss_prefers_final_optimizer_step_over_run_average(tmp_path: Path):
    log = tmp_path / "stdout.log"
    log.write_text("{'loss': 0.0214}\n{'train_loss': 0.1107}\n")

    assert _last_loss(log) == 0.0214
