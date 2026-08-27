import json
from pathlib import Path

import pytest

import tools.evaluation as evaluation

from tools.evaluation import (
    EMBODICHAIN_COMMIT,
    build_server_command,
    build_simulator_command,
    evaluation_environment,
    load_evaluation_spec,
    validate_rollout_artifacts,
)
from tools.pipeline import BASE_MODEL_REVISION, GROOT_COMMIT, Workspace


def make_verified_workspace(
    tmp_path: Path,
    *,
    task: str = "drawer_open_place",
    evaluation_episodes: int = 50,
) -> Workspace:
    workspace = Workspace(tmp_path, tmp_path / "installed-groot")
    checkpoint = (
        workspace.run_dir(task)
        / f"checkpoints/{task}_sim_baseline_4k/checkpoint-4000"
    )
    checkpoint.mkdir(parents=True)
    (workspace.run_dir(task) / "training_result.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "reload_status": "pass",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": "abc123",
            }
        )
    )
    (tmp_path / "configs/tasks").mkdir(parents=True)
    (tmp_path / "configs/tasks/registry.json").write_text(
        json.dumps(
            {
                "base_model_revision": BASE_MODEL_REVISION,
                "tasks": {
                    task: {
                        "hf_repository": f"RoboSynChallenge/cobotmagic_Sim_{task}",
                        "resolved_hf_revision": "6fdf6ee1ed74bbbbc0216d507c8a19c8994a15c3",
                        "evaluation_episodes": evaluation_episodes,
                    }
                },
            }
        )
    )
    return workspace


def test_eval_launcher_uses_verified_task_checkpoint_and_real_simulator(tmp_path):
    workspace = make_verified_workspace(tmp_path)
    spec = load_evaluation_spec(workspace, "drawer_open_place", smoke=False)

    server = build_server_command(workspace, spec)
    simulator = build_simulator_command(spec)
    env = evaluation_environment(workspace, spec)

    assert server[server.index("--model-path") + 1] == str(spec.checkpoint)
    assert server[server.index("--embodiment-tag") + 1] == "NEW_EMBODIMENT"
    assert simulator[0].endswith("RoboSynChallenge/policy/groot/eval.sh")
    assert simulator[1] == "drawer_open_place"
    assert spec.episodes == 50
    assert env["ROBOSYN_GROOT_COMMIT"] == GROOT_COMMIT
    assert env["ROBOSYN_DATASET_REVISION"] == "6fdf6ee1ed74bbbbc0216d507c8a19c8994a15c3"
    assert env["ROBOSYN_PROTOCOL_SOURCE"] == "fallback_50_seeds_0_49"
    assert env["EMBODICHAIN_DATA_ROOT"] == str(
        tmp_path / "cache/embodichain_data"
    )
    assert env["EMBODICHAIN_DATASET_ROOT"] == str(
        tmp_path / "cache/embodichain_datasets"
    )
    assert EMBODICHAIN_COMMIT == "9ebee30011f378f94a7cbe78b01d8c2eacba231a"


def test_table_eval_uses_approved_twenty_episode_protocol(tmp_path):
    workspace = make_verified_workspace(
        tmp_path, task="table_rearrangement", evaluation_episodes=20
    )

    spec = load_evaluation_spec(workspace, "table_rearrangement", smoke=False)
    simulator = build_simulator_command(spec)
    env = evaluation_environment(workspace, spec)

    assert spec.episodes == 20
    assert spec.seeds == tuple(range(20))
    assert simulator[4] == "20"
    assert env["ROBOSYN_PROTOCOL_SOURCE"] == "user_override_20_seeds_0_19"


def test_eval_refuses_unverified_checkpoint(tmp_path):
    workspace = make_verified_workspace(tmp_path)
    result = workspace.run_dir("drawer_open_place") / "training_result.json"
    payload = json.loads(result.read_text())
    payload["reload_status"] = "failed"
    result.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="no verified final checkpoint"):
        load_evaluation_spec(workspace, "drawer_open_place", smoke=True)


def test_eval_rejects_episode_that_never_reached_policy_inference(tmp_path):
    output = tmp_path / "eval"
    (output / "videos").mkdir(parents=True)
    (output / "videos/fail.mp4").write_bytes(b"video")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "number_of_episodes": 1,
                "requested_episodes": 1,
                "seed_list": [0],
                "inference_latency_scope": "model_rpc_only",
            }
        )
    )
    (output / "episode_results.jsonl").write_text(
        json.dumps(
            {
                "episode": 0,
                "seed": 0,
                "error": "KeyError",
                "inference_calls": 0,
                "episode_length": 0,
            }
        )
        + "\n"
    )

    with pytest.raises(RuntimeError, match="non-rollout/error"):
        validate_rollout_artifacts(output, 1)


def test_eval_rejects_corrupt_rollout_video(tmp_path):
    output = tmp_path / "eval"
    (output / "videos").mkdir(parents=True)
    (output / "videos/corrupt.mp4").write_bytes(b"not an mp4")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "number_of_episodes": 1,
                "requested_episodes": 1,
                "seed_list": [0],
                "inference_latency_scope": "model_rpc_only",
            }
        )
    )
    (output / "episode_results.jsonl").write_text(
        json.dumps(
            {
                "episode": 0,
                "seed": 0,
                "error": None,
                "inference_calls": 1,
                "episode_length": 1,
            }
        )
        + "\n"
    )

    with pytest.raises(RuntimeError, match="invalid rollout video"):
        validate_rollout_artifacts(output, 1)


def test_eval_rejects_latency_that_includes_simulator_chunk_execution(tmp_path):
    output = tmp_path / "eval"
    output.mkdir()
    (output / "summary.json").write_text(
        json.dumps(
            {
                "number_of_episodes": 1,
                "inference_latency_scope": "policy_eval_including_environment",
            }
        )
    )

    with pytest.raises(RuntimeError, match="model-RPC-only"):
        validate_rollout_artifacts(output, 1)


def test_eval_rejects_noncanonical_seed_list(tmp_path):
    output = tmp_path / "eval"
    output.mkdir()
    (output / "summary.json").write_text(
        json.dumps(
            {
                "number_of_episodes": 1,
                "requested_episodes": 1,
                "seed_list": [7],
                "inference_latency_scope": "model_rpc_only",
            }
        )
    )

    with pytest.raises(RuntimeError, match="seed list"):
        validate_rollout_artifacts(output, 1)


def test_fresh_server_retry_resumes_after_transient_episode_failure():
    attempts = []

    def operation(attempt):
        attempts.append(attempt)
        if attempt == 1:
            raise RuntimeError("policy returned NaN")
        return "completed"

    assert evaluation.run_with_fresh_server_retries(operation, attempts=3) == "completed"
    assert attempts == [1, 2]


def test_fresh_server_retry_is_bounded():
    attempts = []

    def operation(attempt):
        attempts.append(attempt)
        raise RuntimeError("persistent failure")

    with pytest.raises(RuntimeError, match="failed after 3 fresh-server attempts"):
        evaluation.run_with_fresh_server_retries(operation, attempts=3)

    assert attempts == [1, 2, 3]
