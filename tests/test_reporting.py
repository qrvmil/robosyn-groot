import json
from pathlib import Path

from tools.pipeline import TASKS, Workspace
from tools.reporting import refresh_reports


def test_reporting_uses_actual_results_and_no_fake_eval_rows(tmp_path: Path):
    workspace = Workspace(tmp_path, tmp_path / "groot")
    registry = {
        "tasks": {
            task: {"resolved_hf_revision": f"{index:040x}"}
            for index, task in enumerate(TASKS, start=1)
        }
    }
    path = tmp_path / "configs/tasks/registry.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(registry))
    result = workspace.run_dir("drawer_open_place") / "training_result.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({"status": "pass", "final_loss": 0.0214, "reload_status": "pass"}))
    partial = tmp_path / "eval/drawer_open_place/summary.json"
    partial.parent.mkdir(parents=True)
    partial.write_text(
        json.dumps({"number_of_episodes": 3, "requested_episodes": 50, "successes": 0})
    )

    refresh_reports(workspace)

    training = json.loads((tmp_path / "reports/TRAINING_SUMMARY.json").read_text())
    evaluation = json.loads((tmp_path / "reports/EVALUATION_SUMMARY.json").read_text())
    drawer = next(row for row in training if row["task"] == "drawer_open_place")
    mixer = next(row for row in training if row["task"] == "mixer_operating")
    assert drawer["final_train_loss"] == 0.0214
    assert drawer["target_optimizer_steps"] == 2000
    assert mixer["target_optimizer_steps"] == 4000
    assert evaluation == []
    final_status = (tmp_path / "reports/FINAL_STATUS.md").read_text()
    assert "Completed full evaluations: 0/10" in final_status
    assert "| drawer_open_place |" in final_status
    assert "| pass |" in final_status
    assert "not_started" in final_status


def test_reporting_excludes_full_eval_with_legacy_latency_scope(tmp_path: Path):
    workspace = Workspace(tmp_path, tmp_path / "groot")
    registry = {
        "tasks": {
            task: {"resolved_hf_revision": f"{index:040x}"}
            for index, task in enumerate(TASKS, start=1)
        }
    }
    path = tmp_path / "configs/tasks/registry.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(registry))
    summary = tmp_path / "eval/drawer_open_place/summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            {
                "number_of_episodes": 50,
                "requested_episodes": 50,
                "successes": 0,
                "inference_latency_scope": "policy_eval_including_environment",
            }
        )
    )
    refresh_reports(workspace)

    assert json.loads((tmp_path / "reports/EVALUATION_SUMMARY.json").read_text()) == []


def test_reporting_accepts_twenty_episode_result_only_for_twenty_episode_target(tmp_path: Path):
    workspace = Workspace(tmp_path, tmp_path / "groot")
    registry = {
        "tasks": {
            task: {
                "resolved_hf_revision": f"{index:040x}",
                "evaluation_episodes": 20 if task == "table_rearrangement" else 50,
            }
            for index, task in enumerate(TASKS, start=1)
        }
    }
    path = tmp_path / "configs/tasks/registry.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(registry))
    summary = tmp_path / "eval/table_rearrangement/summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            {
                "number_of_episodes": 20,
                "requested_episodes": 20,
                "seed_list": list(range(20)),
                "successes": 7,
                "inference_latency_scope": "model_rpc_only",
            }
        )
    )
    (summary.parent / "episode_results.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "episode": episode,
                    "seed": episode,
                    "episode_length": 1,
                    "inference_calls": 1,
                    "error": None,
                }
            )
            + "\n"
            for episode in range(20)
        )
    )

    refresh_reports(workspace)

    evaluation = json.loads((tmp_path / "reports/EVALUATION_SUMMARY.json").read_text())
    assert [(row["task"], row["eval_episodes"]) for row in evaluation] == [
        ("table_rearrangement", 20)
    ]


def test_reporting_rejects_summary_that_disagrees_with_registry_target(tmp_path: Path):
    workspace = Workspace(tmp_path, tmp_path / "groot")
    registry = {
        "tasks": {
            task: {
                "resolved_hf_revision": f"{index:040x}",
                "evaluation_episodes": 20 if task == "table_rearrangement" else 50,
            }
            for index, task in enumerate(TASKS, start=1)
        }
    }
    path = tmp_path / "configs/tasks/registry.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(registry))
    summary = tmp_path / "eval/table_rearrangement/summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            {
                "number_of_episodes": 50,
                "requested_episodes": 50,
                "seed_list": list(range(50)),
                "successes": 7,
                "inference_latency_scope": "model_rpc_only",
            }
        )
    )

    refresh_reports(workspace)

    assert json.loads((tmp_path / "reports/EVALUATION_SUMMARY.json").read_text()) == []


def test_reporting_excludes_full_summary_with_error_episode(tmp_path: Path):
    workspace = Workspace(tmp_path, tmp_path / "groot")
    registry = {
        "tasks": {
            task: {
                "resolved_hf_revision": f"{index:040x}",
                "evaluation_episodes": 20,
            }
            for index, task in enumerate(TASKS, start=1)
        }
    }
    registry_path = tmp_path / "configs/tasks/registry.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(json.dumps(registry))
    eval_dir = tmp_path / "eval/table_rearrangement"
    eval_dir.mkdir(parents=True)
    (eval_dir / "summary.json").write_text(
        json.dumps(
            {
                "number_of_episodes": 20,
                "requested_episodes": 20,
                "seed_list": list(range(20)),
                "successes": 3,
                "inference_latency_scope": "model_rpc_only",
            }
        )
    )
    results = [
        {
            "episode": index,
            "seed": index,
            "episode_length": 1,
            "inference_calls": 1,
            "error": "ValueError: NaN" if index == 14 else None,
        }
        for index in range(20)
    ]
    (eval_dir / "episode_results.jsonl").write_text(
        "".join(json.dumps(result) + "\n" for result in results)
    )

    refresh_reports(workspace)

    assert json.loads((tmp_path / "reports/EVALUATION_SUMMARY.json").read_text()) == []
