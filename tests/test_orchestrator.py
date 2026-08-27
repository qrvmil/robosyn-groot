import json
from pathlib import Path

from tools.pipeline import Workspace
from tools.run_all_tasks import (
    ALL_PHASES,
    ORDERED_TASKS,
    Orchestrator,
    command_for_phase,
    run_continuously,
)


def test_orchestrator_starts_with_drawer_and_is_single_phase_per_command():
    assert ORDERED_TASKS[0] == "drawer_open_place"
    assert len(ORDERED_TASKS) == 10
    assert command_for_phase("drawer_open_place", "train") == [
        "robosyn-train",
        "drawer_open_place",
    ]


def test_orchestrator_resumes_completed_phases(tmp_path: Path, monkeypatch):
    workspace = Workspace(tmp_path, tmp_path / "installed-groot")
    workspace.initialize()
    orchestrator = Orchestrator(workspace, ("bootstrap", "smoke"))
    orchestrator.state["tasks"]["drawer_open_place"]["phases"]["bootstrap"] = "completed"
    calls = []

    def fake_run_phase(task, phase):
        calls.append((task, phase))
        orchestrator.state["tasks"][task]["phases"][phase] = "completed"

    monkeypatch.setattr(orchestrator, "run_phase", fake_run_phase)
    assert orchestrator.run() == 0

    assert ("drawer_open_place", "bootstrap") not in calls
    assert calls[0] == ("drawer_open_place", "smoke")
    state = json.loads(orchestrator.state_path.read_text())
    assert state["tasks"]["drawer_open_place"]["status"] == "completed"


def test_all_phases_keep_training_before_evaluation():
    assert ALL_PHASES.index("train") < ALL_PHASES.index("eval")


def test_continuous_orchestrator_retries_failed_pass_until_complete():
    class FakeOrchestrator:
        terminate_requested = False

        def __init__(self):
            self.results = iter((1, 0))
            self.calls = 0

        def run(self):
            self.calls += 1
            return next(self.results)

    orchestrator = FakeOrchestrator()
    sleeps = []

    assert run_continuously(orchestrator, retry_delay=7, sleeper=sleeps.append) == 0
    assert orchestrator.calls == 2
    assert sleeps == [7]


def test_orchestrator_reopens_completed_eval_with_legacy_latency_scope(tmp_path: Path):
    workspace = Workspace(tmp_path, tmp_path / "installed-groot")
    workspace.initialize()
    state_path = tmp_path / "reports/orchestrator_state.json"
    orchestrator = Orchestrator(workspace, ("eval",))
    task_state = orchestrator.state["tasks"]["drawer_open_place"]
    task_state["phases"]["eval"] = "completed"
    task_state["status"] = "completed"
    orchestrator.save()
    summary = tmp_path / "eval/drawer_open_place/summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            {
                "number_of_episodes": 50,
                "inference_latency_scope": "policy_eval_including_environment",
            }
        )
    )

    resumed = Orchestrator(workspace, ("eval",))

    assert resumed.state["tasks"]["drawer_open_place"]["phases"]["eval"] == "pending"
    assert resumed.state["tasks"]["drawer_open_place"]["status"] == "pending"


def test_orchestrator_accepts_completed_twenty_episode_task_target(tmp_path: Path):
    workspace = Workspace(tmp_path, tmp_path / "installed-groot")
    workspace.initialize()
    registry = tmp_path / "configs/tasks/registry.json"
    registry.write_text(
        json.dumps(
            {"tasks": {"table_rearrangement": {"evaluation_episodes": 20}}}
        )
    )
    orchestrator = Orchestrator(workspace, ("eval",))
    task_state = orchestrator.state["tasks"]["table_rearrangement"]
    task_state["phases"]["eval"] = "completed"
    task_state["status"] = "completed"
    orchestrator.save()
    summary = tmp_path / "eval/table_rearrangement/summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            {
                "number_of_episodes": 20,
                "requested_episodes": 20,
                "seed_list": list(range(20)),
                "inference_latency_scope": "model_rpc_only",
            }
        )
    )

    resumed = Orchestrator(workspace, ("eval",))

    assert resumed.state["tasks"]["table_rearrangement"]["phases"]["eval"] == "completed"


def test_orchestrator_reopens_eval_that_does_not_match_per_task_target(tmp_path: Path):
    workspace = Workspace(tmp_path, tmp_path / "installed-groot")
    workspace.initialize()
    registry = tmp_path / "configs/tasks/registry.json"
    registry.write_text(
        json.dumps(
            {"tasks": {"table_rearrangement": {"evaluation_episodes": 20}}}
        )
    )
    orchestrator = Orchestrator(workspace, ("eval",))
    task_state = orchestrator.state["tasks"]["table_rearrangement"]
    task_state["phases"]["eval"] = "completed"
    task_state["status"] = "completed"
    orchestrator.save()
    summary = tmp_path / "eval/table_rearrangement/summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            {
                "number_of_episodes": 50,
                "requested_episodes": 50,
                "seed_list": list(range(50)),
                "inference_latency_scope": "model_rpc_only",
            }
        )
    )

    resumed = Orchestrator(workspace, ("eval",))

    assert resumed.state["tasks"]["table_rearrangement"]["phases"]["eval"] == "pending"
