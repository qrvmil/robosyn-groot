import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "tools/cli/robosyn"


def cli_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    installed = tmp_path / "installed-groot"
    (installed / ".venv/bin").mkdir(parents=True)
    (installed / ".venv/bin/python").symlink_to(Path(sys.executable).resolve())
    env = {
        **os.environ,
        "ROBOSYN_WORK_ROOT": str(tmp_path / "work"),
        "ROBOSYN_GROOT_SRC": str(installed),
        "ROBOSYN_PYTHON_BIN": str(ROOT / "repos/Isaac-GR00T/.venv/bin/python"),
    }
    return env, installed


def test_named_list_tasks_entrypoint_initializes_workspace_symlink(tmp_path: Path):
    env, installed = cli_env(tmp_path)
    named = tmp_path / "robosyn-list-tasks"
    named.symlink_to(ENTRYPOINT)

    result = subprocess.run([named], env=env, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == "click_bell"
    assert "sample_loading" in result.stdout.splitlines()
    assert (tmp_path / "work/repos/Isaac-GR00T").resolve() == installed.resolve()


def test_cli_rejects_missing_and_unknown_task_arguments(tmp_path: Path):
    env, _ = cli_env(tmp_path)

    missing = subprocess.run(
        [ENTRYPOINT, "bootstrap"], env=env, text=True, capture_output=True
    )
    unknown = subprocess.run(
        [ENTRYPOINT, "bootstrap", "not_a_task"],
        env=env,
        text=True,
        capture_output=True,
    )

    assert missing.returncode != 0
    assert "task" in missing.stderr.lower()
    assert unknown.returncode != 0
    assert "invalid choice" in unknown.stderr.lower() or "unknown task" in unknown.stderr.lower()


def test_repo_entrypoint_exports_resilient_online_hub_timeouts():
    source = ENTRYPOINT.read_text()

    assert "HF_HUB_ETAG_TIMEOUT" in source
    assert "HF_HUB_DOWNLOAD_TIMEOUT" in source
    assert "unset HF_HUB_OFFLINE" in source
    assert "HF_HUB_OFFLINE=1" not in source
