#!/usr/bin/env python3
"""Idempotently create and verify the isolated RoboSyn simulator venv."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluation import EMBODICHAIN_COMMIT, ROBOSYN_COMMIT
from tools.pipeline import Workspace


DEXSIM_SHA256 = "9c10f2ff78f6de36cbf2cd9c20d4d04ae900d3755b365207b4829b2fcc4432e3"
DEXSIM_WHEEL = "dexsim_engine-0.4.3-cp311-cp311-manylinux_2_31_x86_64.whl"


def _run(command: list[str], **kwargs: object) -> None:
    subprocess.run(command, check=True, **kwargs)


def _git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(workspace: Workspace) -> dict[str, object]:
    python = workspace.root / ".venvs/robosyn/bin/python"
    if not python.is_file():
        raise RuntimeError(f"simulator Python is missing: {python}")
    env = {
        **os.environ,
        "EMBODICHAIN_DATA_ROOT": str(workspace.root / "cache/embodichain_data"),
        "EMBODICHAIN_DATASET_ROOT": str(
            workspace.root / "cache/embodichain_datasets"
        ),
        "HF_HOME": str(workspace.root / "cache/huggingface"),
    }
    probe = subprocess.check_output(
        [
            str(python),
            "-c",
            (
                "import json,numpy,torch,av,dexsim,embodichain,robosynchallenge;"
                "import robosynchallenge.tasks;"
                "from robosynchallenge.tasks.drawer_open_place.drawer_open_place "
                "import DrawerOpenPlaceEnv;"
                "print(json.dumps({'python':__import__('sys').version.split()[0],"
                "'numpy':numpy.__version__,'torch':torch.__version__,"
                "'cuda':torch.cuda.is_available(),'av':av.__version__,"
                "'dexsim':dexsim.__version__,'embodichain':embodichain.__version__,"
                "'robosyn':robosynchallenge.__version__}))"
            ),
        ],
        cwd=workspace.root / "repos/RoboSynChallenge",
        env=env,
        text=True,
    )
    versions = json.loads(probe.strip().splitlines()[-1])
    if versions["python"].split(".")[:2] != ["3", "11"]:
        raise RuntimeError(f"simulator must use Python 3.11: {versions}")
    if versions["numpy"] != "1.26.4" or not versions["cuda"]:
        raise RuntimeError(f"simulator ABI/CUDA verification failed: {versions}")
    if _git_commit(workspace.root / "repos/EmbodiChain") != EMBODICHAIN_COMMIT:
        raise RuntimeError("EmbodiChain checkout is not at the pinned commit")
    if _git_commit(workspace.root / "repos/RoboSynChallenge") != ROBOSYN_COMMIT:
        raise RuntimeError("RoboSyn checkout is not at the pinned commit")
    return {
        "status": "pass",
        "versions": versions,
        "embodichain_commit": EMBODICHAIN_COMMIT,
        "robosyn_commit": ROBOSYN_COMMIT,
    }


def bootstrap(workspace: Workspace) -> dict[str, object]:
    venv = workspace.root / ".venvs/robosyn"
    python = venv / "bin/python"
    wheel = workspace.root / "cache/simulator_wheels" / DEXSIM_WHEEL
    if not wheel.is_file() or _sha256(wheel) != DEXSIM_SHA256:
        raise RuntimeError(
            f"verified DexSim wheel is unavailable at {wheel}; refusing an unpinned wheel"
        )
    if not python.is_file():
        _run(["uv", "venv", "--python", "3.11", str(venv)])
    pip = [str(python), "-m", "pip"]
    _run(pip + ["install", str(wheel)])
    _run(
        pip
        + [
            "install",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu128",
            "-r",
            str(workspace.root / "configs/simulator/requirements.lock.txt"),
        ]
    )
    for repo in ("EmbodiChain", "RoboSynChallenge"):
        _run(pip + ["install", "--no-deps", "-e", str(workspace.root / "repos" / repo)])
    return verify(workspace)


def main() -> int:
    workspace = Workspace(Path("/workspace/challenge/robosyn-groot"))
    result = bootstrap(workspace) if "--install" in sys.argv else verify(workspace)
    output = workspace.root / "reports/simulator_environment.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
