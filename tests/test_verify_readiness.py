from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.verify_readiness import (
    validate_command_flags,
    validate_launcher_environment,
    verify_readiness,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _workspace(tmp_path: Path) -> Path:
    root = tmp_path / "work"
    config = root / "configs" / "robosyn_cobotmagic_config.py"
    config.parent.mkdir(parents=True)
    config.write_text("CONFIG = 1\n")
    run = root / "runs" / "baseline"
    run.mkdir(parents=True)
    (run / "launch_manifest.json").write_text(
        json.dumps(
            {
                "checksums": {
                    "configs/robosyn_cobotmagic_config.py": _sha256(config),
                }
            }
        )
    )
    return root


def test_readiness_fails_without_tiny_gate(tmp_path: Path):
    root = _workspace(tmp_path)
    result = verify_readiness(root, "baseline")
    assert not result["ready"]
    assert "tiny_overfit" in result["failed_checks"]


def test_readiness_rejects_config_mismatch(tmp_path: Path):
    root = _workspace(tmp_path)
    tiny = root / "reports" / "tiny_overfit.json"
    tiny.parent.mkdir(parents=True)
    tiny.write_text(json.dumps({"status": "pass"}))
    (root / "configs" / "robosyn_cobotmagic_config.py").write_text("changed\n")
    result = verify_readiness(root, "baseline")
    assert "config_checksum" in result["failed_checks"]


def test_validate_command_flags_rejects_unknown_flag():
    help_text = "--dataset-path STR\n--max-steps INT\n"
    errors = validate_command_flags(
        ["launch_finetune.py", "--dataset-path", "/data", "--mystery", "1"], help_text
    )
    assert errors == ["unknown flag: --mystery"]


def test_launcher_environment_rejects_forced_hf_offline_mode():
    command = "exec env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 launch_finetune.py"
    assert validate_launcher_environment(command) == [
        "HF_HUB_OFFLINE=1 blocks required Cosmos processor metadata lookup"
    ]


def test_readiness_requires_full_profile_startup_smoke(tmp_path: Path):
    root = _workspace(tmp_path)
    result = verify_readiness(root, "baseline")
    assert "full_startup_smoke" in result["failed_checks"]
