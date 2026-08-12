#!/usr/bin/env python3
"""Machine-check the click-bell GR00T launch handoff without starting training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from typing import Any, Sequence


MIN_DISK_BYTES = 150 * 1024**3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def validate_command_flags(command: Sequence[str], help_text: str) -> list[str]:
    known = set(re.findall(r"--[a-z][a-z0-9-]*", help_text))
    errors = []
    for token in command:
        if token.startswith("--"):
            flag = token.split("=", 1)[0]
            if flag not in known:
                errors.append(f"unknown flag: {flag}")
    return errors


def validate_launcher_environment(command: str) -> list[str]:
    errors = []
    if re.search(r"(?:^|\s)HF_HUB_OFFLINE=1(?:\s|$)", command):
        errors.append("HF_HUB_OFFLINE=1 blocks required Cosmos processor metadata lookup")
    if re.search(r"(?:^|\s)TRANSFORMERS_OFFLINE=1(?:\s|$)", command):
        errors.append("TRANSFORMERS_OFFLINE=1 blocks required Cosmos processor metadata lookup")
    return errors


def _command_tokens(path: Path) -> list[str]:
    text = _text(path).replace("\\\n", " ")
    try:
        tokens = shlex.split(text, comments=True)
    except ValueError:
        return []
    for index, token in enumerate(tokens):
        if token.endswith("launch_finetune.py"):
            return [token, *tokens[index + 1 :]]
    return []


def _has_pair(tokens: Sequence[str], flag: str, value: str) -> bool:
    return any(tokens[index : index + 2] == [flag, value] for index in range(len(tokens) - 1))


def _runtime() -> dict[str, Any]:
    result: dict[str, Any] = {
        "cuda": False,
        "bf16": False,
        "gpu": "unavailable",
        "ffmpeg": shutil.which("ffmpeg") is not None,
    }
    try:
        import torch

        result["cuda"] = bool(torch.cuda.is_available())
        if result["cuda"]:
            result["bf16"] = bool(torch.cuda.is_bf16_supported())
            result["gpu"] = torch.cuda.get_device_name(0)
            result["torch"] = torch.__version__
            result["torch_cuda"] = torch.version.cuda
    except Exception as exc:  # pragma: no cover - diagnostic boundary
        result["error"] = str(exc)
    return result


def _recursive_numbers(value: Any, key_pattern: str) -> list[float]:
    found: list[float] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if re.search(key_pattern, str(key), re.I) and isinstance(item, (int, float)):
                found.append(float(item))
            found.extend(_recursive_numbers(item, key_pattern))
    elif isinstance(value, list):
        for item in value:
            found.extend(_recursive_numbers(item, key_pattern))
    return found


def verify_readiness(work_root: Path, run_name: str) -> dict[str, object]:
    work_root = Path(work_root).resolve()
    run_dir = work_root / "runs" / run_name
    manifest_path = run_dir / "launch_manifest.json"
    manifest = _json(manifest_path)
    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, passed: bool, detail: str) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    record("launch_manifest", bool(manifest), str(manifest_path))

    tiny = _json(work_root / "reports" / "tiny_overfit.json")
    tiny_pass = (
        tiny.get("status") == "pass"
        and tiny.get("training", {}).get("exit_code") == 0
        and tiny.get("reload", {}).get("passed") is True
        and tiny.get("open_loop", {}).get("detailed_all_finite") is True
        and tiny.get("open_loop", {}).get("detailed_predictions_nonconstant") is True
    )
    record("tiny_overfit", tiny_pass, "500-step training, reload, and open-loop evidence")

    full_smoke = _json(work_root / "reports" / "full_startup_smoke.json")
    full_smoke_pass = (
        full_smoke.get("status") == "pass"
        and full_smoke.get("exit_code") == 0
        and full_smoke.get("global_batch_size") == 32
        and int(full_smoke.get("optimizer_steps", 0)) >= 1
        and full_smoke.get("cosmos_processor_metadata_lookup")
        == "pass after removing forced offline environment"
    )
    record(
        "full_startup_smoke",
        full_smoke_pass,
        "batch 32, full dataset, Cosmos metadata, and one optimizer step",
    )

    checksums = manifest.get("checksums", {}) if isinstance(manifest.get("checksums"), dict) else {}
    config_rel = "configs/robosyn_cobotmagic_config.py"
    config = work_root / config_rel
    expected_config = checksums.get(config_rel)
    config_ok = bool(expected_config and config.is_file() and sha256_file(config) == expected_config)
    record("config_checksum", config_ok, f"expected={expected_config}")

    checksum_failures = []
    for rel, expected in checksums.items():
        candidate = work_root / rel
        if not candidate.is_file() or sha256_file(candidate) != expected:
            checksum_failures.append(rel)
    record(
        "provenance_checksums",
        bool(checksums) and not checksum_failures,
        "matched" if not checksum_failures else "mismatch: " + ", ".join(checksum_failures),
    )

    command_path = run_dir / "command.sh"
    tokens = _command_tokens(command_path)
    record(
        "command_executable",
        command_path.is_file() and os.access(command_path, os.X_OK) and bool(tokens),
        str(command_path),
    )
    flag_errors = validate_command_flags(tokens, _text(work_root / "reports" / "launch_finetune_help.txt"))
    record("cli_flags", bool(tokens) and not flag_errors, "; ".join(flag_errors) or "all flags in pinned help")
    environment_errors = validate_launcher_environment(_text(command_path))
    record(
        "launcher_environment",
        not environment_errors,
        "; ".join(environment_errors) or "Hub metadata lookup is enabled",
    )

    required_pairs = {
        "--base-model-path": str(manifest.get("base_model_path", "")),
        "--embodiment-tag": "NEW_EMBODIMENT",
        "--num-gpus": "1",
        "--global-batch-size": "32",
        "--gradient-accumulation-steps": "1",
        "--dataloader-num-workers": "4",
        "--learning-rate": "1e-4",
        "--weight-decay": "1e-5",
        "--warmup-ratio": "0.05",
        "--max-steps": "2000",
        "--save-steps": "250",
        "--save-total-limit": "6",
    }
    missing_pairs = [f"{key} {value}" for key, value in required_pairs.items() if not _has_pair(tokens, key, value)]
    required_switches = {"--no-tune-llm", "--no-tune-visual", "--tune-projector", "--tune-diffusion-model"}
    missing_switches = sorted(required_switches.difference(tokens))
    unsafe_switches = sorted({"--tune-llm", "--tune-visual", "--save-only-model"}.intersection(tokens))
    scope_ok = not missing_pairs and not missing_switches and not unsafe_switches
    record(
        "launch_scope",
        scope_ok,
        f"missing={missing_pairs + missing_switches}; unsafe={unsafe_switches}",
    )
    full_checkpoint_dir = run_dir / "checkpoints"
    unlaunched = not full_checkpoint_dir.exists() or not any(full_checkpoint_dir.iterdir())
    record("full_run_unlaunched", unlaunched, str(full_checkpoint_dir))

    runtime = _runtime()
    hardware_ok = runtime.get("cuda") and runtime.get("bf16") and "A100" in str(runtime.get("gpu"))
    record("hardware", bool(hardware_ok), json.dumps(runtime, sort_keys=True))
    record("ffmpeg", bool(runtime.get("ffmpeg")), str(shutil.which("ffmpeg")))
    free = shutil.disk_usage(work_root).free
    record("disk_reserve", free >= MIN_DISK_BYTES, f"free_bytes={free}; required={MIN_DISK_BYTES}")

    model_revision = str(manifest.get("model_revision", ""))
    model_snapshot = work_root / "cache" / "huggingface" / "hub" / "models--nvidia--GR00T-N1.7-3B" / "snapshots" / model_revision
    model_files = list(model_snapshot.glob("model-*.safetensors")) if model_snapshot.is_dir() else []
    record("model_access", len(model_files) >= 2, f"revision={model_revision}; shards={len(model_files)}")

    groot_commit = _text(work_root / "reports" / "groot_commit.txt").strip()
    record("groot_revision", groot_commit == manifest.get("groot_revision"), groot_commit)
    smoke = _text(work_root / "runs" / "so100_smoke" / "stdout.log")
    reload_log = _text(work_root / "runs" / "so100_smoke" / "checkpoint_reload.log")
    record("official_smoke", "Training completed!" in smoke and "reload OK" in reload_log, "SO100 train+reload")

    raw_source = _json(work_root / "data" / "manifests" / "cobotmagic_Sim_click_bell.source.json")
    source_ok = raw_source.get("revision") == manifest.get("raw_dataset_revision")
    record("source_revision", source_ok, str(raw_source.get("revision")))
    raw_hash_log = _text(work_root / "reports" / "raw_hash_recheck.txt")
    raw_hash_ok = raw_hash_log.count(": OK") >= 4006 and "FAILED" not in raw_hash_log
    record("raw_hashes", raw_hash_ok, f"verified_files={raw_hash_log.count(': OK')}")

    prepared_audit = _json(work_root / "data" / "manifests" / "cobotmagic_Sim_click_bell__groot_v1.audit.json")
    episodes = prepared_audit.get("episodes", {}).get("count")
    frames = prepared_audit.get("frames", {}).get("count")
    if frames is None:
        frames = prepared_audit.get("frames") if isinstance(prepared_audit.get("frames"), int) else None
    audit_ok = episodes == 1000 and frames == 74000 and bool(prepared_audit.get("reconciliation"))
    record("audit_reconciliation", audit_ok, f"episodes={episodes}; frames={frames}")

    visual_root = work_root / "reports" / "visual_audit"
    visual_videos = list(visual_root.rglob("*.mp4"))
    visual_stills = list(visual_root.rglob("*.png"))
    visual_ok = len(visual_videos) >= 5 and len(visual_stills) >= 15 and (work_root / "reports" / "MODALITY_CONFIG_REVIEW.md").is_file()
    record("visual_review", visual_ok, f"videos={len(visual_videos)}; stills={len(visual_stills)}")

    prepared_root = work_root / "data" / "prepared" / "cobotmagic_Sim_click_bell__groot_v1"
    stats_ok = (prepared_root / "meta" / "stats.json").is_file() and (prepared_root / "meta" / "modality.json").is_file()
    prepared_hash = work_root / "data" / "manifests" / "cobotmagic_Sim_click_bell__groot_v1.sha256.json"
    record("prepared_dataset", prepared_root.is_dir() and stats_ok and prepared_hash.stat().st_size > 1_000_000, str(prepared_root))

    loader = _json(work_root / "reports" / "loader_smoke.json")
    roundtrip = _json(work_root / "reports" / "action_roundtrip.json")
    roundtrip_errors = _recursive_numbers(roundtrip, r"max.*(error|diff)|round.*max")
    loader_ok = bool(loader) and "false" not in json.dumps(loader).lower()
    transform_ok = bool(roundtrip_errors) and max(roundtrip_errors) < 1e-6
    record("loader_stats_roundtrip", loader_ok and transform_ok, f"roundtrip_max={max(roundtrip_errors) if roundtrip_errors else None}")

    semantics = _json(work_root / "configs" / "robosyn_cobotmagic_semantics.json")
    timing = semantics.get("timing", {})
    semantic_ok = timing.get("fps") == 25 and timing.get("action_horizon_steps") == 13 and len(semantics.get("action", {}).get("groups", [])) == 4
    record("semantic_horizons", semantic_ok, json.dumps(timing, sort_keys=True))

    failed = [name for name, value in checks.items() if not value["passed"]]
    return {
        "ready": not failed,
        "run_name": run_name,
        "work_root": str(work_root),
        "failed_checks": failed,
        "checks": checks,
    }


def _markdown(result: dict[str, object]) -> str:
    lines = [
        "# GR00T Fine-Tuning Readiness",
        "",
        f"**Result:** {'READY' if result['ready'] else 'NOT READY'}",
        "",
        f"Run: `{result['run_name']}`",
        "",
        "| Check | Result | Detail |",
        "|---|---:|---|",
    ]
    for name, value in result["checks"].items():  # type: ignore[union-attr]
        mark = "PASS" if value["passed"] else "FAIL"
        detail = str(value["detail"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{name}` | {mark} | {detail} |")
    lines.extend(["", "Full training was not launched by this verifier.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify_readiness(args.work_root, args.run_name)
    rendered = _markdown(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered)
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
