"""Shared paths, invariants, and launcher generation for the RoboSyn pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import time
from dataclasses import dataclass
from pathlib import Path


TASKS = (
    "click_bell",
    "mixer_operating",
    "table_rearrangement",
    "drawer_open_place",
    "manipulate_pipette",
    "water_pouring",
    "item_assembly",
    "handle_basket",
    "items_handover",
    "sample_loading",
)
BASE_MODEL_REPO = "nvidia/GR00T-N1.7-3B"
BASE_MODEL_REVISION = "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
GROOT_COMMIT = "376ba890cff8c9de64d71d982772a9c36185fdd7"


def training_steps_for_task(task: str) -> int:
    """Drawer is the completed 2k baseline; all newly trained tasks use 4k."""
    _require_task(task)
    return 2000 if task == "drawer_open_place" else 4000


def evaluation_episodes_for_task(task: str) -> int:
    """Return the approved immutable evaluation size for a task.

    Drawer, Click Bell, and Mixer already have canonical 50-seed evidence. The
    user-approved 2026-08-27 override applies to Table and every later task.
    """
    _require_task(task)
    completed_fifty_seed_tasks = {
        "drawer_open_place",
        "click_bell",
        "mixer_operating",
    }
    return 50 if task in completed_fifty_seed_tasks else 20


@dataclass(frozen=True)
class Workspace:
    root: Path
    installed_groot: Path = Path("/opt/robosyn/Isaac-GR00T")

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())
        object.__setattr__(self, "installed_groot", Path(self.installed_groot).resolve())

    @property
    def groot_link(self) -> Path:
        return self.root / "repos/Isaac-GR00T"

    @property
    def base_model_snapshot(self) -> Path:
        return (
            self.root
            / "cache/huggingface/hub"
            / f"models--nvidia--GR00T-N1.7-3B/snapshots/{BASE_MODEL_REVISION}"
        )

    def raw_dataset(self, task: str) -> Path:
        _require_task(task)
        return self.root / f"data/raw/RoboSynChallenge/cobotmagic_Sim_{task}"

    def prepared_dataset(self, task: str) -> Path:
        _require_task(task)
        return self.root / f"data/prepared/cobotmagic_Sim_{task}__groot_v1"

    def task_config(self, task: str) -> Path:
        _require_task(task)
        return self.root / f"configs/tasks/{task}_config.py"

    def run_dir(self, task: str) -> Path:
        _require_task(task)
        steps_k = training_steps_for_task(task) // 1000
        return self.root / f"runs/{task}_sim_baseline_{steps_k}k"

    def initialize(self) -> None:
        for path in (
            self.root / "repos",
            self.root / "data/raw",
            self.root / "data/prepared",
            self.root / "data/manifests",
            self.root / "configs/tasks",
            self.root / "runs",
            self.root / "eval",
            self.root / "reports",
            self.root / "cache/huggingface/hub",
        ):
            path.mkdir(parents=True, exist_ok=True)
        link = self.groot_link
        if link.is_symlink():
            if link.resolve() != self.installed_groot:
                link.unlink()
        elif link.exists():
            if link.resolve() != self.installed_groot:
                raise RuntimeError(f"refusing to replace non-symlink GR00T path: {link}")
            return
        if not link.exists():
            link.symlink_to(self.installed_groot)


def _require_task(task: str) -> None:
    if task not in TASKS:
        raise ValueError(f"unknown task: {task!r}; expected one of {', '.join(TASKS)}")


def _semantics_digest(semantics: dict[str, object]) -> str:
    encoded = json.dumps(semantics, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def dataset_materialization_missing_paths(dataset: Path) -> list[Path]:
    """Return absent Parquet/video files declared by LeRobot metadata.

    This is deliberately based on ``info.json`` and ``episodes.jsonl`` rather
    than directory counts: an interrupted Hub download can have valid metadata
    and stats while one or more camera streams are only partly materialized.
    """
    dataset = Path(dataset)
    info_path = dataset / "meta/info.json"
    episodes_path = dataset / "meta/episodes.jsonl"
    missing = [path for path in (info_path, episodes_path) if not path.is_file()]
    if missing:
        return missing
    try:
        info = json.loads(info_path.read_text())
        episodes = [
            int(json.loads(line)["episode_index"])
            for line in episodes_path.read_text().splitlines()
            if line.strip()
        ]
        expected_episodes = int(info["total_episodes"])
        chunk_size = int(info["chunks_size"])
        data_template = str(info["data_path"])
        video_template = str(info["video_path"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return [info_path]
    if len(episodes) != expected_episodes or len(set(episodes)) != expected_episodes:
        missing.append(episodes_path)
    video_keys = [
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") == "video"
    ]
    for episode in episodes:
        values = {
            "episode_chunk": episode // chunk_size,
            "episode_index": episode,
        }
        data_path = dataset / data_template.format(**values)
        if not data_path.is_file():
            missing.append(data_path)
        for video_key in video_keys:
            video_path = dataset / video_template.format(video_key=video_key, **values)
            if not video_path.is_file():
                missing.append(video_path)
    return missing


def prepared_dataset_is_current(
    prepared: Path,
    semantics: dict[str, object],
    *,
    allowed_features: set[str],
    require_stats: bool,
) -> bool:
    prepared = Path(prepared)
    try:
        manifest = json.loads((prepared / "meta/preparation_manifest.json").read_text())
        info = json.loads((prepared / "meta/info.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("semantics_sha256") != _semantics_digest(semantics):
        return False
    if not set(info.get("features", {})).issubset(allowed_features):
        return False
    required = {
        str(semantics["state"]["original_key"]),
        str(semantics.get("action", {}).get("original_key", "action")),
    }
    if not required.issubset(info.get("features", {})):
        return False
    if require_stats and not all(
        (prepared / "meta" / name).is_file()
        for name in ("stats.json", "relative_stats.json")
    ):
        return False
    # Old prepared copies can look current by manifest while inheriting an
    # interrupted raw snapshot. Never allow those past the idempotency gate.
    if info.get("total_episodes") is not None and dataset_materialization_missing_paths(
        prepared
    ):
        return False
    return True


def build_train_command(
    workspace: Workspace,
    task: str,
    *,
    max_steps: int,
    global_batch_size: int = 16,
    gradient_accumulation_steps: int = 2,
    run_dir: Path | None = None,
) -> list[str]:
    _require_task(task)
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if global_batch_size * gradient_accumulation_steps != 32:
        raise ValueError("effective optimizer batch must remain 32")
    run_dir = Path(run_dir) if run_dir is not None else workspace.run_dir(task)
    return [
        str(workspace.groot_link / ".venv/bin/python"),
        str(workspace.groot_link / "gr00t/experiment/launch_finetune.py"),
        "--base-model-path",
        str(workspace.base_model_snapshot),
        "--dataset-path",
        str(workspace.prepared_dataset(task)),
        "--embodiment-tag",
        "NEW_EMBODIMENT",
        "--modality-config-path",
        str(workspace.task_config(task)),
        "--num-gpus",
        "1",
        "--output-dir",
        str(run_dir / "checkpoints"),
        "--experiment-name",
        f"{task}_sim_baseline_{training_steps_for_task(task) // 1000}k",
        "--no-use-wandb",
        "--no-tune-llm",
        "--no-tune-visual",
        "--tune-projector",
        "--tune-diffusion-model",
        "--state-dropout-prob",
        "0.0",
        "--use-percentiles",
        "--global-batch-size",
        str(global_batch_size),
        "--gradient-accumulation-steps",
        str(gradient_accumulation_steps),
        "--dataloader-num-workers",
        "4",
        "--learning-rate",
        "1e-4",
        "--weight-decay",
        "1e-5",
        "--warmup-ratio",
        "0.05",
        "--episode-sampling-rate",
        "1.0",
        "--shard-size",
        "1024",
        "--num-shards-per-epoch",
        "100",
        "--save-steps",
        "250",
        "--save-total-limit",
        "2",
        "--max-steps",
        str(max_steps),
    ]


def write_train_launcher(
    workspace: Workspace,
    task: str,
    *,
    run_dir: Path,
    max_steps: int,
    global_batch_size: int,
    gradient_accumulation_steps: int,
) -> tuple[Path, Path]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    command = build_train_command(
        workspace,
        task,
        max_steps=max_steps,
        global_batch_size=global_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        run_dir=run_dir,
    )
    resume_candidates: list[tuple[int, Path]] = []
    checkpoints_root = run_dir / "checkpoints"
    if checkpoints_root.exists():
        for candidate in checkpoints_root.rglob("checkpoint-*"):
            if not candidate.is_dir() or not (candidate / "optimizer.pt").is_file():
                continue
            try:
                step = int(candidate.name.rsplit("-", 1)[1])
            except ValueError:
                continue
            resume_candidates.append((step, candidate))
    resume_checkpoint = (
        max(resume_candidates, key=lambda item: item[0])[1]
        if resume_candidates
        else None
    )
    if resume_checkpoint is not None:
        command.append("--resume-from-checkpoint")
    command_path = run_dir / "command.sh"
    command_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"export HF_HOME={shlex.quote(str(workspace.root / 'cache/huggingface'))}\n"
        f"export HUGGINGFACE_HUB_CACHE={shlex.quote(str(workspace.root / 'cache/huggingface/hub'))}\n"
        f"export TORCH_HOME={shlex.quote(str(workspace.root / 'cache/torch'))}\n"
        "export CUDA_HOME=/usr/local/cuda\n"
        "export TOKENIZERS_PARALLELISM=false\n"
        "export PYTHONUNBUFFERED=1\n"
        f"cd {shlex.quote(str(workspace.groot_link))}\n"
        f"exec {shlex.join(command)}\n"
    )
    command_path.chmod(0o755)
    manifest_path = run_dir / "run_manifest.json"
    manifest = {
        "schema_version": 1,
        "task": task,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_model_repo": BASE_MODEL_REPO,
        "base_model_revision": BASE_MODEL_REVISION,
        "base_model_path": str(workspace.base_model_snapshot),
        "dataset_path": str(workspace.prepared_dataset(task)),
        "modality_config_path": str(workspace.task_config(task)),
        "groot_commit": GROOT_COMMIT,
        "architecture": "Gr00tN1d7",
        "global_batch_size": global_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": global_batch_size * gradient_accumulation_steps,
        "learning_rate": 1e-4,
        "optimizer": "AdamW",
        "scheduler": "cosine",
        "weight_decay": 1e-5,
        "warmup_ratio": 0.05,
        "max_grad_norm": 1.0,
        "compute_dtype": "BF16",
        "max_steps": max_steps,
        "save_steps": 250,
        "save_total_limit": 2,
        "resume_from_checkpoint": resume_checkpoint is not None,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "status": "created",
        "command": command,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return command_path, manifest_path


def pipeline_environment(workspace: Workspace) -> dict[str, str]:
    env = {
        **os.environ,
        "WORK_ROOT": str(workspace.root),
        "HF_HOME": str(workspace.root / "cache/huggingface"),
        "HUGGINGFACE_HUB_CACHE": str(workspace.root / "cache/huggingface/hub"),
        "TORCH_HOME": str(workspace.root / "cache/torch"),
        "UV_CACHE_DIR": str(workspace.root / "cache/uv"),
        "WANDB_DIR": str(workspace.root / "runs/wandb"),
        "CUDA_HOME": "/usr/local/cuda",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_DISABLE_XET": os.environ.get("HF_HUB_DISABLE_XET", "1"),
        "HF_HUB_ETAG_TIMEOUT": os.environ.get("HF_HUB_ETAG_TIMEOUT", "30"),
        "HF_HUB_DOWNLOAD_TIMEOUT": os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", "60"),
    }
    # Cosmos performs Hub metadata resolution even with cached model weights.
    env.pop("HF_HUB_OFFLINE", None)
    return env
