#!/usr/bin/env python3
"""Prepare a RoboSyn LeRobot v2.1 snapshot for GR00T N1.7."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import random
import shutil
import stat
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def validate_semantics(semantics: dict[str, object], state_dim: int, action_dim: int) -> None:
    for name, dimension in (("state", state_dim), ("action", action_dim)):
        section = semantics.get(name)
        if not isinstance(section, dict):
            raise ValueError(f"missing semantics section: {name}")
        groups = section.get("groups")
        if not isinstance(groups, list) or not groups:
            raise ValueError(f"{name}.groups must be a non-empty list")
        cursor = 0
        seen = set()
        for group in groups:
            key = str(group["key"])
            start, end = int(group["start"]), int(group["end"])
            if key in seen:
                raise ValueError(f"duplicate {name} key: {key}")
            if start != cursor or end <= start:
                raise ValueError(f"{name} slice {key} [{start}:{end}) is not contiguous at {cursor}")
            seen.add(key)
            cursor = end
        if cursor != dimension:
            raise ValueError(f"{name} slices end at {cursor}, measured dimension is {dimension}")
    if [g["key"] for g in semantics["state"]["groups"]] != [
        g["key"] for g in semantics["action"]["groups"]
    ]:
        raise ValueError("state/action keys and order must match")


def build_modality_json(semantics: dict[str, object]) -> dict[str, object]:
    def slices(name: str) -> dict[str, object]:
        original_key = str(semantics[name]["original_key"])
        return {
            str(g["key"]): {
                "start": int(g["start"]),
                "end": int(g["end"]),
                "original_key": original_key,
            }
            for g in semantics[name]["groups"]
        }

    language = semantics["language"]
    return {
        "state": slices("state"),
        "action": slices("action"),
        "video": {
            str(key): {"original_key": str(value["original_key"])}
            for key, value in semantics["video"].items()
        },
        "annotation": {
            str(language["annotation_key"]): {
                "original_key": str(language["modality_key"])
            }
        },
    }


def split_episode_ids(
    ids: Sequence[int], validation_fraction: float = 0.2, seed: int = 17
) -> dict[str, list[int]]:
    ordered = sorted(set(map(int, ids)))
    if len(ordered) != len(ids):
        raise ValueError("episode IDs must be unique")
    if not ordered:
        raise ValueError("episode ID list is empty")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    shuffled = ordered.copy()
    random.Random(seed).shuffle(shuffled)
    count = 0
    if len(shuffled) > 1 and validation_fraction:
        count = min(max(1, round(len(shuffled) * validation_fraction)), len(shuffled) - 1)
    return {"train": sorted(shuffled[count:]), "validation": sorted(shuffled[:count])}


LEROBOT_REQUIRED_FEATURES = {
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
}


def policy_feature_allowlist(semantics: dict[str, object]) -> set[str]:
    language = semantics["language"]
    return {
        str(semantics["state"]["original_key"]),
        str(semantics["action"]["original_key"]),
        *(str(value["original_key"]) for value in semantics["video"].values()),
        str(language["original_key"]),
        str(language["modality_key"]),
        *LEROBOT_REQUIRED_FEATURES,
    }


def prune_to_policy_features(
    root: Path, semantics: dict[str, object]
) -> dict[str, object]:
    info_path = root / "meta/info.json"
    info = load_json(info_path)
    allowed = policy_feature_allowlist(semantics)
    keys = sorted(set(map(str, info["features"])) - allowed)
    dropped_columns = 0
    parquet_files = 0
    for path in sorted(root.glob("data/**/*.parquet")):
        table = pq.read_table(path)
        present = [key for key in keys if key in table.column_names]
        if present:
            table = table.drop(present)
            temporary = path.with_suffix(".parquet.tmp")
            pq.write_table(table, temporary)
            os.replace(temporary, path)
            dropped_columns += len(present)
        parquet_files += 1

    removed_metadata = []
    for key in keys:
        if info["features"].pop(key, None) is not None:
            removed_metadata.append(key)
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    return {
        "allowlist": sorted(allowed),
        "removed_metadata": removed_metadata,
        "parquet_files": parquet_files,
        "dropped_column_instances": dropped_columns,
    }


def align_language_annotation(root: Path, language: dict[str, object]) -> dict[str, object]:
    source_key = str(language["original_key"])
    aligned_key = str(language["modality_key"])
    tasks = {}
    for line in (root / "meta/tasks.jsonl").read_text().splitlines():
        record = json.loads(line)
        tasks[int(record["task_index"])] = str(record["task"])
    if not tasks:
        raise ValueError("meta/tasks.jsonl contains no tasks")

    files = rows = 0
    for path in sorted(root.glob("data/**/*.parquet")):
        table = pq.read_table(path)
        if source_key not in table.column_names:
            raise KeyError(f"language source column missing in {path}: {source_key}")
        values = [int(value) for value in table[source_key].to_pylist()]
        unknown = sorted(set(values) - set(tasks))
        if unknown:
            raise ValueError(f"unknown task indices in {path}: {unknown}")
        column = pa.array(values, type=pa.int64())
        if aligned_key in table.column_names:
            table = table.set_column(table.schema.get_field_index(aligned_key), aligned_key, column)
        else:
            table = table.append_column(aligned_key, column)
        temporary = path.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary)
        os.replace(temporary, path)
        files += 1
        rows += len(table)

    info_path = root / "meta/info.json"
    info = load_json(info_path)
    info["features"][aligned_key] = {"dtype": "int64", "shape": [1], "names": None}
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    return {
        "source_key": source_key,
        "aligned_key": aligned_key,
        "tasks": {str(key): value for key, value in sorted(tasks.items())},
        "parquet_files": files,
        "rows": rows,
    }


def render_groot_config(semantics: dict[str, object]) -> str:
    timing = semantics["timing"]
    states = [g["key"] for g in semantics["state"]["groups"]]
    actions = semantics["action"]["groups"]
    action_lines = []
    for group in actions:
        arguments = [
            f"rep=ActionRepresentation.{group['training_representation']}",
            f"type=ActionType.{group['type']}",
            f"format=ActionFormat.{group['format']}",
        ]
        if group.get("state_key"):
            arguments.append(f"state_key={group['state_key']!r}")
        action_lines.append("            ActionConfig(" + ", ".join(arguments) + "),")
    lines = [
        "# Generated from robosyn_cobotmagic_semantics.json; do not hand-edit.",
        "from gr00t.configs.data.embodiment_configs import register_modality_config",
        "from gr00t.data.embodiment_tags import EmbodimentTag",
        "from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig",
        "",
        "robosyn_cobotmagic_config = {",
        f'    "video": ModalityConfig(delta_indices={timing["video_delta_indices"]!r}, modality_keys={list(semantics["video"])!r}),',
        f'    "state": ModalityConfig(delta_indices={timing["state_delta_indices"]!r}, modality_keys={states!r}),',
        '    "action": ModalityConfig(',
        f'        delta_indices={timing["action_delta_indices"]!r},',
        f'        modality_keys={[g["key"] for g in actions]!r},',
        "        action_configs=[",
        *action_lines,
        "        ],",
        "    ),",
        f'    "language": ModalityConfig(delta_indices=[0], modality_keys={[semantics["language"]["modality_key"]]!r}),',
        "}",
        "",
        "register_modality_config(robosyn_cobotmagic_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)",
        "",
    ]
    return "\n".join(lines)


def _make_writable(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        relative = path.relative_to(root)
        if relative.parts[:1] == ("videos",) and path.is_file():
            continue
        path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR)


def _copy_dataset_file(source: str, destination: str) -> str:
    source_path = Path(source)
    if "videos" in source_path.parts:
        try:
            os.link(source, destination)
            return destination
        except OSError as exc:
            if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.EMLINK}:
                raise
    return shutil.copy2(source, destination)


def _episode_ids(root: Path) -> list[int]:
    return [
        int(json.loads(line)["episode_index"])
        for line in (root / "meta/episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]


def prepare_dataset(src: Path, dst: Path, semantics: dict[str, object]) -> dict[str, object]:
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise FileExistsError(dst)
    info = load_json(src / "meta/info.json")
    state_dim = int(info["features"][semantics["state"]["original_key"]]["shape"][0])
    action_dim = int(info["features"][semantics["action"]["original_key"]]["shape"][0])
    validate_semantics(semantics, state_dim, action_dim)
    if float(info["fps"]) != float(semantics["timing"]["fps"]):
        raise ValueError("dataset FPS does not match semantic evidence")

    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{dst.name}.tmp-", dir=dst.parent))
    try:
        shutil.copytree(
            src, temporary, dirs_exist_ok=True, copy_function=_copy_dataset_file
        )
        _make_writable(temporary)
        pruning = prune_to_policy_features(temporary, semantics)
        language = align_language_annotation(temporary, semantics["language"])
        split = split_episode_ids(_episode_ids(temporary))
        (temporary / "meta/modality.json").write_text(
            json.dumps(build_modality_json(semantics), indent=2) + "\n"
        )
        (temporary / "meta/split.json").write_text(
            json.dumps({"seed": 17, "validation_fraction": 0.2, **split}, indent=2) + "\n"
        )
        semantic_bytes = json.dumps(semantics, sort_keys=True, separators=(",", ":")).encode()
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": str(src.resolve()),
            "destination": str(dst.resolve()),
            "semantics_sha256": hashlib.sha256(semantic_bytes).hexdigest(),
            "transformations": [
                "copied immutable source snapshot",
                "hardlinked immutable videos when supported by the filesystem",
                f"retained only policy/data-loader allowlist features; removed: {pruning['removed_metadata']}",
                f"aligned {language['source_key']} to {language['aligned_key']}",
                "added modality metadata",
                "added deterministic whole-episode split",
            ],
            "pruning": pruning,
            "language": language,
            "split": split,
            "state_dimension": state_dim,
            "action_dimension": action_dim,
        }
        (temporary / "meta/preparation_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        (temporary / "README.md").write_text(
            "# Prepared RoboSyn Click-the-Bell Dataset\n\n"
            "Derived GR00T-ready copy; raw source remains immutable.\n"
            f"Action horizon: {len(semantics['timing']['action_delta_indices'])} frames.\n"
            f"Whole-episode split seed 17: train={len(split['train'])}, "
            f"validation={len(split['validation'])}.\n"
        )
        os.replace(temporary, dst)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--semantics", required=True, type=Path)
    parser.add_argument("--config-output", type=Path)
    args = parser.parse_args(argv)
    semantics = load_json(args.semantics)
    report = prepare_dataset(args.src, args.dst, semantics)
    if args.config_output:
        args.config_output.parent.mkdir(parents=True, exist_ok=True)
        args.config_output.write_text(render_groot_config(semantics))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
