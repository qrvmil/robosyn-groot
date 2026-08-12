#!/usr/bin/env python3
"""Audit a LeRobot v2/v2.1 dataset without mutating it."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_metadata(root: Path) -> dict[str, object]:
    root = Path(root)
    info_path = root / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing required metadata: {info_path}")
    return {
        "info": _read_json(info_path),
        "episodes": _read_jsonl(root / "meta/episodes.jsonl"),
        "tasks": _read_jsonl(root / "meta/tasks.jsonl"),
    }


def iter_episode_tables(root: Path) -> Iterator[tuple[Path, pa.Table]]:
    files = sorted(Path(root).glob("data/**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {Path(root) / 'data'}")
    for path in files:
        yield path, pq.read_table(path)


def _column_to_numpy(column: pa.ChunkedArray) -> np.ndarray:
    array = column.combine_chunks()
    if pa.types.is_fixed_size_list(array.type):
        shape: list[int] = []
        values: pa.Array = array
        while pa.types.is_fixed_size_list(values.type):
            shape.append(values.type.list_size)
            values = values.values
        return np.asarray(values.to_numpy(zero_copy_only=False)).reshape(len(array), *shape)
    if pa.types.is_integer(array.type) or pa.types.is_floating(array.type):
        return np.asarray(array.to_numpy(zero_copy_only=False))
    raise TypeError(f"column is not numeric: {array.type}")


def summarize_numeric_column(table: pa.Table, name: str) -> dict[str, object]:
    values = _column_to_numpy(table[name])
    matrix = values.reshape(len(values), -1)
    finite = np.isfinite(matrix)
    finite_values = matrix[finite]
    near_zero_frames = np.all(np.abs(matrix) <= 1e-6, axis=1)
    return {
        "shape": list(values.shape[1:]) or [1],
        "count": int(matrix.size),
        "frame_count": int(len(matrix)),
        "nan_count": int(np.isnan(matrix).sum()),
        "inf_count": int(np.isinf(matrix).sum()),
        "min": float(finite_values.min()) if finite_values.size else None,
        "max": float(finite_values.max()) if finite_values.size else None,
        "sum": float(finite_values.sum(dtype=np.float64)),
        "sum_squares": float(np.square(finite_values, dtype=np.float64).sum()),
        "finite_count": int(finite_values.size),
        "near_zero_frame_count": int(near_zero_frames.sum()),
    }


def probe_video(path: Path) -> dict[str, object]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    streams = json.loads(completed.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"expected one video stream in {path}, got {len(streams)}")
    return streams[0]


def _merge_numeric(parts: list[dict[str, object]]) -> dict[str, object]:
    if not parts:
        raise ValueError("cannot merge an empty numeric summary")
    shape = parts[0]["shape"]
    if any(part["shape"] != shape for part in parts):
        raise ValueError(f"inconsistent shapes: {[part['shape'] for part in parts]}")
    count = sum(int(part["count"]) for part in parts)
    finite_count = sum(int(part["finite_count"]) for part in parts)
    total = sum(float(part["sum"]) for part in parts)
    total_sq = sum(float(part["sum_squares"]) for part in parts)
    mean = total / finite_count if finite_count else None
    variance = max(total_sq / finite_count - mean * mean, 0.0) if finite_count else None
    minima = [float(part["min"]) for part in parts if part["min"] is not None]
    maxima = [float(part["max"]) for part in parts if part["max"] is not None]
    frames = sum(int(part["frame_count"]) for part in parts)
    near_zero = sum(int(part["near_zero_frame_count"]) for part in parts)
    return {
        "shape": shape,
        "count": count,
        "frame_count": frames,
        "nan_count": sum(int(part["nan_count"]) for part in parts),
        "inf_count": sum(int(part["inf_count"]) for part in parts),
        "min": min(minima) if minima else None,
        "max": max(maxima) if maxima else None,
        "mean": mean,
        "std": math.sqrt(variance) if variance is not None else None,
        "near_zero_frame_count": near_zero,
        "near_zero_frame_rate": near_zero / frames if frames else None,
    }


def _numeric_names(table: pa.Table) -> list[str]:
    names: list[str] = []
    for field in table.schema:
        leaf = field.type
        while pa.types.is_fixed_size_list(leaf):
            leaf = leaf.value_type
        if pa.types.is_integer(leaf) or pa.types.is_floating(leaf):
            names.append(field.name)
    return names


def inspect_dataset(root: Path, *, probe_videos: bool = True) -> dict[str, object]:
    root = Path(root)
    metadata = load_metadata(root)
    info = metadata["info"]
    episode_records = metadata["episodes"]
    field_parts: dict[str, list[dict[str, object]]] = {}
    schemas: Counter[str] = Counter()
    parquet_lengths: list[int] = []
    episode_ids: list[int] = []
    all_timestamp_diffs: list[np.ndarray] = []

    for _, table in iter_episode_tables(root):
        schemas[str(table.schema)] += 1
        parquet_lengths.append(len(table))
        if "episode_index" in table.column_names and len(table):
            values = _column_to_numpy(table["episode_index"])
            unique = np.unique(values)
            if len(unique) != 1:
                raise ValueError(f"parquet contains multiple episode IDs: {unique.tolist()}")
            episode_ids.append(int(unique[0]))
        if "timestamp" in table.column_names:
            timestamps = _column_to_numpy(table["timestamp"]).reshape(-1)
            if len(timestamps) > 1:
                all_timestamp_diffs.append(np.diff(timestamps.astype(np.float64)))
        for name in _numeric_names(table):
            field_parts.setdefault(name, []).append(summarize_numeric_column(table, name))

    fields = {name: _merge_numeric(parts) for name, parts in sorted(field_parts.items())}
    frames = sum(parquet_lengths)
    episode_meta_lengths = [int(record["length"]) for record in episode_records]
    expected_episodes = int(info.get("total_episodes", len(parquet_lengths)))
    expected_frames = int(info.get("total_frames", frames))
    diffs = np.concatenate(all_timestamp_diffs) if all_timestamp_diffs else np.array([])

    videos: dict[str, object] = {
        "count": 0,
        "camera_counts": {},
        "metadata_variants": [],
        "decode_failures": [],
    }
    video_files = sorted(root.glob("videos/**/*.mp4"))
    videos["count"] = len(video_files)
    camera_counts = Counter(path.parent.name for path in video_files)
    videos["camera_counts"] = dict(sorted(camera_counts.items()))
    if probe_videos:
        variants: Counter[str] = Counter()
        failures: list[dict[str, str]] = []
        for path in video_files:
            try:
                result = probe_video(path)
                variants[json.dumps(result, sort_keys=True)] += 1
            except (subprocess.CalledProcessError, ValueError, json.JSONDecodeError) as exc:
                failures.append({"path": str(path.relative_to(root)), "error": str(exc)})
        videos["metadata_variants"] = [
            {"count": count, "stream": json.loads(key)} for key, count in sorted(variants.items())
        ]
        videos["decode_failures"] = failures

    report: dict[str, object] = {
        "dataset_root": str(root.resolve()),
        "codebase_version": info.get("codebase_version"),
        "episodes": {
            "count": len(parquet_lengths),
            "ids": episode_ids,
            "lengths": parquet_lengths,
            "min_length": min(parquet_lengths),
            "max_length": max(parquet_lengths),
            "mean_length": float(np.mean(parquet_lengths)),
        },
        "frames": {"count": frames},
        "tasks": metadata["tasks"],
        "timing": {
            "metadata_fps": info.get("fps"),
            "median_dt": float(np.median(diffs)) if diffs.size else None,
            "min_dt": float(diffs.min()) if diffs.size else None,
            "max_dt": float(diffs.max()) if diffs.size else None,
        },
        "schemas": [{"count": count, "schema": schema} for schema, count in schemas.items()],
        "fields": fields,
        "videos": videos,
        "reconciliation": {
            "episodes_match": len(parquet_lengths) == expected_episodes,
            "frames_match": frames == expected_frames,
            "episode_metadata_count_match": not episode_records
            or len(episode_records) == expected_episodes,
            "episode_lengths_match": not episode_meta_lengths
            or episode_meta_lengths == parquet_lengths,
            "unique_episode_ids": len(set(episode_ids)) == len(episode_ids),
        },
    }
    return report


def _validate_json_numbers(value: object, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON number at {path}: {value}")
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_json_numbers(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_json_numbers(child, f"{path}[{index}]")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--skip-video-probe", action="store_true")
    args = parser.parse_args(argv)
    report = inspect_dataset(args.dataset, probe_videos=not args.skip_video_probe)
    _validate_json_numbers(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
