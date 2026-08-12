#!/usr/bin/env python3
"""Render synchronized LeRobot camera/state/action episode overlays."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq


def select_spread_episode_ids(ids: list[int], count: int) -> list[int]:
    if count < 1:
        raise ValueError("count must be positive")
    ordered = sorted(set(ids))
    if not ordered:
        raise ValueError("episode ID list is empty")
    indices = np.rint(np.linspace(0, len(ordered) - 1, min(count, len(ordered)))).astype(int)
    return [ordered[index] for index in indices]


def _episode_path(root: Path, episode_id: int) -> Path:
    matches = list(root.glob(f"data/**/episode_{episode_id:06d}.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"episode {episode_id:06d} parquet: expected one file, found {len(matches)}"
        )
    return matches[0]


def _video_path(root: Path, episode_id: int, camera_key: str) -> Path:
    matches = list(root.glob(f"videos/**/{camera_key}/episode_{episode_id:06d}.mp4"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"episode {episode_id:06d} camera {camera_key}: expected one MP4, found {len(matches)}"
        )
    return matches[0]


def decode_video_frames(path: Path) -> tuple[list[np.ndarray], float]:
    """Decode a video with the system FFmpeg rather than OpenCV's bundled codec."""
    path = Path(path)
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,nb_frames",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(probe.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"expected one video stream in {path}, found {len(streams)}")
    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])
    fps = float(Fraction(stream["avg_frame_rate"]))
    expected_frames = int(stream["nb_frames"])
    decoded = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
        ],
        check=True,
        capture_output=True,
    ).stdout
    frame_bytes = width * height * 3
    if len(decoded) % frame_bytes:
        raise RuntimeError(f"decoded byte count is not frame-aligned for {path}")
    frame_array = np.frombuffer(decoded, dtype=np.uint8).reshape(-1, height, width, 3)
    if len(frame_array) != expected_frames:
        raise RuntimeError(
            f"decoded frame count mismatch for {path}: {len(frame_array)} != {expected_frames}"
        )
    return list(frame_array), fps


def _task_text(root: Path, task_index: int) -> str:
    tasks_path = root / "meta/tasks.jsonl"
    for line in tasks_path.read_text().splitlines():
        record = json.loads(line)
        if int(record["task_index"]) == task_index:
            return str(record["task"])
    raise KeyError(f"task index {task_index} absent from {tasks_path}")


def _vector_text(name: str, values: np.ndarray) -> str:
    formatted = np.array2string(
        values,
        precision=3,
        suppress_small=True,
        separator=",",
        max_line_width=200,
    )
    return f"{name}: {formatted}"


def render_episode(
    root: Path,
    episode_id: int,
    output: Path,
    camera_keys: list[str],
    state_key: str,
    action_key: str,
) -> None:
    root = Path(root)
    table = pq.read_table(_episode_path(root, episode_id))
    for key in (state_key, action_key, "timestamp", "frame_index", "task_index"):
        if key not in table.column_names:
            raise KeyError(f"required parquet column missing: {key}")
    states = np.asarray(table[state_key].to_pylist(), dtype=np.float32)
    actions = np.asarray(table[action_key].to_pylist(), dtype=np.float32)
    timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float32)
    frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    task_indices = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)
    task = _task_text(root, int(task_indices[0]))

    decoded = [decode_video_frames(_video_path(root, episode_id, key)) for key in camera_keys]
    videos = [frames for frames, _ in decoded]
    fps_values = [fps for _, fps in decoded]
    frame_counts = [len(frames) for frames in videos]
    if any(count != len(table) for count in frame_counts):
        raise ValueError(f"video/parquet frame mismatch: video={frame_counts}, parquet={len(table)}")
    if max(fps_values) - min(fps_values) > 1e-6:
        raise ValueError(f"camera FPS mismatch: {fps_values}")

    target_height = min(frames[0].shape[0] for frames in videos)
    view_width = int(videos[0][0].shape[1] * target_height / videos[0][0].shape[0])
    canvas_width = view_width * len(camera_keys)
    panel_height = 150
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_values[0],
        (canvas_width, target_height + panel_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to create output video: {output}")
    try:
        for row in range(len(table)):
            pending = [frames[row] for frames in videos]
            views: list[np.ndarray] = []
            for key, frame in zip(camera_keys, pending):
                resized = cv2.resize(frame, (view_width, target_height))
                cv2.putText(resized, key, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                views.append(resized)
            canvas = np.zeros((target_height + panel_height, canvas_width, 3), dtype=np.uint8)
            canvas[:target_height] = np.hstack(views)
            lines = [
                f"episode={episode_id} frame={int(frame_indices[row])} timestamp={timestamps[row]:.3f}s task={task}",
                _vector_text(state_key, states[row]),
                _vector_text(action_key, actions[row]),
            ]
            for line_index, line in enumerate(lines):
                cv2.putText(
                    canvas,
                    line,
                    (12, target_height + 35 + line_index * 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            writer.write(canvas)
    finally:
        writer.release()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--episode-id", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--camera-keys", nargs="+", required=True)
    parser.add_argument("--state-key", default="observation.qpos")
    parser.add_argument("--action-key", default="action")
    args = parser.parse_args(argv)
    render_episode(
        args.dataset,
        args.episode_id,
        args.output,
        args.camera_keys,
        args.state_key,
        args.action_key,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
