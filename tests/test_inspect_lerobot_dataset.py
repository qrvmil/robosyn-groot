import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.inspect_lerobot_dataset import inspect_dataset


@pytest.fixture
def dataset_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)

    info = {
        "codebase_version": "v2.1",
        "total_episodes": 2,
        "total_frames": 5,
        "total_tasks": 1,
        "fps": 25,
        "features": {
            "observation.qpos": {"dtype": "float32", "shape": [2]},
            "action": {"dtype": "float32", "shape": [2]},
            "timestamp": {"dtype": "float32", "shape": [1]},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/tasks.jsonl").write_text('{"task_index":0,"task":"test task"}\n')
    (root / "meta/episodes.jsonl").write_text(
        '{"episode_index":0,"tasks":["test task"],"length":3}\n'
        '{"episode_index":1,"tasks":["test task"],"length":2}\n'
    )

    for episode, length in [(0, 3), (1, 2)]:
        base = episode * 10
        table = pa.table(
            {
                "observation.qpos": pa.array(
                    [[base + i, base + i + 0.5] for i in range(length)],
                    type=pa.list_(pa.float32(), 2),
                ),
                "action": pa.array(
                    [[0.0, 0.0] if i == 0 else [i, -i] for i in range(length)],
                    type=pa.list_(pa.float32(), 2),
                ),
                "timestamp": pa.array(np.arange(length, dtype=np.float32) / 25),
                "frame_index": pa.array(range(length), type=pa.int64()),
                "episode_index": pa.array([episode] * length, type=pa.int64()),
                "task_index": pa.array([0] * length, type=pa.int64()),
            }
        )
        pq.write_table(table, root / f"data/chunk-000/episode_{episode:06d}.parquet")
    return root


def test_inspect_reports_episode_and_numeric_statistics(dataset_fixture: Path):
    report = inspect_dataset(dataset_fixture, probe_videos=False)

    assert report["episodes"]["count"] == 2
    assert report["episodes"]["lengths"] == [3, 2]
    assert report["frames"]["count"] == 5
    assert report["fields"]["action"]["shape"] == [2]
    assert report["fields"]["action"]["nan_count"] == 0
    assert report["fields"]["action"]["inf_count"] == 0
    assert report["fields"]["action"]["near_zero_frame_count"] == 2
    assert report["timing"]["metadata_fps"] == 25
    assert report["timing"]["median_dt"] == pytest.approx(0.04)


def test_inspect_rejects_missing_info_json(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="meta/info.json"):
        inspect_dataset(tmp_path, probe_videos=False)


def test_inspect_detects_metadata_frame_mismatch(dataset_fixture: Path):
    info_path = dataset_fixture / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["total_frames"] = 6
    info_path.write_text(json.dumps(info))

    report = inspect_dataset(dataset_fixture, probe_videos=False)

    assert report["reconciliation"]["frames_match"] is False
