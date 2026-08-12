import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.subset_lerobot_v21 import create_subset


@pytest.fixture
def dataset_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    for episode in (0, 1):
        (root / "data/chunk-000").mkdir(parents=True, exist_ok=True)
        (root / "videos/chunk-000/front").mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "episode_index": pa.array([episode, episode], type=pa.int64()),
                "frame_index": pa.array([0, 1], type=pa.int64()),
                "action": pa.array([[0.0], [1.0]], type=pa.list_(pa.float32(), 1)),
            }
        )
        pq.write_table(table, root / f"data/chunk-000/episode_{episode:06d}.parquet")
        (root / f"videos/chunk-000/front/episode_{episode:06d}.mp4").write_bytes(
            b"video" + bytes([episode])
        )
    info = {
        "total_episodes": 2,
        "total_frames": 4,
        "total_videos": 2,
        "total_chunks": 1,
        "chunks_size": 1000,
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/episodes.jsonl").write_text(
        '{"episode_index":0,"tasks":["test"],"length":2}\n'
        '{"episode_index":1,"tasks":["test"],"length":2}\n'
    )
    (root / "meta/episodes_stats.jsonl").write_text(
        '{"episode_index":0,"stats":{}}\n'
        '{"episode_index":1,"stats":{}}\n'
    )
    (root / "meta/stats.json").write_text("{}")
    (root / "meta/relative_stats.json").write_text("{}")
    (root / "meta/tasks.jsonl").write_text('{"task_index":0,"task":"test"}\n')
    (root / "meta/modality.json").write_text("{}")
    return root


def test_subset_keeps_requested_episode_and_resets_stats(dataset_fixture: Path, tmp_path: Path):
    destination = tmp_path / "tiny"

    report = create_subset(dataset_fixture, destination, [1])

    assert report["episode_ids"] == [1]
    assert report["frames"] == 2
    assert (destination / "data/chunk-000/episode_000001.parquet").is_file()
    assert not (destination / "data/chunk-000/episode_000000.parquet").exists()
    assert (destination / "videos/chunk-000/front/episode_000001.mp4").is_file()
    assert not (destination / "meta/stats.json").exists()
    assert not (destination / "meta/relative_stats.json").exists()
    info = json.loads((destination / "meta/info.json").read_text())
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 2
    assert info["total_videos"] == 1


def test_subset_rejects_unknown_episode(dataset_fixture: Path, tmp_path: Path):
    with pytest.raises(ValueError, match="episode 99"):
        create_subset(dataset_fixture, tmp_path / "tiny", [99])
