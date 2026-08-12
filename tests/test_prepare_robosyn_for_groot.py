import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.prepare_robosyn_for_groot import prepare_dataset


@pytest.fixture
def dataset_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "fixture",
        "total_episodes": 3,
        "total_frames": 6,
        "total_tasks": 1,
        "fps": 25,
        "features": {
            "observation.qpos": {"dtype": "float32", "shape": [4]},
            "action": {"dtype": "float32", "shape": [4]},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/tasks.jsonl").write_text('{"task_index":0,"task":"Click the bell"}\n')
    (root / "meta/episodes.jsonl").write_text(
        ''.join(
            json.dumps({"episode_index": episode, "tasks": ["Click the bell"], "length": 2}) + "\n"
            for episode in range(3)
        )
    )
    for episode in range(3):
        table = pa.table(
            {
                "observation.qpos": pa.array(
                    [[episode, 0.0, episode + 0.5, 0.0]] * 2,
                    type=pa.list_(pa.float32(), 4),
                ),
                "action": pa.array(
                    [[episode, 0.0, episode + 0.5, 0.0]] * 2,
                    type=pa.list_(pa.float32(), 4),
                ),
                "task_index": pa.array([0, 0], type=pa.int64()),
                "episode_index": pa.array([episode, episode], type=pa.int64()),
                "frame_index": pa.array([0, 1], type=pa.int64()),
                "timestamp": pa.array(np.array([0.0, 0.04], dtype=np.float32)),
            }
        )
        pq.write_table(table, root / f"data/chunk-000/episode_{episode:06d}.parquet")
    return root


@pytest.fixture
def semantics() -> dict[str, object]:
    return {
        "schema_version": 1,
        "timing": {
            "fps": 25,
            "video_delta_indices": [0],
            "state_delta_indices": [0],
            "action_delta_indices": [0, 1],
        },
        "state": {
            "original_key": "observation.qpos",
            "groups": [
                {"key": "left_arm", "start": 0, "end": 1},
                {"key": "left_gripper", "start": 1, "end": 2},
                {"key": "right_arm", "start": 2, "end": 3},
                {"key": "right_gripper", "start": 3, "end": 4},
            ],
        },
        "action": {
            "original_key": "action",
            "groups": [
                {"key": "left_arm", "start": 0, "end": 1, "training_representation": "RELATIVE", "type": "NON_EEF", "format": "DEFAULT", "state_key": "left_arm"},
                {"key": "left_gripper", "start": 1, "end": 2, "training_representation": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT"},
                {"key": "right_arm", "start": 2, "end": 3, "training_representation": "RELATIVE", "type": "NON_EEF", "format": "DEFAULT", "state_key": "right_arm"},
                {"key": "right_gripper", "start": 3, "end": 4, "training_representation": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT"},
            ],
        },
        "video": {
            "front": {"original_key": "cam_high.color"},
            "left_wrist": {"original_key": "cam_left_wrist.color"},
            "right_wrist": {"original_key": "cam_right_wrist.color"},
        },
        "language": {
            "modality_key": "annotation.human.task_description",
            "annotation_key": "human.task_description",
            "original_key": "task_index",
        },
    }


def test_prepare_refuses_existing_destination(dataset_fixture: Path, semantics, tmp_path: Path):
    destination = tmp_path / "prepared"
    destination.mkdir()

    with pytest.raises(FileExistsError):
        prepare_dataset(dataset_fixture, destination, semantics)


def test_prepare_aligns_language_split_and_preserves_raw(dataset_fixture: Path, semantics, tmp_path: Path):
    destination = tmp_path / "prepared"

    report = prepare_dataset(dataset_fixture, destination, semantics)

    modality = json.loads((destination / "meta/modality.json").read_text())
    assert modality["annotation"]["human.task_description"]["original_key"] == "annotation.human.task_description"
    split = report["split"]
    assert set(split["train"]).isdisjoint(split["validation"])
    assert sorted(split["train"] + split["validation"]) == [0, 1, 2]
    prepared_table = pq.read_table(destination / "data/chunk-000/episode_000000.parquet")
    assert prepared_table["annotation.human.task_description"].to_pylist() == [0, 0]
    raw_table = pq.read_table(dataset_fixture / "data/chunk-000/episode_000000.parquet")
    assert "annotation.human.task_description" not in raw_table.column_names
    info = json.loads((destination / "meta/info.json").read_text())
    assert info["features"]["annotation.human.task_description"] == {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    }
    assert (destination / "meta/preparation_manifest.json").is_file()
