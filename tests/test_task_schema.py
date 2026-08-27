import copy
import json

import pytest

from tools.task_schema import COBOTMAGIC_JOINT_NAMES, detect_schema
from tools.generate_task_semantics import generate_semantics
from tools.prepare_robosyn_for_groot import render_groot_config


def features_with_state(state_key: str) -> dict[str, dict[str, object]]:
    joint_feature = {
        "dtype": "float32",
        "shape": [14],
        "names": list(COBOTMAGIC_JOINT_NAMES),
    }
    return {
        state_key: copy.deepcopy(joint_feature),
        "action": copy.deepcopy(joint_feature),
        "observation.images.cam_high": {
            "dtype": "video",
            "shape": [480, 640, 3],
        },
        "observation.images.cam_left_wrist": {
            "dtype": "video",
            "shape": [480, 640, 3],
        },
        "observation.images.cam_right_wrist": {
            "dtype": "video",
            "shape": [480, 640, 3],
        },
    }


@pytest.mark.parametrize("state_key", ["observation.qpos", "observation.state"])
def test_detect_schema_accepts_verified_state_variants(state_key: str):
    schema = detect_schema(features_with_state(state_key))

    assert schema.state_key == state_key
    assert schema.action_key == "action"
    assert schema.state_dimension == 14
    assert schema.action_dimension == 14


def test_detect_schema_maps_exact_three_real_cameras():
    schema = detect_schema(features_with_state("observation.state"))

    assert schema.camera_keys == {
        "front": "observation.images.cam_high",
        "left_wrist": "observation.images.cam_left_wrist",
        "right_wrist": "observation.images.cam_right_wrist",
    }


def test_detect_schema_rejects_state_action_name_mismatch():
    features = features_with_state("observation.state")
    features["action"]["names"] = list(reversed(COBOTMAGIC_JOINT_NAMES))

    with pytest.raises(ValueError, match="joint names/order"):
        detect_schema(features)


def test_detect_schema_does_not_select_invalid_named_candidate():
    features = features_with_state("observation.state")
    features["observation.qpos"] = {"dtype": "float32", "shape": [13]}

    schema = detect_schema(features)

    assert schema.state_key == "observation.state"


def test_generate_task_semantics_uses_detected_schema_and_metadata_language(tmp_path):
    raw = tmp_path / "raw"
    (raw / "meta").mkdir(parents=True)
    (raw / "meta/info.json").write_text(
        json.dumps({"fps": 25, "features": features_with_state("observation.state")})
    )
    (raw / "meta/tasks.jsonl").write_text(
        '{"task_index":3,"task":"Open the drawer and place the tomato."}\n'
    )
    base = {
        "schema_version": 1,
        "dataset": {},
        "timing": {
            "fps": 25,
            "video_delta_indices": [0],
            "state_delta_indices": [0],
            "action_delta_indices": list(range(13)),
        },
        "state": {"original_key": "observation.qpos", "groups": []},
        "action": {"original_key": "action", "groups": []},
        "video": {
            "front": {},
            "left_wrist": {},
            "right_wrist": {},
        },
        "language": {
            "original_key": "task_index",
            "modality_key": "annotation.human.task_description",
            "annotation_key": "human.task_description",
        },
    }

    generated = generate_semantics(
        base,
        raw,
        task="drawer_open_place",
        repo="RoboSynChallenge/cobotmagic_Sim_drawer_open_place",
        revision="abc123",
    )

    assert generated["state"]["original_key"] == "observation.state"
    assert generated["action"]["original_key"] == "action"
    assert generated["language"]["task_index"] == 3
    assert generated["language"]["task"] == "Open the drawer and place the tomato."
    assert generated["dataset"]["revision"] == "abc123"
    assert generated["evidence"]["state_action_names_match"] is True


def test_rendered_task_config_keeps_13_step_action_horizon():
    semantics = {
        "timing": {
            "video_delta_indices": [0],
            "state_delta_indices": [0],
            "action_delta_indices": list(range(13)),
        },
        "state": {
            "original_key": "observation.state",
            "groups": [
                {"key": "left_arm"},
                {"key": "left_gripper"},
                {"key": "right_arm"},
                {"key": "right_gripper"},
            ],
        },
        "action": {
            "original_key": "action",
            "groups": [
                {"key": "left_arm", "training_representation": "RELATIVE", "type": "NON_EEF", "format": "DEFAULT", "state_key": "left_arm"},
                {"key": "left_gripper", "training_representation": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT"},
                {"key": "right_arm", "training_representation": "RELATIVE", "type": "NON_EEF", "format": "DEFAULT", "state_key": "right_arm"},
                {"key": "right_gripper", "training_representation": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT"},
            ],
        },
        "video": {"front": {}, "left_wrist": {}, "right_wrist": {}},
        "language": {"modality_key": "annotation.human.task_description"},
    }

    config = render_groot_config(semantics)

    assert "delta_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]" in config
    assert "ActionRepresentation.RELATIVE" in config
    assert "ActionRepresentation.ABSOLUTE" in config
