"""Strict schema detection for RoboSyn CobotMagic LeRobot datasets."""

from __future__ import annotations

from dataclasses import dataclass


COBOTMAGIC_JOINT_NAMES = tuple(
    [f"LEFT_JOINT{index}" for index in range(1, 8)]
    + [f"RIGHT_JOINT{index}" for index in range(1, 8)]
)

STATE_CANDIDATES = ("observation.qpos", "observation.state")
CAMERA_CANDIDATES = {
    "front": ("observation.images.cam_high", "cam_high.color"),
    "left_wrist": (
        "observation.images.cam_left_wrist",
        "cam_left_wrist.color",
    ),
    "right_wrist": (
        "observation.images.cam_right_wrist",
        "cam_right_wrist.color",
    ),
}


@dataclass(frozen=True)
class DetectedSchema:
    state_key: str
    action_key: str
    camera_keys: dict[str, str]
    state_dimension: int
    action_dimension: int
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]


def _validated_joint_names(feature: dict[str, object], label: str) -> tuple[str, ...]:
    if feature.get("shape") != [14]:
        raise ValueError(f"{label} must have shape [14], got {feature.get('shape')}")
    if "float" not in str(feature.get("dtype", "")).lower():
        raise ValueError(f"{label} must use a floating dtype, got {feature.get('dtype')}")
    names = tuple(map(str, feature.get("names") or ()))
    if names != COBOTMAGIC_JOINT_NAMES:
        raise ValueError(
            f"{label} joint names/order are not the verified CobotMagic order: {names}"
        )
    return names


def _detect_camera(features: dict[str, dict[str, object]], logical: str) -> str:
    failures = []
    for key in CAMERA_CANDIDATES[logical]:
        feature = features.get(key)
        if feature is None:
            continue
        if str(feature.get("dtype", "")).lower() != "video":
            failures.append(f"{key}: dtype={feature.get('dtype')}")
            continue
        shape = feature.get("shape")
        if not isinstance(shape, list) or len(shape) != 3 or shape[-1] != 3:
            failures.append(f"{key}: shape={shape}")
            continue
        return key
    raise ValueError(
        f"no valid {logical} camera among {CAMERA_CANDIDATES[logical]}; "
        f"failures={failures}"
    )


def detect_schema(features: dict[str, dict[str, object]]) -> DetectedSchema:
    action_key = "action"
    if action_key not in features:
        raise KeyError("action feature is missing")
    action_names = _validated_joint_names(features[action_key], "action")

    state_key = None
    state_names: tuple[str, ...] = ()
    failures = []
    for candidate in STATE_CANDIDATES:
        if candidate not in features:
            continue
        try:
            candidate_names = _validated_joint_names(features[candidate], candidate)
        except ValueError as exc:
            failures.append(str(exc))
            continue
        if candidate_names != action_names:
            failures.append(f"{candidate} joint names/order differ from action")
            continue
        state_key, state_names = candidate, candidate_names
        break
    if state_key is None:
        raise ValueError(
            "no verified 14-D state feature with matching joint names/order; "
            + "; ".join(failures)
        )

    return DetectedSchema(
        state_key=state_key,
        action_key=action_key,
        camera_keys={
            logical: _detect_camera(features, logical) for logical in CAMERA_CANDIDATES
        },
        state_dimension=14,
        action_dimension=14,
        state_names=state_names,
        action_names=action_names,
    )
