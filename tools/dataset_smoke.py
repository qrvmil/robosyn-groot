#!/usr/bin/env python3
"""Exercise the official GR00T episode loader and a real torch DataLoader batch."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag


def load_config(path: Path) -> dict[str, object]:
    spec = importlib.util.spec_from_file_location("robosyn_dataset_smoke_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.robosyn_cobotmagic_config


class StepDataset(Dataset):
    def __init__(self, steps):
        self.steps = steps

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, index):
        return self.steps[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    loader = LeRobotEpisodeLoader(args.dataset, config)
    episode = loader[0]
    horizon = len(config["action"].delta_indices)
    steps = [
        extract_step_data(episode, index, config, EmbodimentTag.NEW_EMBODIMENT)
        for index in range(2)
    ]
    batch = next(iter(DataLoader(StepDataset(steps), batch_size=2, collate_fn=list)))

    first = batch[0]
    expected_state = {"left_arm": 6, "left_gripper": 1, "right_arm": 6, "right_gripper": 1}
    expected_video = {"front", "left_wrist", "right_wrist"}
    if set(first.states) != set(expected_state):
        raise AssertionError(f"state keys differ: {set(first.states)}")
    if set(first.actions) != set(expected_state):
        raise AssertionError(f"action keys differ: {set(first.actions)}")
    if set(first.images) != expected_video:
        raise AssertionError(f"camera keys differ: {set(first.images)}")
    for key, width in expected_state.items():
        if first.states[key].shape != (1, width):
            raise AssertionError(f"state.{key} shape={first.states[key].shape}")
        if first.actions[key].shape != (horizon, width):
            raise AssertionError(f"action.{key} shape={first.actions[key].shape}")
        if not np.isfinite(first.states[key]).all() or not np.isfinite(first.actions[key]).all():
            raise AssertionError(f"non-finite state/action values for {key}")
    camera_shapes = {}
    for key, frames in first.images.items():
        array = np.asarray(frames)
        camera_shapes[key] = list(array.shape)
        if array.shape[0] != 1 or array.shape[-1] != 3:
            raise AssertionError(f"video.{key} shape={array.shape}")
        if not np.isfinite(array).all():
            raise AssertionError(f"non-finite camera values for {key}")
    if not isinstance(first.text, str) or not first.text.strip():
        raise AssertionError(f"invalid language instruction: {first.text!r}")

    result = {
        "status": "pass",
        "dataset": str(args.dataset.resolve()),
        "episodes": len(loader),
        "data_loader_batch_size": len(batch),
        "action_horizon": horizon,
        "state_shapes": {key: list(value.shape) for key, value in first.states.items()},
        "action_shapes": {key: list(value.shape) for key, value in first.actions.items()},
        "camera_shapes": camera_shapes,
        "camera_keys": sorted(first.images),
        "language": first.text,
        "all_finite": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
