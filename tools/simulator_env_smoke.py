#!/usr/bin/env python3
"""Create a real RoboSyn env, reset it, and execute one closed-loop hold step."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ROBOSYN = ROOT / "repos/RoboSynChallenge"
sys.path.insert(0, str(ROBOSYN))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="drawer_open_place")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROBOSYN)

    from scripts.eval_policy import (
        find_action_config,
        find_gym_config,
        make_env_from_configs,
    )

    config = {
        "task_name": args.task,
        "setting": "clear",
        "num_envs": 1,
        "device": "cuda",
        "headless": True,
        "renderer": "hybrid",
        "gpu_id": 0,
        "filter_dataset_saving": True,
        "max_steps": None,
    }
    gym_config = find_gym_config(config)
    action_config = find_action_config(config)
    env, effective_config = make_env_from_configs(config, gym_config, action_config)
    result = None
    try:
        obs, info = env.reset(seed=0)
        qpos = obs["robot"]["qpos"]
        if tuple(qpos.shape) != (1, 14):
            raise ValueError(f"unexpected qpos shape: {tuple(qpos.shape)}")
        camera_shapes = {}
        for key in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            image = obs["sensor"][key]["color"]
            array = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
            if array.ndim != 4 or array.shape[0] != 1 or array.shape[-1] < 3:
                raise ValueError(f"unexpected {key} shape: {array.shape}")
            camera_shapes[key] = list(array.shape)
        next_obs, _, terminated, truncated, next_info = env.step(qpos.clone())
        success = env.get_wrapper_attr("is_task_success")()
        result = {
            "status": "pass",
            "task": args.task,
            "env_id": effective_config["id"],
            "seed": 0,
            "qpos_shape": list(qpos.shape),
            "camera_shapes": camera_shapes,
            "elapsed_steps": int(next_info["elapsed_steps"].item()),
            "terminated": bool(terminated.any().item()),
            "truncated": bool(truncated.any().item()),
            "official_success_signal_shape": list(success.shape),
            "next_qpos_finite": bool(np.isfinite(next_obs["robot"]["qpos"].detach().cpu().numpy()).all()),
        }
        # DexSim may terminate the interpreter as part of engine shutdown, so
        # persist evidence before env.close().
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
