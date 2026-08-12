#!/usr/bin/env python3
"""Record denormalized open-loop predictions from the pinned GR00T checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.utils import parse_observation_gr00t
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np


def _concat_columns(traj, columns: list[str]) -> np.ndarray:
    return np.concatenate(
        [np.vstack([np.asarray(value) for value in traj[column]]) for column in columns],
        axis=-1,
    )


def _lag_mae(gt: np.ndarray, pred: np.ndarray, lag: int) -> float:
    if lag < 0:
        left, right = gt[-lag:], pred[:lag]
    elif lag > 0:
        left, right = gt[:-lag], pred[lag:]
    else:
        left, right = gt, pred
    return float(np.mean(np.abs(left - right)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-horizon", type=int, default=13)
    parser.add_argument("--denoising-steps", type=int, default=4)
    args = parser.parse_args()

    embodiment = EmbodimentTag.resolve("NEW_EMBODIMENT")
    policy = Gr00tPolicy(embodiment_tag=embodiment, model_path=str(args.checkpoint), device="cuda")
    policy.model.action_head.num_inference_timesteps = args.denoising_steps
    modality = policy.get_modality_config()
    loader = LeRobotEpisodeLoader(dataset_path=str(args.dataset), modality_configs=modality)
    action_keys = modality["action"].modality_keys
    state_keys = modality["state"].modality_keys
    observation_modality = dict(modality)
    observation_modality.pop("action")

    episode_results = []
    all_gt = []
    all_pred = []
    for episode_id in range(len(loader)):
        traj = loader[episode_id]
        predictions = []
        for step in range(0, len(traj), args.execution_horizon):
            point = extract_step_data(traj, step, observation_modality, embodiment)
            obs = {f"state.{key}": value for key, value in point.states.items()}
            obs.update({f"video.{key}": np.asarray(value) for key, value in point.images.items()})
            for language_key in modality["language"].modality_keys:
                obs[language_key] = point.text
            action, _ = policy.get_action(parse_observation_gr00t(obs, modality))
            chunk = np.concatenate(
                [np.asarray(action[key][0]) for key in action_keys], axis=-1
            )
            predictions.append(chunk[: args.execution_horizon])

        gt = _concat_columns(traj, [f"action.{key}" for key in action_keys])
        pred = np.concatenate(predictions, axis=0)[: len(gt)]
        arm_dims = [*range(6), *range(7, 13)]
        lag_errors = {str(lag): _lag_mae(gt[:, arm_dims], pred[:, arm_dims], lag) for lag in range(-5, 6)}
        best_lag = min(lag_errors, key=lag_errors.get)
        episode_results.append(
            {
                "episode_id": episode_id,
                "shape": list(pred.shape),
                "finite": bool(np.isfinite(pred).all()),
                "mae": float(np.mean(np.abs(gt - pred))),
                "mse": float(np.mean((gt - pred) ** 2)),
                "prediction_std": float(np.std(pred)),
                "mean_dimension_std": float(np.mean(np.std(pred, axis=0))),
                "best_arm_lag": int(best_lag),
                "arm_lag_mae": lag_errors,
            }
        )
        all_gt.append(gt)
        all_pred.append(pred)

    gt = np.concatenate(all_gt)
    pred = np.concatenate(all_pred)
    gripper_dims = [6, 13]
    result = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "execution_horizon": args.execution_horizon,
        "denoising_steps": args.denoising_steps,
        "episodes": episode_results,
        "all_shapes_match": all(item["shape"] == [74, 14] for item in episode_results),
        "all_finite": bool(np.isfinite(pred).all()),
        "predictions_nonconstant": bool(np.any(np.std(pred, axis=0) > 1e-6)),
        "prediction_std_by_dimension": np.std(pred, axis=0).tolist(),
        "aggregate_mae": float(np.mean(np.abs(gt - pred))),
        "aggregate_mse": float(np.mean((gt - pred) ** 2)),
        "ground_truth_gripper_max_abs": float(np.max(np.abs(gt[:, gripper_dims]))),
        "prediction_gripper_mae": float(np.mean(np.abs(pred[:, gripper_dims] - gt[:, gripper_dims]))),
        "prediction_gripper_max_abs": float(np.max(np.abs(pred[:, gripper_dims]))),
        "episode_best_arm_lags": [item["best_arm_lag"] for item in episode_results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
