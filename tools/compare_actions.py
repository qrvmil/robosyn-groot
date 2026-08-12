#!/usr/bin/env python3
"""Evaluate GR00T action normalization and relative-action round trips."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


def normalize(values: np.ndarray, stats: Mapping[str, np.ndarray]) -> np.ndarray:
    std = np.maximum(np.asarray(stats["std"]), 1e-8)
    return (np.asarray(values) - np.asarray(stats["mean"])) / std


def denormalize(values: np.ndarray, stats: Mapping[str, np.ndarray]) -> np.ndarray:
    std = np.maximum(np.asarray(stats["std"]), 1e-8)
    return np.asarray(values) * std + np.asarray(stats["mean"])


def to_relative(target: np.ndarray, state: np.ndarray) -> np.ndarray:
    return np.asarray(target) - np.asarray(state)[-1]


def from_relative(delta: np.ndarray, state: np.ndarray) -> np.ndarray:
    return np.asarray(delta) + np.asarray(state)[-1]


def _load_config(path: Path) -> dict[str, object]:
    spec = importlib.util.spec_from_file_location("robosyn_roundtrip_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.robosyn_cobotmagic_config


def evaluate_roundtrip(
    dataset: Path, config_path: Path, samples: int, seed: int
) -> dict[str, object]:
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.state_action.state_action_processor import StateActionProcessor

    if samples < 1:
        raise ValueError("samples must be positive")
    config = _load_config(Path(config_path))
    tag = EmbodimentTag.NEW_EMBODIMENT
    loader = LeRobotEpisodeLoader(Path(dataset), config)
    statistics = loader.get_dataset_statistics()
    processor_args = {
        "modality_configs": {tag.value: config},
        "statistics": {tag.value: statistics},
        "use_percentiles": True,
        "use_relative_action": True,
    }
    exact_processor = StateActionProcessor(**processor_args, clip_outliers=False)
    production_processor = StateActionProcessor(**processor_args, clip_outliers=True)

    rng = np.random.default_rng(seed)
    episode_positions = np.sort(
        rng.choice(len(loader), size=min(8, len(loader)), replace=False)
    ).tolist()
    cached = {position: loader[position] for position in episode_positions}
    candidates = [
        (position, step)
        for position, frame in cached.items()
        for step in range(len(frame) - len(config["action"].delta_indices) + 1)
    ]
    chosen = rng.choice(len(candidates), size=samples, replace=len(candidates) < samples)

    keys = config["action"].modality_keys
    exact_errors = {key: [] for key in keys}
    clipped_errors = {key: [] for key in keys}
    clipped_counts = {key: 0 for key in keys}
    value_counts = {key: 0 for key in keys}
    observed_shapes = {}
    sampled_pairs = []

    for candidate_index in chosen:
        position, step_index = candidates[int(candidate_index)]
        raw = extract_step_data(
            cached[position], step_index, config, EmbodimentTag.NEW_EMBODIMENT
        )
        sampled_pairs.append(
            {
                "episode_position": position,
                "episode_id": int(loader.episodes_metadata[position]["episode_index"]),
                "step": step_index,
            }
        )
        processed_state, exact_action = exact_processor.apply(
            raw.states, raw.actions, tag.value
        )
        _, exact_reconstruction = exact_processor.unapply(
            processed_state, exact_action, tag.value, raw_state=raw.states
        )
        production_state, production_action = production_processor.apply(
            raw.states, raw.actions, tag.value
        )
        _, clipped_reconstruction = production_processor.unapply(
            production_state,
            production_action,
            tag.value,
            raw_state=raw.states,
        )
        for key in keys:
            observed_shapes[key] = list(raw.actions[key].shape)
            exact_errors[key].append(
                float(np.max(np.abs(exact_reconstruction[key] - raw.actions[key])))
            )
            clipped_errors[key].append(
                float(np.max(np.abs(clipped_reconstruction[key] - raw.actions[key])))
            )
            clipped_counts[key] += int(np.count_nonzero(np.abs(exact_action[key]) > 1.0))
            value_counts[key] += int(exact_action[key].size)

    groups = {}
    for key in keys:
        groups[key] = {
            "raw_shape": observed_shapes[key],
            "exact_roundtrip_max_abs_error": max(exact_errors[key]),
            "production_clipped_roundtrip_max_abs_error": max(clipped_errors[key]),
            "q01_q99_clip_count": clipped_counts[key],
            "value_count": value_counts[key],
            "q01_q99_clip_fraction": clipped_counts[key] / value_counts[key],
        }
    return {
        "dataset": str(Path(dataset).resolve()),
        "config": str(Path(config_path).resolve()),
        "seed": seed,
        "samples": samples,
        "episode_positions_loaded": episode_positions,
        "action_horizon": len(config["action"].delta_indices),
        "processor": {
            "use_percentiles": True,
            "use_relative_action": True,
            "production_clip_outliers": True,
        },
        "groups": groups,
        "overall_exact_roundtrip_max_abs_error": max(
            group["exact_roundtrip_max_abs_error"] for group in groups.values()
        ),
        "overall_production_clipped_roundtrip_max_abs_error": max(
            group["production_clipped_roundtrip_max_abs_error"]
            for group in groups.values()
        ),
        "sampled_pairs": sampled_pairs,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = evaluate_roundtrip(
        args.dataset, args.config, samples=args.samples, seed=args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "sampled_pairs"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
