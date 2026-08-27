#!/usr/bin/env python3
"""Generate per-task GR00T semantics from immutable RoboSyn metadata."""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Sequence
from pathlib import Path

from tools.task_schema import detect_schema


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _load_single_task(path: Path) -> tuple[int, str]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError("meta/tasks.jsonl contains no tasks")
    if len(records) != 1:
        raise ValueError(f"expected one task instruction, found {len(records)}")
    return int(records[0]["task_index"]), str(records[0]["task"])


def generate_semantics(
    base: dict[str, object],
    raw: Path,
    *,
    task: str,
    repo: str,
    revision: str,
) -> dict[str, object]:
    raw = Path(raw)
    info_path = raw / "meta/info.json"
    tasks_path = raw / "meta/tasks.jsonl"
    info = _load_json(info_path)
    schema = detect_schema(info["features"])
    task_index, instruction = _load_single_task(tasks_path)

    semantics = copy.deepcopy(base)
    semantics["state"]["original_key"] = schema.state_key
    semantics["action"]["original_key"] = schema.action_key
    for logical, original_key in schema.camera_keys.items():
        semantics["video"][logical]["original_key"] = original_key
    semantics["dataset"].update(
        {
            "repo_id": repo,
            "revision": revision,
            "raw_root": str(raw.resolve()),
        }
    )
    semantics["language"].update(
        {"task_index": task_index, "task": instruction}
    )
    semantics["timing"]["fps"] = info["fps"]
    semantics["evidence"] = {
        "generated_from_raw_metadata": str(info_path.resolve()),
        "generated_from_task_metadata": str(tasks_path.resolve()),
        "state_action_names_match": schema.state_names == schema.action_names,
        "joint_names": list(schema.state_names),
        "task": task,
    }
    return semantics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    semantics = generate_semantics(
        _load_json(args.base),
        args.raw,
        task=args.task,
        repo=args.repo,
        revision=args.revision,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(semantics, indent=2) + "\n")
    print(
        json.dumps(
            {
                "task": args.task,
                "state_key": semantics["state"]["original_key"],
                "action_key": semantics["action"]["original_key"],
                "cameras": {
                    key: value["original_key"]
                    for key, value in semantics["video"].items()
                },
                "instruction": semantics["language"]["task"],
                "revision": args.revision,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
