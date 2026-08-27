#!/usr/bin/env python3
"""Continuously retain one validated resume checkpoint for an active run."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.cleanup import (
    _CHECKPOINT_NAME,
    _validate_live_resume_checkpoint,
    prune_superseded_live_checkpoints,
)


def _write_record(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _newest_complete_step(run_dir: Path) -> int | None:
    candidates = [
        path
        for path in (run_dir / "checkpoints").glob("*/checkpoint-*")
        if path.is_dir() and _CHECKPOINT_NAME.fullmatch(path.name)
    ]
    if not candidates:
        return None
    newest = max(
        candidates,
        key=lambda path: int(_CHECKPOINT_NAME.fullmatch(path.name).group(1)),
    )
    return _validate_live_resume_checkpoint(newest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Keep only the newest complete checkpoint in a live task run."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--until-step", type=int)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    if args.interval <= 0:
        raise SystemExit("--interval must be positive")
    while True:
        try:
            actions = prune_superseded_live_checkpoints(run_dir)
            step = _newest_complete_step(run_dir)
        except RuntimeError as error:
            if args.once:
                raise
            time.sleep(args.interval)
            continue
        if actions:
            _write_record(
                args.log,
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "status": "pruned",
                    "actions": actions,
                },
            )
        if args.once or (
            args.until_step is not None and step is not None and step >= args.until_step
        ):
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
