#!/usr/bin/env python3
"""Load a GR00T N1.7 checkpoint in a fresh process and record basic evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_model_weights(root: Path) -> str:
    """Hash only immutable model shards, independent of resume/evidence files."""
    digest = hashlib.sha256()
    paths = sorted(root.glob("model*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"no model safetensor shards under {root}")
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import torch
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7

    model = Gr00tN1d7.from_pretrained(
        str(args.checkpoint),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "sha256": sha256_model_weights(args.checkpoint),
        "sha256_scope": "model_safetensors",
        "parameter_count": parameter_count,
        "dtype": "bfloat16",
        "reload_status": "pass",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
