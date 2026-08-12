#!/usr/bin/env bash
set -euo pipefail

source /workspace/challenge/robosyn-groot/env.sh
cd "$WORK_ROOT/repos/Isaac-GR00T"

exec env CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  "$WORK_ROOT/runs/tiny_click_bell_v1/evaluation/analyze_open_loop.py" \
  --dataset "$WORK_ROOT/data/tiny/cobotmagic_click_bell_4ep_v1" \
  --checkpoint "$WORK_ROOT/runs/tiny_click_bell_v1/checkpoints/tiny_click_bell_v1/checkpoint-500" \
  --output "$WORK_ROOT/runs/tiny_click_bell_v1/evaluation/metrics.json" \
  --execution-horizon 13 \
  --denoising-steps 4
