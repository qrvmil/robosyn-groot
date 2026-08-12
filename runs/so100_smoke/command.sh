#!/usr/bin/env bash
set -euo pipefail

source /workspace/challenge/robosyn-groot/env.sh
cd "$WORK_ROOT/repos/Isaac-GR00T"

exec env CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path ./demo_data/cube_to_bowl_5 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/SO100/so100_config.py \
  --num-gpus 1 \
  --output-dir "$WORK_ROOT/runs/so100_smoke/checkpoints" \
  --save-total-limit 2 \
  --save-steps 10 \
  --max-steps 10 \
  --global-batch-size 4 \
  --dataloader-num-workers 2
