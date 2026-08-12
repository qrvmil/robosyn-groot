#!/usr/bin/env bash
set -euo pipefail

source /workspace/challenge/robosyn-groot/env.sh
cd "$WORK_ROOT/repos/Isaac-GR00T"

exec env CUDA_VISIBLE_DEVICES=0 .venv/bin/python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$WORK_ROOT/data/tiny/cobotmagic_click_bell_4ep_v1" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$WORK_ROOT/configs/robosyn_cobotmagic_config.py" \
  --num-gpus 1 \
  --output-dir "$WORK_ROOT/runs/tiny_click_bell_v1/checkpoints" \
  --experiment-name tiny_click_bell_v1 \
  --no-use-wandb \
  --no-tune-llm \
  --no-tune-visual \
  --tune-projector \
  --tune-diffusion-model \
  --state-dropout-prob 0.0 \
  --use-percentiles \
  --global-batch-size 4 \
  --gradient-accumulation-steps 1 \
  --dataloader-num-workers 2 \
  --learning-rate 1e-4 \
  --weight-decay 1e-5 \
  --warmup-ratio 0.05 \
  --episode-sampling-rate 1.0 \
  --shard-size 248 \
  --num-shards-per-epoch 20 \
  --save-steps 100 \
  --save-total-limit 5 \
  --save-only-model \
  --max-steps 500
