#!/usr/bin/env bash
set -euo pipefail

source /workspace/challenge/robosyn-groot/env.sh
cd "$WORK_ROOT/repos/Isaac-GR00T"

exec env \
  CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python gr00t/experiment/launch_finetune.py \
  --base-model-path /workspace/challenge/robosyn-groot/cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495 \
  --dataset-path "$WORK_ROOT/data/prepared/cobotmagic_Sim_click_bell__groot_v1" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$WORK_ROOT/configs/robosyn_cobotmagic_config.py" \
  --num-gpus 1 \
  --output-dir "$WORK_ROOT/runs/click_bell_sim_baseline_2k/checkpoints" \
  --experiment-name click_bell_sim_baseline_2k \
  --no-use-wandb \
  --no-tune-llm \
  --no-tune-visual \
  --tune-projector \
  --tune-diffusion-model \
  --state-dropout-prob 0.0 \
  --use-percentiles \
  --global-batch-size 32 \
  --gradient-accumulation-steps 1 \
  --dataloader-num-workers 4 \
  --learning-rate 1e-4 \
  --weight-decay 1e-5 \
  --warmup-ratio 0.05 \
  --episode-sampling-rate 1.0 \
  --shard-size 1024 \
  --num-shards-per-epoch 100 \
  --save-steps 250 \
  --save-total-limit 6 \
  --max-steps 2000
