#!/usr/bin/env bash
set -euo pipefail

work_root="${1:?usage: bootstrap_workspace.sh WORK_ROOT}"

mkdir -p "$work_root"/{repos,data/{raw,prepared,tiny,manifests},configs,tools,tests,runs,reports,cache/{huggingface,torch,uv},backups,vendor/runbook}

cat > "$work_root/env.sh" <<ENV
export WORK_ROOT="$work_root"
export HF_HOME="$work_root/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$work_root/cache/huggingface/hub"
export TORCH_HOME="$work_root/cache/torch"
export UV_CACHE_DIR="$work_root/cache/uv"
export WANDB_DIR="$work_root/runs/wandb"
export CUDA_HOME="/usr/local/cuda"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
ENV

chmod 0644 "$work_root/env.sh"
