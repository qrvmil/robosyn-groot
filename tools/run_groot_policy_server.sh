#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT=${WORK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CHECKPOINT=${1:-$WORK_ROOT/runs/click_bell_sim_baseline_2k/checkpoints/click_bell_sim_baseline_2k/checkpoint-2000}
PORT=${2:-5555}
GROOT_ROOT=$WORK_ROOT/repos/Isaac-GR00T
PYTHON_BIN=$GROOT_ROOT/.venv/bin/python
CONFIG_PATH=$WORK_ROOT/configs/robosyn_cobotmagic_config.py
OUTPUT_DIR=$WORK_ROOT/runs/click_bell_sim_baseline_2k/evaluation/robosyn_closed_loop

if [[ ! -d "$CHECKPOINT" ]]; then
  echo "Error: checkpoint directory does not exist: $CHECKPOINT" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Error: GR00T Python does not exist: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Error: modality config does not exist: $CONFIG_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
cd "$GROOT_ROOT"
exec "$PYTHON_BIN" -c '
import importlib.util
import runpy
import sys

config_path = sys.argv.pop(1)
spec = importlib.util.spec_from_file_location("robosyn_cobotmagic_config", config_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.argv[0] = "gr00t/eval/run_gr00t_server.py"
runpy.run_path("gr00t/eval/run_gr00t_server.py", run_name="__main__")
' "$CONFIG_PATH" \
  --model-path "$CHECKPOINT" \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda \
  --host 127.0.0.1 \
  --port "$PORT" \
  2>&1 | tee "$OUTPUT_DIR/server.log"
