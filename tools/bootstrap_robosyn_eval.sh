#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT=${WORK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
EMBODICHAIN_REPO=${EMBODICHAIN_REPO:-$WORK_ROOT/repos/EmbodiChain}
ROBOSYN_REPO=${ROBOSYN_REPO:-$WORK_ROOT/repos/RoboSynChallenge}
VENV_DIR=${ROBOSYN_VENV:-$WORK_ROOT/.venvs/robosyn}
NO_CLONE=false
if [[ ${1:-} == "--no-clone" ]]; then
  NO_CLONE=true
fi

if [[ ! -d "$EMBODICHAIN_REPO" ]]; then
  if [[ "$NO_CLONE" == true ]]; then
    echo "Error: EmbodiChain checkout does not exist: $EMBODICHAIN_REPO" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$EMBODICHAIN_REPO")"
  git clone https://github.com/DexForce/EmbodiChain.git "$EMBODICHAIN_REPO"
fi
if [[ ! -d "$ROBOSYN_REPO" ]]; then
  echo "Error: RoboSynChallenge checkout does not exist: $ROBOSYN_REPO" >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is required" >&2
  exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  uv venv --python 3.11 "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1 || uv pip install --python "$VENV_DIR/bin/python" pip
uv pip install --python "$VENV_DIR/bin/python" -e "$EMBODICHAIN_REPO" \
  --extra-index-url http://pyp.open3dv.site:2345/simple/ \
  --trusted-host pyp.open3dv.site
if [[ -d "$EMBODICHAIN_REPO/embodichain_tasks" ]]; then
  uv pip install --python "$VENV_DIR/bin/python" -e "$EMBODICHAIN_REPO/embodichain_tasks"
fi
uv pip install --python "$VENV_DIR/bin/python" -e "$ROBOSYN_REPO" 'numpy<2' pyzmq msgpack msgpack-numpy pytest

PYTHONPATH="$EMBODICHAIN_REPO:$ROBOSYN_REPO" "$VENV_DIR/bin/python" - <<'PY'
import dexsim
import embodichain
import embodichain_tasks
import msgpack_numpy
import robosynchallenge
import zmq
print("RoboSyn evaluation imports OK")
PY

