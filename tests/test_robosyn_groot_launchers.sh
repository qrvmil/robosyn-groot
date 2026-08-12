#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

SERVER_OUTPUT="$TMP_ROOT/server.out"
if WORK_ROOT="$WORK_ROOT" bash "$WORK_ROOT/tools/run_groot_policy_server.sh" "$TMP_ROOT/missing-checkpoint" >"$SERVER_OUTPUT" 2>&1; then
  fail "server launcher accepted a missing checkpoint"
fi
grep -F "checkpoint directory does not exist" "$SERVER_OUTPUT" >/dev/null || fail "server launcher missing checkpoint diagnostic"

BOOT_OUTPUT="$TMP_ROOT/bootstrap.out"
if WORK_ROOT="$TMP_ROOT/work" EMBODICHAIN_REPO="$TMP_ROOT/missing-source" bash "$WORK_ROOT/tools/bootstrap_robosyn_eval.sh" --no-clone >"$BOOT_OUTPUT" 2>&1; then
  fail "bootstrap accepted missing EmbodiChain source with --no-clone"
fi
grep -F "EmbodiChain checkout does not exist" "$BOOT_OUTPUT" >/dev/null || fail "bootstrap missing source diagnostic"

FAKE_PYTHON="$TMP_ROOT/fake-python"
ARGV_LOG="$TMP_ROOT/argv.log"
cat >"$FAKE_PYTHON" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$@" >"$ARGV_LOG"
EOF
chmod +x "$FAKE_PYTHON"

ARGV_LOG="$ARGV_LOG" \
PYTHON_BIN="$FAKE_PYTHON" \
EMBODICHAIN_ROOT="$TMP_ROOT/EmbodiChain" \
bash "$WORK_ROOT/repos/RoboSynChallenge/policy/groot/eval.sh" \
  click_bell clear checkpoint-2000 0 \
  --max_episodes 7 --headless True --filter_dataset_saving True \
  --server_host 127.0.0.1 --server_port 6001

python3 - "$ARGV_LOG" <<'PY'
from pathlib import Path
import sys

args = Path(sys.argv[1]).read_text().splitlines()
required_pairs = {
    "--task_name": "click_bell",
    "--setting": "clear",
    "--model_name": "checkpoint-2000",
    "--max_episodes": "7",
    "--headless": "True",
    "--filter_dataset_saving": "True",
    "--server_host": "127.0.0.1",
    "--server_port": "6001",
}
assert args[0].endswith("scripts/eval_policy.py"), args
for key, value in required_pairs.items():
    index = args.index(key)
    assert args[index + 1] == value, (key, args[index + 1])
PY

echo "launcher tests passed"
