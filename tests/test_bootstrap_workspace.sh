#!/usr/bin/env bash
set -euo pipefail

fixture="$(mktemp -d)"
trap 'rm -rf "$fixture"' EXIT

bash tools/bootstrap_workspace.sh "$fixture/work"
test -d "$fixture/work/data/raw"
test -d "$fixture/work/data/prepared"
test -d "$fixture/work/data/tiny"
test -d "$fixture/work/runs"
test -f "$fixture/work/env.sh"
! grep -Eq 'HF_TOKEN|WANDB_API_KEY' "$fixture/work/env.sh"

# A second run must remain safe and produce the same layout.
bash tools/bootstrap_workspace.sh "$fixture/work"
test -f "$fixture/work/env.sh"
