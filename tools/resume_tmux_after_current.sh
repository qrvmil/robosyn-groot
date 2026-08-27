#!/usr/bin/env bash
set -euo pipefail

session="robosyn-10task"
work_root="/workspace/challenge/robosyn-groot"

while tmux has-session -t "$session" 2>/dev/null; do
  sleep 30
done

tmux new-session -d -s "$session" \
  "cd '$work_root' && exec robosyn-train-all >> reports/orchestrator.log 2>&1"
