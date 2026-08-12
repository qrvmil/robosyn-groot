# Click Bell Baseline 2K

This launcher is prepared and syntax/preflight checked, but intentionally unlaunched.

## Start

```bash
bash /workspace/challenge/robosyn-groot/runs/click_bell_sim_baseline_2k/command.sh 2>&1 | tee /workspace/challenge/robosyn-groot/runs/click_bell_sim_baseline_2k/stdout.log
```

The command uses the immutable local GR00T model snapshot, the fully audited prepared dataset, and the reviewed modality config. It trains only the projector and diffusion model; the LLM and visual backbone remain frozen.

Expected checkpoints are written below `checkpoints/click_bell_sim_baseline_2k/` every 250 steps, retaining the newest six. These are full Trainer checkpoints so optimizer/scheduler state is available for resumption.

Before starting, rerun:

```bash
repos/Isaac-GR00T/.venv/bin/python tools/verify_readiness.py \
  --work-root /workspace/challenge/robosyn-groot \
  --run-name click_bell_sim_baseline_2k \
  --output reports/READINESS.md
```
