# Current Status

- Updated: 2026-08-12 UTC
- Current gate: **READY**
- Current task/dataset: GR00T N1.7 3B fine-tuning / RoboSyn Click the bell
- Active training process: none

## Evidence

- Official SO100 10-step smoke and checkpoint reload: pass
- Full prepared dataset: 1,000 episodes, 74,000 frames, 3,000 decoded videos
- Official loader/stats and 128-sample action round-trip: pass; max error `5.55e-17`
- Tiny training: 500/500 steps; last-window loss `1.0409`; no NaN/OOM
- Tiny open loop: four episodes, `(74, 14)` each; aggregate MAE `0.01669`; best arm lag `0` for all
- Full-profile startup smoke: full dataset, batch 32, four workers, one optimizer step, loss `1.19684`, exit code 0
- Final machine preflight: all checks pass in `reports/READINESS.md`

## Known data constraint

Both demonstrated gripper action dimensions are zero in all 74,000 source frames. The baseline therefore validates arm/contact motion and learns zero gripper targets; it cannot learn an open/close transition absent from the source data.

## Resource headroom

- GPU: NVIDIA A100 80GB PCIe; CUDA and BF16 available
- Disk at final preflight: 262,802,636,800 bytes free
- Required reserve: 161,061,273,600 bytes

## Next exact command

```bash
bash /workspace/challenge/robosyn-groot/runs/click_bell_sim_baseline_2k/command.sh 2>&1 | tee /workspace/challenge/robosyn-groot/runs/click_bell_sim_baseline_2k/stdout.log
```

The 2,000-step run has not been launched.
