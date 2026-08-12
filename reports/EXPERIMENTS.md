# Experiments

| Run | Data | Steps | Scope | Result |
|---|---|---:|---|---|
| `so100_smoke` | Official SO100 smoke data | 10 | Official smoke configuration | Passed training and checkpoint reload |
| `tiny_click_bell_v1` | Episodes 0, 250, 500, 749 | 500 | Projector + diffusion | Passed; loss fell 23.3% comparing first/last ten windows; open-loop passed |
| `click_bell_sim_baseline_2k` | Full prepared Click Bell dataset | 2,000 | Projector + diffusion | Prepared and preflight checked; intentionally unlaunched |

## Tiny open-loop result

All four trajectories produced finite, nonconstant `(74, 14)` denormalized predictions. Aggregate MAE/MSE were `0.01669`/`0.00238`; best arm lag was zero for every episode. Both source gripper targets and predictions were exactly zero, consistent with the dataset.

## Full baseline profile

- Immutable base snapshot: `nvidia/GR00T-N1.7-3B@2fc962b973bccdd5d8ce4f67cc63b264d6886495`
- Dataset snapshot: `RoboSynChallenge/cobotmagic_Sim_click_bell@dd1f0de495cfec3240beaf2714f239168d5b3fae`
- GR00T revision: `376ba890cff8c9de64d71d982772a9c36185fdd7`
- One A100, global batch 32, accumulation 1, four workers
- LR `1e-4`, weight decay `1e-5`, warmup ratio `0.05`
- Frozen LLM/visual; trainable projector/diffusion
- Save every 250 steps; retain six full resumable checkpoints
