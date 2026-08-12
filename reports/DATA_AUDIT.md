# Structural and Numerical Data Audit

Dataset: `RoboSynChallenge/cobotmagic_Sim_click_bell` at revision `dd1f0de495cfec3240beaf2714f239168d5b3fae`.

## Reconciliation

- Episodes: 1,000 metadata / 1,000 parquet, all IDs unique.
- Frames: 74,000 metadata / 74,000 parquet.
- Episode length: exactly 74 frames for every episode.
- Timing: 25 Hz metadata; measured median dt `0.0399999619 s`.
- Tasks: one task, `Click the bell`.

## Numerical integrity

- `observation.qpos`: shape 14; NaN 0; Inf 0; range `[-2.3997223, 2.4936187]`.
- `observation.qvel`: shape 14; NaN 0; Inf 0; range `[-7.5136952, 10.2674074]`.
- `observation.qf`: shape 14; NaN 0; Inf 0; constant zero in all 74,000 frames. It must not be selected as useful state without new evidence.
- `action`: shape 14; NaN 0; Inf 0; range `[-2.3999429, 2.4942770]`.

These aggregate values do not establish units, slice semantics, gripper convention, or absolute/delta representation. Those remain gated on source and visual evidence.

## Video integrity

- Files: 3,000 total, exactly 1,000 for each of `cam_high.color`, `cam_left_wrist.color`, and `cam_right_wrist.color`.
- All files: AV1, 640×480, 25 fps, 74 frames, 2.96 seconds.
- `ffprobe` failures: 0 / 3,000.

Machine-readable evidence: `data/manifests/cobotmagic_Sim_click_bell.audit.json`.
