# Modality Configuration Review

Verdict: **PASS for baseline configuration**. The mapping below is supported by the pinned RoboSyn source, the complete 1,000-episode audit, full-dataset action/state statistics, and visual inspection of five spread episodes.

## Dataset identity

- Dataset: `RoboSynChallenge/cobotmagic_Sim_click_bell` at `dd1f0de495cfec3240beaf2714f239168d5b3fae`.
- RoboSyn source used only as semantic evidence: `93f95f898b76548cc259d20e2b90860a6f79120d`.
- 1,000 episodes, 74,000 frames, 25 FPS; every episode is 74 frames (2.96 s).
- All 3,000 AV1 videos decode successfully with system FFmpeg and reconcile with parquet frame counts.

## State and action decision

The demonstrated state is `observation.qpos[14]`; the demonstrated action is `action[14]`.

| Group | Slice `[start:end)` | Stored value | GR00T action representation |
|---|---:|---|---|
| `left_arm` | `0:6` | absolute joint position | `RELATIVE`, referenced to `state.left_arm` |
| `left_gripper` | `6:7` | normalized position | `ABSOLUTE` |
| `right_arm` | `7:13` | absolute joint position | `RELATIVE`, referenced to `state.right_arm` |
| `right_gripper` | `13:14` | normalized position | `ABSOLUTE` |

Primary evidence:

- `docs/tutorials/policy/your_own_policy.md:81-83` defines `[left_arm + left_gripper + right_arm + right_gripper]` and default absolute joint control.
- `configs/click_bell/action_config.json` declares 6 + 1 + 6 + 1 dimensions.
- `tasks/click_bell/click_bell.py:98-129` assembles qpos targets into active-joint order.
- `tasks/click_bell/action_bank.py:131-157` defines gripper open as 1 and closed as 0.
- `replay.py:64-95` explicitly denormalizes CobotMagic gripper observations from `[0,1]` to physical qpos.

Across all 74,000 frames, `|action - qpos(t)|` has MAE `0.0007517`, versus `0.0045815` for `|action - qpos(t+1)|`. This supports same-step absolute target alignment and contradicts interpreting the stored vector as a delta. Both gripper action coordinates are always zero for this click-bell task. They remain in the embodiment schema, but this dataset alone cannot teach open/close transitions.

`observation.qvel`, `observation.qf`, and the two 4x4 EEF pose matrices are excluded from the baseline. In particular, every `qf` dimension is zero across the dataset, and the source control representation is joint qpos rather than EEF pose.

## Video and language decision

| GR00T key | Original LeRobot key | Physical view |
|---|---|---|
| `front` | `cam_high.color` | base/overhead |
| `left_wrist` | `cam_left_wrist.color` | left wrist |
| `right_wrist` | `cam_right_wrist.color` | right wrist |

The source mapping is explicit in `policy/pi0/deploy_policy.py:39-55`. Visual inspection used episodes `0`, `250`, `500`, `749`, and `999`, with start/middle/end frames for each. The overhead view shows the workspace and both arms; the right wrist view follows the bell/contact; the left wrist view remains physically distinct. Motion is synchronized across views and no left/right or temporal swap was observed.

Language uses GR00T key `annotation.human.task_description`, mapped by `meta/modality.json` to the existing `task_index`. `meta/tasks.jsonl` resolves task index 0 to `Click the bell`, matching the official GR00T demo pattern; parquet rewriting is unnecessary.

## Temporal horizon

Video/state use `[0]`. Action uses `range(0, 13)`: 13 frames at 25 FPS is 0.52 seconds, the closest whole-frame horizon to the approved 0.5-second target. Statistics must be generated with this exact config and regenerated if the horizon changes.

## Evidence artifacts

- `configs/robosyn_cobotmagic_semantics.json`
- `reports/action_semantics_metrics.json`
- `data/manifests/cobotmagic_Sim_click_bell.audit.json`
- `reports/visual_audit/episode_{0,250,500,749,999}.mp4`
- `reports/visual_audit/stills/`
