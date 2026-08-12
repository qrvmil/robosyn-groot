# RoboSyn GR00T Closed-Loop Evaluation Design

## Goal

Run the trained GR00T `checkpoint-2000` through RoboSynChallenge's original
`scripts/eval_policy.py` evaluator for `click_bell`, producing the evaluator's
native episode success rate, per-episode videos, and reproducible logs.

## Scope

The integration targets this workspace and these pinned inputs:

- RoboSynChallenge commit `93f95f898b76548cc259d20e2b90860a6f79120d`.
- Isaac-GR00T checkout already present under `repos/Isaac-GR00T`.
- Model checkpoint
  `runs/click_bell_sim_baseline_2k/checkpoints/click_bell_sim_baseline_2k/checkpoint-2000`.
- RoboSyn task `click_bell`, setting `clear`, one smoke episode followed by ten
  final episodes.

Random-setting evaluation, simulator training-data generation, and changes to
the trained checkpoint are outside this first integration.

## Architecture

The simulator and model run in separate processes because RoboSyn/EmbodiChain
and GR00T require incompatible Python and native dependency stacks.

1. A Python 3.12 GR00T process loads the checkpoint and serves the existing
   ZeroMQ policy protocol on localhost.
2. A Python 3.11 RoboSyn process runs the unmodified original
   `scripts/eval_policy.py` loop.
3. A new `policy/groot` package implements RoboSyn's policy adapter contract and
   uses a lightweight client for the existing GR00T ZeroMQ protocol. The client
   depends only on packages safe to install in the simulator environment.
4. The adapter converts RoboSyn observations into the flat observation mapping
   expected by `Gr00tPolicy`, requests an action chunk, converts the response to
   the simulator's 14-dimensional absolute joint-target order, and steps the
   environment for at most 13 actions before replanning.

The original evaluator remains authoritative for episode reset, seed choice,
timeout, success checks, progress reporting, video capture, and aggregate
success rate.

## Observation Contract

Each inference request carries:

- `state.left_arm`: `robot/qpos[0:6]`;
- `state.left_gripper`: `robot/qpos[6:7]`;
- `state.right_arm`: `robot/qpos[7:13]`;
- `state.right_gripper`: `robot/qpos[13:14]`;
- `video.front`: `sensor/cam_high/color`;
- `video.left_wrist`: `sensor/cam_left_wrist/color`;
- `video.right_wrist`: `sensor/cam_right_wrist/color`;
- `annotation.human.task_description`: the RoboSyn instruction, expected to be
  `Click the bell`.

Camera batches and state batches must preserve the single-environment batch
dimension required by the GR00T policy protocol. Inputs may originate as Torch
tensors, NumPy arrays, or array-like values; conversion to NumPy occurs before
serialization.

## Action Contract

The GR00T checkpoint predicts the four modalities in this order:

1. six absolute left-arm joint targets returned after GR00T internally reverses
   the relative training transform;
2. one absolute normalized left-gripper target;
3. six absolute right-arm joint targets returned after GR00T internally reverses
   the relative training transform;
4. one absolute normalized right-gripper target.

The adapter concatenates these into the simulator's demonstrated order:

```text
[left_arm(6), left_eef(1), right_arm(6), right_eef(1)]
```

The adapter must reject missing keys, inconsistent chunk lengths, non-finite
values, or action dimensions other than 14. It executes at most 13 actions per
inference request and stops a chunk immediately if the environment truncates.
Relative-to-absolute conversion must not be applied in the adapter because the
GR00T policy output is already denormalized into absolute targets.

## Policy Adapter Files

`repos/RoboSynChallenge/policy/groot/` will contain:

- `__init__.py`: exports the RoboSyn adapter functions.
- `client.py`: lightweight, timeout-aware ZeroMQ client compatible with
  GR00T's `MsgSerializer` wire format.
- `deploy_policy.py`: observation/action conversion plus RoboSyn's
  `get_model`, `eval`, and `reset_model` functions.
- `deploy_policy.yml`: fixed evaluator defaults and localhost server settings.
- `eval.sh`: reproducible entry point accepting task, setting, checkpoint label,
  GPU id, and normal RoboSyn evaluator overrides.

The checkpoint itself is loaded only by the GR00T server. The RoboSyn adapter
receives host/port settings and a checkpoint label used for artifact paths.

## Environment and Launching

EmbodiChain will be cloned under `repos/EmbodiChain` at a recorded commit. A
dedicated Python 3.11 environment will be created at `.venvs/robosyn`; it will
contain EmbodiChain, `embodichain_tasks`, RoboSynChallenge, and the lightweight
client dependencies. The existing GR00T Python 3.12 environment will not be
modified for simulator dependencies.

Two launch scripts will make the process reproducible:

- a GR00T server command using `checkpoint-2000`, `NEW_EMBODIMENT`, CUDA, and a
  localhost port;
- the RoboSyn `policy/groot/eval.sh` command invoking the original evaluator.

Both commands write logs below
`runs/click_bell_sim_baseline_2k/evaluation/robosyn_closed_loop/`. The evaluation
uses headless mode, disables dataset saving, enables video capture, and records
the pinned repository/environment state.

## Success Semantics

No proxy metric replaces the simulator result. RoboSyn's `ClickBellEnv` marks
an episode successful once the button's prismatic joint reaches a press depth
of at least `0.0048`. `scripts/eval_policy.py` calls `is_task_success()` during
the rollout and prints the aggregate as:

```text
Evaluation Results Summary: successes/episodes (percentage%)
```

The final reported result is the ten-episode `clear` success rate. The one-episode
run is only an integration smoke test and is reported separately.

## Failure Handling

- The launcher fails before evaluation if the checkpoint, simulator checkout,
  config files, or Python environments are missing.
- The client uses finite send/receive timeouts and recreates its REQ socket after
  a timeout so a failed request cannot poison subsequent attempts.
- The adapter pings the server during model creation and reports a direct error
  when the server is unavailable.
- Observation/action contract violations include the offending key or shape in
  the error.
- Server and simulator processes write separate logs; neither process silently
  falls back to replay or expert actions.
- A failed smoke episode caused by a runtime/integration exception blocks the
  ten-episode run. A valid rollout with zero task successes does not block the
  final run; zero is a legitimate policy result.

## Testing and Acceptance

Unit tests run without GPU or simulator native libraries and cover:

- Torch/NumPy observation conversion and exact camera/state/language keys;
- exact 14D action concatenation and 13-step chunk limit;
- early stop on truncation;
- missing, malformed, and non-finite action responses;
- client request/response serialization and server-unavailable errors;
- config and shell entry-point wiring to the original evaluator.

Integration acceptance requires:

1. all new unit tests pass;
2. RoboSyn/EmbodiChain import smoke passes in Python 3.11;
3. the GR00T server loads `checkpoint-2000` and answers `ping`;
4. one headless `click_bell clear` episode completes through
   `scripts/eval_policy.py` without an integration exception;
5. ten headless `clear` episodes complete and print the original RoboSyn success
   summary;
6. server log, evaluator log, environment manifest, and available episode videos
   are retained under the run's closed-loop evaluation directory.

