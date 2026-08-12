# RoboSyn GR00T Closed-Loop Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the trained GR00T `checkpoint-2000` to RoboSynChallenge's original `scripts/eval_policy.py` and complete one smoke plus ten `click_bell clear` episodes with native success-rate output.

**Architecture:** Keep GR00T in its existing Python 3.12 environment and run RoboSyn/EmbodiChain in a dedicated Python 3.11 environment. A new RoboSyn `policy/groot` adapter uses GR00T's existing ZeroMQ request/reply protocol, maps the exact three-camera/14D CobotMagic observation and action contracts, and lets the unmodified RoboSyn evaluator own rollout and success measurement.

**Tech Stack:** Python 3.11 and 3.12, NumPy, PyTorch, pyzmq, msgpack/msgpack-numpy, pytest, Bash, GR00T PolicyServer, RoboSynChallenge, EmbodiChain.

## Global Constraints

- Preserve RoboSynChallenge commit `93f95f898b76548cc259d20e2b90860a6f79120d` as the recorded source baseline.
- Do not modify trained checkpoint files or raw datasets.
- Do not install RoboSyn/EmbodiChain dependencies into the GR00T Python 3.12 environment.
- Use `scripts/eval_policy.py` unchanged as the authority for episode success and aggregate success rate.
- Use `click_bell`, setting `clear`, one smoke episode, then ten final episodes.
- Use an execution horizon of 13 and exactly 14 absolute simulator action dimensions ordered `[left_arm, left_eef, right_arm, right_eef]`.
- Treat a valid zero-success rollout as a model result, not an integration failure.
- Store logs and manifests under `runs/click_bell_sim_baseline_2k/evaluation/robosyn_closed_loop/`.
- Preserve existing unrelated changes in `reports/READINESS.md` and `runs/click_bell_sim_baseline_2k/stdout.log`.

---

### Task 1: Observation and Action Adapter Contract

**Files:**
- Create: `repos/RoboSynChallenge/tests/test_groot_policy_adapter.py`
- Create: `repos/RoboSynChallenge/policy/groot/__init__.py`
- Create: `repos/RoboSynChallenge/policy/groot/deploy_policy.py`

**Interfaces:**
- Consumes: RoboSyn observation dictionaries containing `robot/qpos`, three camera color tensors, and `env._current_instruction`.
- Produces: `encode_observation(observation: dict, instruction: str) -> dict[str, numpy.ndarray | list[str]]`, `decode_action_chunk(response: object) -> numpy.ndarray`, `eval(env, model, obs) -> tuple`, and `reset_model(model) -> None`.

- [ ] **Step 1: Write failing observation-mapping tests**

Create literal fixtures with qpos `[0, 1, ..., 13]` and distinct camera arrays. Assert the encoded mapping contains exactly:

```python
{
    "state.left_arm",
    "state.left_gripper",
    "state.right_arm",
    "state.right_gripper",
    "video.front",
    "video.left_wrist",
    "video.right_wrist",
    "annotation.human.task_description",
}
```

Assert the state slices are `[0:6]`, `[6:7]`, `[7:13]`, `[13:14]`, all values are NumPy arrays with a single-environment batch dimension, and the instruction is `Click the bell`.

- [ ] **Step 2: Run the observation tests and verify RED**

Run:

```bash
cd repos/RoboSynChallenge
../Isaac-GR00T/.venv/bin/python -m pytest -q tests/test_groot_policy_adapter.py
```

Expected: collection fails because `policy.groot.deploy_policy` does not exist.

- [ ] **Step 3: Implement minimal observation conversion**

Implement a path lookup helper, tensor-to-NumPy conversion using `detach().cpu().numpy()` when available, explicit qpos length validation, exact camera mapping, and exact instruction mapping. Do not import GR00T or EmbodiChain in this module.

- [ ] **Step 4: Run the observation tests and verify GREEN**

Run the command from Step 2. Expected: observation tests pass.

- [ ] **Step 5: Write failing action and stepping tests**

Use a literal response matching the actual GR00T client return shape:

```python
(
    {
        "left_arm": np.arange(78, dtype=np.float32).reshape(1, 13, 6),
        "left_gripper": np.zeros((1, 13, 1), dtype=np.float32),
        "right_arm": np.ones((1, 13, 6), dtype=np.float32),
        "right_gripper": np.ones((1, 13, 1), dtype=np.float32),
    },
    {},
)
```

Assert decoding returns shape `(13, 14)` and the exact modality order. Add separate tests that missing keys, mismatched horizons, NaN, and wrong final width raise `ValueError`. With a small real fake environment object, assert `eval()` executes 13 actions, passes tensors of shape `(1, 14)`, and stops immediately when truncation becomes true.

- [ ] **Step 6: Run action tests and verify RED**

Run the Task 1 pytest command. Expected: failures identify absent decoder and evaluator behavior.

- [ ] **Step 7: Implement minimal action decoding and evaluator loop**

Implement strict validation, concatenation, per-step tensor conversion onto `env.unwrapped.device`, and early truncation. `get_model()` constructs the lightweight client from `server_host`, `server_port`, and `server_timeout_ms`, pings it, and raises a direct `RuntimeError` when unreachable. `reset_model()` calls the remote reset endpoint.

- [ ] **Step 8: Run Task 1 tests and verify GREEN**

Run the Task 1 pytest command. Expected: all adapter contract tests pass without simulator or GPU imports.

- [ ] **Step 9: Commit Task 1 in the RoboSyn repository**

```bash
git -C repos/RoboSynChallenge add tests/test_groot_policy_adapter.py policy/groot/__init__.py policy/groot/deploy_policy.py
git -C repos/RoboSynChallenge commit -m "feat: add groot robosyn policy adapter"
```

### Task 2: Lightweight GR00T ZeroMQ Client

**Files:**
- Create: `repos/RoboSynChallenge/tests/test_groot_policy_client.py`
- Create: `repos/RoboSynChallenge/policy/groot/client.py`
- Modify: `repos/RoboSynChallenge/policy/groot/deploy_policy.py`

**Interfaces:**
- Consumes: GR00T `PolicyServer` wire requests `{endpoint, data}` serialized through msgpack-numpy.
- Produces: `GrootPolicyClient(host: str, port: int, timeout_ms: int)`, with `ping() -> bool`, `get_action(observation: dict) -> object`, `reset() -> object`, `call_endpoint(...) -> object`, and `close() -> None`.

- [ ] **Step 1: Write a failing real socket protocol test**

Start a local ZeroMQ REP socket on an ephemeral TCP port in a test thread. Decode the client's msgpack request, assert the exact endpoint and complete observation payload, then return a literal action tuple. Assert the client decodes that tuple without converting its array values to lists.

- [ ] **Step 2: Run the client test and verify RED**

```bash
cd repos/RoboSynChallenge
../Isaac-GR00T/.venv/bin/python -m pytest -q tests/test_groot_policy_client.py
```

Expected: collection fails because `policy.groot.client` does not exist.

- [ ] **Step 3: Implement serializer and request/reply client**

Mirror the installed GR00T serializer's msgpack-numpy behavior, configure finite send/receive timeouts, close old REQ sockets before recreation, surface server `{error: ...}` replies as `RuntimeError`, and make `close()` idempotent.

- [ ] **Step 4: Run the protocol test and verify GREEN**

Run the command from Step 2. Expected: the real local socket round-trip passes.

- [ ] **Step 5: Write failing timeout recovery test**

Point the client at an unused ephemeral port, assert the first call raises a timeout, then attach a REP server to the same port and assert a subsequent ping succeeds using the recreated REQ socket.

- [ ] **Step 6: Run timeout test and verify RED**

Run the client tests. Expected: the recovery assertion fails until socket recreation is implemented correctly.

- [ ] **Step 7: Implement timeout recovery and wire adapter imports**

Finish socket recreation and update `deploy_policy.py` to import `GrootPolicyClient` using a package-relative import.

- [ ] **Step 8: Run all adapter/client tests and verify GREEN**

```bash
cd repos/RoboSynChallenge
../Isaac-GR00T/.venv/bin/python -m pytest -q tests/test_groot_policy_adapter.py tests/test_groot_policy_client.py
```

Expected: all tests pass.

- [ ] **Step 9: Commit Task 2 in the RoboSyn repository**

```bash
git -C repos/RoboSynChallenge add tests/test_groot_policy_client.py policy/groot/client.py policy/groot/deploy_policy.py
git -C repos/RoboSynChallenge commit -m "feat: add lightweight groot policy client"
```

### Task 3: RoboSyn Configuration and Launchers

**Files:**
- Create: `repos/RoboSynChallenge/policy/groot/deploy_policy.yml`
- Create: `repos/RoboSynChallenge/policy/groot/eval.sh`
- Create: `tests/test_robosyn_groot_launchers.sh`
- Create: `tools/run_groot_policy_server.sh`
- Create: `tools/bootstrap_robosyn_eval.sh`

**Interfaces:**
- Consumes: `$WORK_ROOT`, checkpoint path, task, setting, GPU id, server host/port, and ordinary `eval_policy.py --overrides` values.
- Produces: validated server/bootstrap/evaluator commands and logs in the closed-loop run directory.

- [ ] **Step 1: Write failing launcher behavior tests**

The shell test invokes each script with `--help` or deliberately missing paths. Assert scripts fail with precise diagnostics for a missing checkpoint or missing EmbodiChain checkout, and assert the evaluator launcher forwards `click_bell`, `clear`, `--max_episodes`, `--headless`, `--filter_dataset_saving`, host, port, and checkpoint label to `scripts/eval_policy.py` using a fake Python executable that records argv.

- [ ] **Step 2: Run launcher test and verify RED**

```bash
bash tests/test_robosyn_groot_launchers.sh
```

Expected: failure because the launchers and config do not exist.

- [ ] **Step 3: Implement evaluator config and launcher**

Set `policy_name: groot`, default `execution_horizon: 13`, `server_host: 127.0.0.1`, `server_port: 5555`, `server_timeout_ms: 120000`, `filter_dataset_saving: true`, `eval_video_log: true`, all three video keys, `eval_reset_sync_steps: 1`, and headless defaults. The shell launcher must use arrays for argument safety and must not evaluate user-provided shell text.

- [ ] **Step 4: Implement server launcher**

Validate the checkpoint and GR00T environment, import `configs/robosyn_cobotmagic_config.py` before server startup so `NEW_EMBODIMENT` resolves to the trained modality contract, bind only to `127.0.0.1`, and tee output to `server.log`.

- [ ] **Step 5: Implement idempotent bootstrap launcher**

Validate Python 3.11 availability, clone EmbodiChain only when absent, record its commit, create `.venvs/robosyn`, install EmbodiChain plus `embodichain_tasks` when present, install RoboSynChallenge plus `numpy<2`, `pyzmq`, `msgpack`, and `msgpack-numpy`, and run import smoke checks without printing secrets.

- [ ] **Step 6: Run launcher tests and verify GREEN**

Run the command from Step 2. Expected: all launcher behavior tests pass.

- [ ] **Step 7: Commit Task 3 in both repositories**

```bash
git -C repos/RoboSynChallenge add policy/groot/deploy_policy.yml policy/groot/eval.sh
git -C repos/RoboSynChallenge commit -m "feat: wire groot into robosyn evaluator"
git add tests/test_robosyn_groot_launchers.sh tools/run_groot_policy_server.sh tools/bootstrap_robosyn_eval.sh
git commit -m "feat: add robosyn closed-loop launchers"
```

### Task 4: Environment Bootstrap and Contract Verification

**Files:**
- Create: `runs/click_bell_sim_baseline_2k/evaluation/robosyn_closed_loop/environment.md`
- Create: `reports/robosyn_eval_python_freeze.txt`

**Interfaces:**
- Consumes: `tools/bootstrap_robosyn_eval.sh` and network access for missing source/packages.
- Produces: importable Python 3.11 simulator runtime and pinned environment evidence.

- [ ] **Step 1: Run bootstrap**

```bash
bash tools/bootstrap_robosyn_eval.sh
```

Expected: `.venvs/robosyn/bin/python` imports `dexsim`, `embodichain`, `embodichain_tasks`, `robosynchallenge`, `zmq`, and `msgpack_numpy`.

- [ ] **Step 2: Record environment evidence**

Record UTC timestamp, GPU summary, Python version, RoboSyn commit, EmbodiChain commit, GR00T commit, checkpoint path, and package freeze without environment secrets.

- [ ] **Step 3: Run all unit and launcher tests in the target environment**

```bash
.venvs/robosyn/bin/python -m pytest -q \
  repos/RoboSynChallenge/tests/test_groot_policy_adapter.py \
  repos/RoboSynChallenge/tests/test_groot_policy_client.py
bash tests/test_robosyn_groot_launchers.sh
```

Expected: all tests pass.

### Task 5: GR00T Server and Original RoboSyn Rollouts

**Files:**
- Create: `runs/click_bell_sim_baseline_2k/evaluation/robosyn_closed_loop/server.log`
- Create: `runs/click_bell_sim_baseline_2k/evaluation/robosyn_closed_loop/smoke_eval.log`
- Create: `runs/click_bell_sim_baseline_2k/evaluation/robosyn_closed_loop/final_eval.log`
- Create: `runs/click_bell_sim_baseline_2k/evaluation/robosyn_closed_loop/result.json`

**Interfaces:**
- Consumes: running GR00T server on localhost port 5555 and installed RoboSyn runtime.
- Produces: native RoboSyn per-episode results, final success rate, videos, logs, and a machine-readable summary extracted from the original evaluator output.

- [ ] **Step 1: Start GR00T server and verify ping**

Run `tools/run_groot_policy_server.sh` in a managed background session. Use the lightweight client in `.venvs/robosyn` to poll `ping` with a bounded deadline. Expected: checkpoint shards load and ping returns true.

- [ ] **Step 2: Run one original-evaluator smoke episode**

```bash
PYTHON_BIN="$WORK_ROOT/.venvs/robosyn/bin/python" \
EMBODICHAIN_ROOT="$WORK_ROOT/repos/EmbodiChain" \
bash repos/RoboSynChallenge/policy/groot/eval.sh \
  click_bell clear checkpoint-2000 0 \
  --max_episodes 1 --headless True --filter_dataset_saving True
```

Expected: `scripts/eval_policy.py` completes one episode and prints `Evaluation Results Summary` without adapter, IPC, shape, simulator, or CUDA exceptions.

- [ ] **Step 3: Diagnose smoke failures test-first**

For any integration defect, add the smallest failing regression test to Task 1, 2, or 3's test file, verify RED, implement the minimal fix, verify GREEN, and rerun the smoke episode. Do not bypass a contract failure with replay or expert actions.

- [ ] **Step 4: Run ten original-evaluator episodes**

Repeat Step 2 with `--max_episodes 10`, teeing output to `final_eval.log`. Expected: ten episodes complete and the evaluator prints `Evaluation Results Summary: N/10 (P%)`.

- [ ] **Step 5: Persist machine-readable result**

Parse only the final summary line into JSON fields `task`, `setting`, `checkpoint`, `episodes`, `successes`, `success_rate`, `smoke_log`, `eval_log`, and `server_log`. Confirm the JSON count matches the original log exactly.

- [ ] **Step 6: Run final verification**

Run the complete unit suite, launcher tests, import smoke, JSON/log consistency check, `git diff --check` in both repositories, and list recorded videos. Expected: all code tests pass and every available artifact path exists.

