# 09. Evaluation: open-loop и closed-loop RoboSyn

## 1. Open-loop — обязательный минимум

Для каждого сохранённого checkpoint оценить:

- несколько training episodes;
- фиксированный held-out episode set;
- при возможности отдельные sim random/real episodes.

Пример:

```bash
source "$WORK_ROOT/env.sh"
cd "$WORK_ROOT/repos/Isaac-GR00T"

DATA="$WORK_ROOT/data/prepared/cobotmagic_Sim_click_bell__groot_v1"
CFG="$WORK_ROOT/configs/robosyn_cobotmagic_config.py"
CKPT="$WORK_ROOT/runs/click_bell_sim_baseline_2k/checkpoints/checkpoint-2000"
OUT="$WORK_ROOT/runs/click_bell_sim_baseline_2k/evaluation/checkpoint-2000"
mkdir -p "$OUT"

uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path "$DATA" \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path "$CKPT" \
  --traj-ids 10 11 12 13 \
  --execution-horizon 16 \
  --steps 400 \
  --save-plot-path "$OUT" \
  --modality-keys left_arm left_gripper right_arm right_gripper \
  2>&1 | tee "$OUT/open_loop.log"
```

Exact flags сверить с `--help`; если custom modality config нужен open-loop script, передать его поддерживаемым текущим способом либо зарегистрировать config в process.

## 2. Offline metrics

Считать после denormalization:

- MAE/MSE overall;
- MAE per action dimension/slice;
- error per chunk position;
- gripper accuracy/error отдельно;
- mean/std/range predictions;
- clipping rate;
- temporal lag/correlation;
- flat-action rate.

Не сравнивать normalized MSE между datasets с разными stats как универсальную метрику.

## 3. Diagnostic tests

- **Shuffled instruction test:** результат должен ухудшаться, если задача действительно language-conditioned.
- **Camera ablation:** убрать/замаскировать одну camera только как diagnostic; проверить зависимость policy.
- **State ablation:** убедиться, что state используется.
- **Chunk execution sweep:** execution horizon меньше model horizon может повысить reactivity.

## 4. Closed-loop в RoboSyn

После рабочего open-loop создать отдельную интеграцию, не встраивая GR00T dependencies в RoboSyn process.

Рекомендуемая архитектура:

```text
RoboSyn/EmbodiChain process (Python 3.11)
        │ observations/actions over IPC or socket
        ▼
GR00T policy server (Python 3.12)
```

Это лучше общего environment из-за конфликтующих native/Python dependencies.

Policy adapter обязан:

1. взять `cam_high`, wrist cameras, qpos/state и instruction;
2. применить **тот же preprocessing**, что использовался в training;
3. получить GR00T action chunk;
4. denormalize в исходные units;
5. преобразовать relative arm actions обратно в абсолютные targets ровно один раз;
6. выполнить только `execution_horizon` первых actions;
7. соблюдать control frequency и latency;
8. записать video, actions, timing, success/failure reason.

## 5. Порядок rollout

```text
1 episode click_bell clear
→ 10 episodes clear с фиксированными seeds
→ 20+ episodes clear/new seeds
→ random setting
→ sim/real transfer, если доступен robot setup
```

Для длинных задач считать stage success:

```text
reach → grasp/contact → manipulate → release/finish
```

## 6. Failure taxonomy

Для каждого episode записывать одну основную и при необходимости вторичную причину:

```text
perception
instruction grounding
reach/planning
gripper
contact/dynamics
chunk/timing
latency
oscillation
clipping
wrong left/right
premature termination
```

## 7. Выбор checkpoint

Отобрать 3–5 checkpoints по rollout success и устойчивости на новых seeds. Затем повторить evaluation. Не использовать только последний checkpoint или минимальный train loss.
