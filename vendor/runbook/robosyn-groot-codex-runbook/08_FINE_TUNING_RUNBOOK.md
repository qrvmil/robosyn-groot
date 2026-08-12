# 08. Основной fine-tuning GR00T N1.7

## 1. Что является baseline

Для первого RoboSyn run использовать максимально близкий к official N1.7 custom-embodiment scope:

```text
base model: nvidia/GR00T-N1.7-3B
trainable: projector + diffusion action model
frozen: LLM + visual backbone
optimizer: AdamW
LR: 1e-4
scheduler: cosine
warmup: 5%
max grad norm: 1.0
precision: BF16
weight decay: 1e-5 (official baseline)
global batch: 32 target on 1× A100 80 GB
```

`cosine`, BF16 и `max_grad_norm=1.0` заданы текущим GR00T training config; перед запуском подтвердить кодом/current `--help`.

Cookbook рекомендует weight decay `0.01`; это не подменяет official baseline. Сравнить `1e-5` и `1e-2` отдельным controlled experiment после первого успешного run.

## 2. Diagnostic run

```bash
source "$WORK_ROOT/env.sh"
cd "$WORK_ROOT/repos/Isaac-GR00T"

DATA="$WORK_ROOT/data/prepared/cobotmagic_Sim_click_bell__groot_v1"
CFG="$WORK_ROOT/configs/robosyn_cobotmagic_config.py"
RUN="$WORK_ROOT/runs/click_bell_sim_baseline_2k"
mkdir -p "$RUN"

cat > "$RUN/command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "$WORK_ROOT/env.sh"
cd "$WORK_ROOT/repos/Isaac-GR00T"
CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \\
  --base-model-path nvidia/GR00T-N1.7-3B \\
  --dataset-path "$DATA" \\
  --embodiment-tag NEW_EMBODIMENT \\
  --modality-config-path "$CFG" \\
  --num-gpus 1 \\
  --output-dir "$RUN/checkpoints" \\
  --experiment-name click_bell_sim_baseline_2k \\
  --save-total-limit 6 \\
  --save-steps 250 \\
  --max-steps 2000 \\
  --global-batch-size 32 \\
  --gradient-accumulation-steps 1 \\
  --dataloader-num-workers 4 \\
  --learning-rate 1e-4 \\
  --weight-decay 1e-5 \\
  --warmup-ratio 0.05
EOF
chmod +x "$RUN/command.sh"

"$RUN/command.sh" 2>&1 | tee "$RUN/stdout.log"
```

Перед execution проверить, что все flags присутствуют в сохранённом `launch_finetune_help.txt`.

## 3. Batch semantics

В GR00T N1.7 `global_batch_size` — batch одного forward/backward, суммарный по GPUs, **до** gradient accumulation. Effective optimizer-step batch:

```text
global_batch_size × gradient_accumulation_steps
```

На одной A100 80 GB начать с 32. Если OOM:

1. подтвердить, что LLM/visual frozen;
2. проверить посторонние GPU processes;
3. снизить batch 32 → 16 → 8;
4. при необходимости поднять accumulation, но записать effective batch;
5. не менять одновременно LR и batch без отдельного run.

## 4. Step budget

- 10 steps — installation smoke;
- 500 steps — tiny overfit;
- 2,000 steps — diagnostic full-dataset run;
- 5,000–10,000 — первый полноценный run, только если metrics продолжают улучшаться.

Не выбирать steps по привычке: учитывать число episodes, episode sampling rate, batch и повторное количество просмотров данных.

## 5. Checkpoints

Сохранять несколько checkpoints. Последний и минимальный offline error не обязательно лучшие в rollout.

Для возможности resume **не** использовать `save_only_model`. Model-only checkpoint не содержит optimizer/scheduler/RNG state.

Resume:

```bash
# Использовать тот же output_dir и только после проверки, что current checkout/config/data идентичны.
... --resume-from-checkpoint
```

## 6. W&B

Если ключ доступен безопасно:

```bash
uv run wandb login "$WANDB_API_KEY"
```

Добавить `--use-wandb --wandb-project robosyn-groot`. Если ключа нет, не блокировать training: сохранять local logs/JSON/plots.

Минимально логировать:

- train loss;
- learning rate;
- gradient norm;
- GPU memory;
- target/predicted action stats after denormalization;
- metrics per action slice and chunk position;
- checkpoint evaluation results.

## 7. Validation discrepancy

В документации GR00T могут упоминаться `eval_strategy` flags, но фактический `FinetuneConfig` текущего commit может их не экспонировать. Codex обязан сверить `--help`.

Если flags отсутствуют:

- не добавлять undocumented arguments;
- baseline запускать без встроенного validation;
- делать отдельный fixed held-out open-loop evaluation для каждого checkpoint;
- при необходимости добавить validation позже отдельным минимальным patch, не до baseline.

## 8. Мониторинг

В другом tmux window:

```bash
watch -n 1 nvidia-smi
```

Периодически:

```bash
free -h
df -h "$WORK_ROOT"
du -sh "$RUN" "$HF_HOME" "$UV_CACHE_DIR"
```

После run записать peak VRAM, wall time, samples/steps per second, disk growth и exit code.
