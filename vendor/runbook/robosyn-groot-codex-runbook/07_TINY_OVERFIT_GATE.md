# 07. Tiny overfit — обязательный gate

## Зачем

Tiny overfit проверяет весь path:

```text
parquet/video/annotation
→ modality mapping
→ normalization
→ checkpoint loading
→ trainable parameters
→ loss/backprop
→ denormalized predictions
```

Если модель не может почти запомнить 1–4 коротких эпизода, большой dataset только скроет ошибку.

## 1. Сделать tiny dataset

Codex должен создать `tools/subset_lerobot_v21.py`, который:

- принимает список episode IDs;
- копирует соответствующие parquet/videos;
- переписывает `episodes.jsonl` и summary counts;
- сохраняет `tasks.jsonl`, `info.json`, `modality.json` согласованными;
- не меняет ordering dimensions;
- затем запускает GR00T stats generation заново.

Пример:

```bash
python "$WORK_ROOT/tools/subset_lerobot_v21.py" \
  --src "$WORK_ROOT/data/prepared/cobotmagic_Sim_click_bell__groot_v1" \
  --dst "$WORK_ROOT/data/tiny/cobotmagic_Sim_click_bell__episodes_0_1_2_3" \
  --episodes 0 1 2 3
```

## 2. Tiny training

Начать без дополнительных архитектурных изменений:

```bash
source "$WORK_ROOT/env.sh"
cd "$WORK_ROOT/repos/Isaac-GR00T"

TINY="$WORK_ROOT/data/tiny/cobotmagic_Sim_click_bell__episodes_0_1_2_3"
CFG="$WORK_ROOT/configs/robosyn_cobotmagic_config.py"
RUN="$WORK_ROOT/runs/tiny_click_bell_v1"
mkdir -p "$RUN"

CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$TINY" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$CFG" \
  --num-gpus 1 \
  --output-dir "$RUN/checkpoints" \
  --save-total-limit 5 \
  --save-steps 100 \
  --max-steps 500 \
  --global-batch-size 4 \
  --dataloader-num-workers 2 \
  --learning-rate 1e-4 \
  --weight-decay 1e-5 \
  --warmup-ratio 0.05 \
  2>&1 | tee "$RUN/stdout.log"
```

Если batch 4 не помещается, это уже ненормально для A100 80 GB standard scope; сначала диагностировать, а не сразу уменьшать до 1.

## 3. Проверить trainable parameters

До первого optimizer step сохранить:

```text
name | shape | requires_grad | dtype | device
```

Официальный default для N1.7 custom fine-tuning:

```text
tune_projector = true
tune_diffusion_model = true
tune_llm = false
tune_visual = false
```

В official GR00T CLI нет документированного LoRA flag. Не считать LoRA включённой и не добавлять стороннюю LoRA реализацию до рабочего baseline.

## 4. Open-loop на training episodes

```bash
CKPT="$RUN/checkpoints/checkpoint-500"
uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path "$TINY" \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path "$CKPT" \
  --traj-ids 0 1 2 3 \
  --execution-horizon 16 \
  --steps 400 \
  --save-plot-path "$RUN/open_loop" \
  --modality-keys left_arm left_gripper right_arm right_gripper
```

Сверить `--help`: exact flag names могут измениться в current commit.

## 5. Критерии прохождения

- loss finite и значительно ниже начального;
- error уменьшается по checkpoints;
- predictions не flat/constant;
- predicted curves повторяют форму GT на training trajectories;
- после denormalization units/ranges разумны;
- gripper не маскируется и не остаётся постоянным без причины;
- нет систематического shift по времени;
- модель загружена именно из base checkpoint, а не инициализирована с нуля;
- checkpoint можно повторно загрузить в чистом process.

Абсолютного MSE threshold нет: важны запоминание tiny data и монотонный тренд.

## 6. Если tiny overfit не проходит

Проверять в этом порядке:

1. action masks/padding;
2. `meta/modality.json` keys и slices;
3. Python modality config order;
4. dedicated language annotation column;
5. absolute/relative double conversion;
6. normalization/stats shapes;
7. timestamp/action offset;
8. checkpoint loading;
9. `requires_grad` list;
10. LR;
11. only then model code.

Полный run запрещён до исправления.
