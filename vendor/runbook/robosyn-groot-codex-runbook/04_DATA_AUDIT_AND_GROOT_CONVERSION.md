# 04. Data audit и конвертация RoboSyn → GR00T LeRobot

GR00T N1.7 читает свой вариант LeRobot v2: стандартные videos/parquet/meta плюс `meta/modality.json`, корректная language annotation и вычисленные normalization statistics.

## Этап 1. Структурная проверка

```bash
DATASET="$WORK_ROOT/data/raw/RoboSynChallenge/cobotmagic_Sim_click_bell"
find "$DATASET/meta" -maxdepth 2 -type f -print | sort
jq . "$DATASET/meta/info.json" | sed -n '1,240p'
head -5 "$DATASET/meta/tasks.jsonl" || true
head -5 "$DATASET/meta/episodes.jsonl" || true
find "$DATASET/data" -type f | head
find "$DATASET/videos" -type f | head
```

Codex должен создать `tools/inspect_lerobot_dataset.py`, который печатает и сохраняет JSON report:

- все columns и Arrow types;
- shape первых/последних frame каждого low-dimensional field;
- min/max/mean/std и долю NaN/Inf;
- episode lengths;
- timestamps/fps/dt;
- camera keys и video metadata;
- task indices/annotations;
- долю frames с почти нулевым action;
- долю значений на физических/нормализованных limits.

## Этап 2. Визуальный audit

Создать `tools/visualize_episode.py` и минимум для 5 episodes сохранить contact sheet или mp4 с overlay:

```text
camera frames
instruction
timestep/timestamp
state values or selected joints
action values
```

Проверить вручную:

- observation и action синхронизированы;
- camera order не перепутан;
- left/right wrist cameras соответствуют рукам;
- action начинается в правильный момент;
- gripper sign/range совпадает с изображением;
- нет систематического one-frame shift;
- failed/recovery trajectories не ошибочно помечены как success.

## Этап 3. Семантика state/action

Документировать для каждой slice:

```text
name | start:end | units | coordinate frame | absolute/delta | expected range
```

Особенно проверить:

- joint order;
- radians vs degrees;
- position vs velocity;
- absolute joint target vs delta;
- gripper open/close convention;
- control frequency;
- одинаковы ли state/action ordering и dimension.

GR00T `ActionRepresentation.RELATIVE` предполагает, что dataset хранит absolute state и absolute target action, а relative conversion делает processor. Не применять relative conversion дважды.

## Этап 4. Создать prepared copy

```bash
RAW="$WORK_ROOT/data/raw/RoboSynChallenge/cobotmagic_Sim_click_bell"
PREP="$WORK_ROOT/data/prepared/cobotmagic_Sim_click_bell__groot_v1"
rsync -a --info=progress2 "$RAW/" "$PREP/"
chmod -R u+w "$PREP"
```

Raw остаётся read-only по процессу, даже если UNIX permissions этого не запрещают.

## Этап 5. Language annotations

GR00T требует согласованности трёх слоёв:

1. parquet column, например `annotation.human.task_description`;
2. ключ `human.task_description` в `meta/modality.json` under `annotation`;
3. `annotation.human.task_description` в Python modality config.

Если parquet содержит только `task_index`, подготовительный script должен детерминированно добавить dedicated annotation column как копию соответствующего task index, обновить `meta/info.json` и сохранить audit diff. Не заменять текст инструкции на произвольную формулировку: использовать `meta/tasks.jsonl`/RoboSyn gym instruction.

## Этап 6. `meta/modality.json`

Создать его из доказанных slices. Шаблон — только форма, не готовые индексы:

```json
{
  "state": {
    "left_arm": {"start": 0, "end": "<derived>"},
    "left_gripper": {"start": "<derived>", "end": "<derived>"},
    "right_arm": {"start": "<derived>", "end": "<derived>"},
    "right_gripper": {"start": "<derived>", "end": "<derived>"}
  },
  "action": {
    "left_arm": {"start": 0, "end": "<derived>"},
    "left_gripper": {"start": "<derived>", "end": "<derived>"},
    "right_arm": {"start": "<derived>", "end": "<derived>"},
    "right_gripper": {"start": "<derived>", "end": "<derived>"}
  },
  "video": {
    "cam_high": {"original_key": "observation.images.cam_high"},
    "cam_left_wrist": {"original_key": "observation.images.cam_left_wrist"},
    "cam_right_wrist": {"original_key": "observation.images.cam_right_wrist"}
  },
  "annotation": {
    "human.task_description": {"original_key": "annotation.human.task_description"}
  }
}
```

Удалить отсутствующие камеры; не создавать fake views.

## Этап 7. Video compatibility

GR00T использует torchcodec и ожидает декодируемые MP4. Проверить все codecs:

```bash
find "$PREP/videos" -type f -name '*.mp4' -print0 | \
  xargs -0 -n1 ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,r_frame_rate \
  -of default=noprint_wrappers=1
```

Если codec не H.264 и torchcodec не читает файл, создать новую prepared version с H.264 conversion. Не менять raw.

## Этап 8. Python modality config и stats

После создания config из `06_ROBOSYN_MODALITY_CONFIG.md`:

```bash
cd "$WORK_ROOT/repos/Isaac-GR00T"
source "$WORK_ROOT/env.sh"

uv run python gr00t/data/stats.py \
  --dataset-path "$PREP" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$WORK_ROOT/configs/robosyn_cobotmagic_config.py"
```

Проверить наличие и размеры:

```bash
jq 'keys' "$PREP/meta/stats.json"
jq 'keys' "$PREP/meta/relative_stats.json"
```

После изменения action horizon, slices или action representation stats нужно пересчитать.

## Этап 9. Normalize → denormalize

Создать automated check на нескольких samples:

- исходные absolute state/action;
- processed/normalized tensors;
- reconstructed/denormalized actions;
- max error per dimension;
- clipping rate q01/q99, если используются percentiles.

Нельзя запускать training, если:

- shapes не совпадают;
- arm/gripper slices перепутаны;
- predictions после inverse transform не имеют исходных units;
- большая доля targets постоянно clipped.

## Data gate checklist

- [ ] 5+ episodes визуализированы.
- [ ] Observation/action sync проверена.
- [ ] State/action slices документированы.
- [ ] Absolute/delta semantics доказаны.
- [ ] Language annotation согласована на трёх уровнях.
- [ ] Video decode проходит.
- [ ] Train/validation episode IDs зафиксированы.
- [ ] `stats.json` и `relative_stats.json` созданы.
- [ ] Normalize/denormalize sanity check проходит.
- [ ] Dataset card и SHA manifest сохранены.
