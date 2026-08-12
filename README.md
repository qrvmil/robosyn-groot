# RoboSyn Click Bell → NVIDIA GR00T N1.7 Fine-Tuning

Этот workspace полностью подготовлен для дообучения `nvidia/GR00T-N1.7-3B` на симуляционных демонстрациях задачи **Click the bell**. Исходные данные, семантика робота, статистики нормализации, loader, преобразование actions, tiny-training, загрузка checkpoint и open-loop inference проверены. Полный 2 000-шаговый запуск подготовлен, но ещё не запускался.

## Быстрый старт

Перейти в workspace:

```bash
cd /workspace/challenge/robosyn-groot
```

Повторить машинную проверку готовности:

```bash
repos/Isaac-GR00T/.venv/bin/python tools/verify_readiness.py \
  --work-root /workspace/challenge/robosyn-groot \
  --run-name click_bell_sim_baseline_2k \
  --output reports/READINESS.md
```

Ожидаемый результат: `READY`.

Запустить полное дообучение:

```bash
bash /workspace/challenge/robosyn-groot/runs/click_bell_sim_baseline_2k/command.sh \
  2>&1 | tee /workspace/challenge/robosyn-groot/runs/click_bell_sim_baseline_2k/stdout.log
```

Это единственная команда, необходимая для старта. Скрипт сам активирует окружение, выбирает GPU 0 и запускает закреплённую версию Isaac-GR00T. Доступ к Hugging Face metadata должен оставаться включённым: Cosmos processor выполняет metadata lookup даже при наличии файлов в локальном cache.

## Что было сделано

1. Загружен и закреплён исходный Hugging Face dataset `RoboSynChallenge/cobotmagic_Sim_click_bell`.
2. Проверены SHA256 всех 4 006 исходных файлов; raw-копия сделана read-only.
3. Проведён полный аудит 1 000 эпизодов, 74 000 кадров и 3 000 видео.
4. Все видео проверены через `ffprobe`/FFmpeg: AV1, 640×480, 25 FPS, 74 кадра, три камеры.
5. Вручную просмотрены пять репрезентативных эпизодов и 15 контрольных кадров.
6. Восстановлена семантика 14-D state/action и физическое соответствие камер.
7. Создан GR00T-совместимый LeRobot v2.1 dataset с языковой командой `Click the bell`.
8. Сгенерированы обычные и relative-action statistics.
9. Проверен официальный GR00T loader: изображения, state, action horizon и язык имеют ожидаемые формы.
10. Проверено преобразование action → normalization → denormalization на 128 примерах: максимальная ошибка `5.55e-17`.
11. Выполнено 500-шаговое tiny-training на четырёх эпизодах без NaN/OOM.
12. Tiny checkpoint загружен обратно и проверен open-loop inference на четырёх эпизодах.
13. Создан полный 2 000-шаговый launcher и машинный preflight. После исправления offline-конфигурации тем же full-profile выполнен отдельный успешный one-step smoke: полный dataset, batch 32, четыре worker-а, loss `1.19684`. Основной 2 000-шаговый run не запускался.

Подробные доказательства находятся в:

- [`reports/READINESS.md`](reports/READINESS.md) — итоговый preflight;
- [`reports/STATUS.md`](reports/STATUS.md) — текущий статус;
- [`reports/PREPARED_DATASET.md`](reports/PREPARED_DATASET.md) — подготовка данных;
- [`reports/MODALITY_CONFIG_REVIEW.md`](reports/MODALITY_CONFIG_REVIEW.md) — семантика state/action/cameras;
- [`reports/EXPERIMENTS.md`](reports/EXPERIMENTS.md) — выполненные и подготовленные эксперименты;
- [`runs/click_bell_sim_baseline_2k/launch_manifest.json`](runs/click_bell_sim_baseline_2k/launch_manifest.json) — точные revisions, параметры и checksums.

## Какой checkpoint используется

### Базовый checkpoint полного обучения

```text
Repository: nvidia/GR00T-N1.7-3B
Revision:   2fc962b973bccdd5d8ce4f67cc63b264d6886495
```

Локальный immutable snapshot:

```text
/workspace/challenge/robosyn-groot/cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495
```

Полный launcher начинает обучение именно с этого базового checkpoint NVIDIA. Он не использует плавающий `main`, поэтому повторный запуск получает те же исходные веса.

Версия кода Isaac-GR00T:

```text
376ba890cff8c9de64d71d982772a9c36185fdd7
```

### Локальный tiny checkpoint

Для проверки пайплайна был получен отдельный checkpoint:

```text
runs/tiny_click_bell_v1/checkpoints/tiny_click_bell_v1/checkpoint-500
```

Он прошёл reload и open-loop проверку, но **не является стартовой точкой полного обучения**. Он обучался лишь на четырёх эпизодах и служит интеграционным тестом.

### Будущие checkpoints полного запуска

После запуска они появятся в:

```text
runs/click_bell_sim_baseline_2k/checkpoints/click_bell_sim_baseline_2k/
```

Сохранение выполняется каждые 250 optimizer steps; сохраняются шесть последних полных checkpoints с состояниями модели, optimizer, scheduler и RNG.

## Модель и обучаемые параметры

Проверенный tiny-run использовал тот же trainable scope, что и подготовленный full-run:

| Параметр | Значение |
|---|---:|
| Архитектура | `Gr00tN1d7` |
| Базовая модель | `nvidia/GR00T-N1.7-3B` |
| Vision-language backbone | `nvidia/Cosmos-Reason2-2B` |
| Всего параметров | 3 144 016 000 |
| Обучаемых параметров | 1 620 515 968 |
| Доля обучаемых | 51,54% |
| Compute dtype | BF16 |
| LLM | frozen |
| Visual backbone | frozen |
| Multimodal projector | trainable |
| Diffusion action model | trainable |
| Diffusion inference steps | 4 |
| State history | 1 кадр |
| Внутренний max state/action dim | 132 |
| Фактический state/action dim | 14 |

Базовая архитектура допускает action horizon до 40, но modality config этой задачи использует 13 будущих действий: `0..12`.

## Входы и выходы модели

### Video

Модель получает один синхронный кадр с трёх камер:

| GR00T key | Исходная камера | Физический вид |
|---|---|---|
| `front` | `cam_high.color` | базовая/верхняя камера |
| `left_wrist` | `cam_left_wrist.color` | левое запястье |
| `right_wrist` | `cam_right_wrist.color` | правое запястье |

### Language

```text
annotation.human.task_description = "Click the bell"
```

### State и action

14 координат разделены одинаково:

| Группа | Индексы | Размер | Action representation |
|---|---:|---:|---|
| `left_arm` | `0:6` | 6 | relative к `state.left_arm` |
| `left_gripper` | `6:7` | 1 | absolute |
| `right_arm` | `7:13` | 6 | relative к `state.right_arm` |
| `right_gripper` | `13:14` | 1 | absolute |

Action horizon равен 13 шагам при 25 FPS, то есть `0,52 секунды`.

Используемый modality config: [`configs/robosyn_cobotmagic_config.py`](configs/robosyn_cobotmagic_config.py).

## Параметры полного обучения

Фактический launcher: [`runs/click_bell_sim_baseline_2k/command.sh`](runs/click_bell_sim_baseline_2k/command.sh).

| Параметр | Значение |
|---|---:|
| GPU | 1 × NVIDIA A100 80 GB |
| Global batch size | 32 |
| Gradient accumulation | 1 |
| Effective optimizer-step batch | 32 |
| DataLoader workers | 4 |
| Learning rate | `1e-4` |
| Optimizer | AdamW (`adamw_torch`) |
| Scheduler | cosine |
| Weight decay | `1e-5` |
| Warmup ratio | `0.05` |
| Max gradient norm | `1.0` |
| Steps | 2 000 |
| Save interval | 250 |
| Retained checkpoints | 6 |
| State dropout | `0.0` |
| Normalization | q01/q99 percentiles |
| Episode sampling rate | `1.0` |
| W&B | disabled |

Batch 32 успешно выполнил один реальный optimizer step на полном 1 000-эпизодном dataset (`train_loss=1.19684`) и помещается в A100 80 GB. Полноценный 2 000-шаговый прогон ещё не выполнялся. Если позднее всё же возникнет CUDA OOM из-за изменившегося окружения, уменьшить `--global-batch-size` сначала до `16`, затем при необходимости до `8`. При изменении batch для сохранения effective batch можно увеличить gradient accumulation.

## Как следить за обучением

Лог:

```bash
tail -f /workspace/challenge/robosyn-groot/runs/click_bell_sim_baseline_2k/stdout.log
```

GPU:

```bash
watch -n 2 nvidia-smi
```

Checkpoints:

```bash
find /workspace/challenge/robosyn-groot/runs/click_bell_sim_baseline_2k/checkpoints \
  -maxdepth 2 -type d -name 'checkpoint-*' | sort
```

В нормальном логе должны появляться конечные `loss` и `grad_norm`, а на шагах 250, 500 и далее — сообщения о сохранении checkpoint. `NaN`, `CUDA out of memory`, DataLoader traceback или отсутствие движения loss требуют остановки и диагностики.

## Как продолжить прерванное обучение

Текущий launcher специально начинает новый эксперимент и не включает silent resume. Для продолжения того же run необходимо добавить флаг:

```bash
--resume-from-checkpoint
```

в [`runs/click_bell_sim_baseline_2k/command.sh`](runs/click_bell_sim_baseline_2k/command.sh), сохранив прежние `--output-dir` и `--experiment-name`. GR00T найдёт последний `checkpoint-*` в output directory.

Не добавлять `--save-only-model`: model-only checkpoints не содержат optimizer/scheduler/RNG state и не подходят для полноценного resume.

## Ограничение текущего датасета

Обе gripper-action координаты равны нулю во **всех 74 000 кадрах** исходного Click Bell dataset. Это означает:

- модель может учиться движениям двух рук и контакту с колокольчиком;
- модель получает цель `0` для обоих grippers;
- модель не может научиться открытию или закрытию захвата по данным, где такого действия нет.

Tiny open-loop корректно предсказывает нулевые gripper actions. Для манипуляции предметами новый dataset обязан содержать физически корректные ненулевые open/close targets.

## Как добавить вторую задачу, например «положить в корзину»

Пинованный `launch_finetune.py` принимает один LeRobot root либо несколько roots, разделённых системным `os.pathsep`. На Linux разделитель — двоеточие:

```text
/path/to/click_bell:/path/to/put_in_basket
```

Не нужно вручную смешивать parquet/video-файлы двух datasets в одной директории. Безопаснее подготовить и проверить каждый dataset отдельно, а затем передать оба пути launcher-у.

### Требования совместимости

Простой совместный запуск возможен, если второй dataset использует:

- того же робота и тот же `NEW_EMBODIMENT`;
- тот же порядок и физический смысл 14-D state/action;
- те же единицы и conventions gripper;
- те же modality keys и камеры `front`, `left_wrist`, `right_wrist`;
- совместимую частоту кадров и action horizon;
- LeRobot v2/v2.1 metadata;
- правильный текст задачи в `meta/tasks.jsonl` и `task_index` каждого кадра.

Если робот, размер state/action или набор камер отличается, одной строкой `dataset-path` это не решается: нужен новый embodiment/modality config и отдельная валидация.

### 1. Положить raw dataset отдельно

Пример:

```text
data/raw/RoboSynChallenge/cobotmagic_Sim_put_in_basket
```

Не изменять уже проверенный Click Bell raw snapshot.

### 2. Провести аудит нового dataset

```bash
cd /workspace/challenge/robosyn-groot

repos/Isaac-GR00T/.venv/bin/python tools/inspect_lerobot_dataset.py \
  data/raw/RoboSynChallenge/cobotmagic_Sim_put_in_basket \
  --output reports/put_in_basket.audit.json
```

Проверить количество эпизодов/кадров, конечность state/action, video decode, FPS, размерности, task metadata и реальные изменения gripper action.

### 3. Создать semantics нового dataset

Взять за основу:

```text
configs/robosyn_cobotmagic_semantics.json
```

и сохранить, например, как:

```text
configs/put_in_basket_semantics.json
```

Обновить dataset repository/revision, language task и любые реально отличающиеся mappings. Не копировать Click Bell mappings вслепую: сначала подтвердить их аудитом и кодом среды.

`meta/tasks.jsonl` нового dataset должен содержать точную команду, например:

```text
Put the object in the basket
```

### 4. Создать отдельную prepared-копию

Если schema робота совместима:

```bash
repos/Isaac-GR00T/.venv/bin/python tools/prepare_robosyn_for_groot.py \
  --src data/raw/RoboSynChallenge/cobotmagic_Sim_put_in_basket \
  --dst data/prepared/cobotmagic_Sim_put_in_basket__groot_v1 \
  --semantics configs/put_in_basket_semantics.json
```

Утилита откажется перезаписывать существующий destination. Это защищает уже подготовленные данные.

Если modality полностью совпадает, обе задачи должны использовать текущий [`configs/robosyn_cobotmagic_config.py`](configs/robosyn_cobotmagic_config.py). Не генерировать второй конфликтующий config для того же `NEW_EMBODIMENT`.

### 5. Сгенерировать и проверить statistics

Launcher делает это автоматически на rank 0, но перед длинным запуском лучше выполнить отдельно:

```bash
cd /workspace/challenge/robosyn-groot/repos/Isaac-GR00T

.venv/bin/python gr00t/data/stats.py \
  --dataset-path /workspace/challenge/robosyn-groot/data/prepared/cobotmagic_Sim_put_in_basket__groot_v1 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path /workspace/challenge/robosyn-groot/configs/robosyn_cobotmagic_config.py
```

После этого нужно повторить loader smoke и action round-trip для нового dataset, а также tiny-training/open-loop с эпизодами обеих задач.

### 6. Создать отдельный multi-task launcher

Скопировать baseline launcher под новым именем, например:

```text
runs/click_bell_and_basket_v1/command.sh
```

Изменить dataset argument на:

```bash
--dataset-path "$WORK_ROOT/data/prepared/cobotmagic_Sim_click_bell__groot_v1:$WORK_ROOT/data/prepared/cobotmagic_Sim_put_in_basket__groot_v1"
```

и обязательно изменить:

```bash
--output-dir "$WORK_ROOT/runs/click_bell_and_basket_v1/checkpoints"
--experiment-name click_bell_and_basket_v1
```

Для примерно равного prior-веса datasets добавить:

```bash
--ds-weights-alpha 0
```

Без `--ds-weights-alpha` launcher задаёт вес пропорционально размеру dataset. Например, 100 basket-эпизодов могут редко встречаться на фоне 1 000 Click Bell эпизодов. `alpha=0` задаёт одинаковый length-based weight каждому dataset; фактический sampling также проходит через shard scheduling, поэтому его распределение следует подтвердить в startup log.

Multi-task обучение рекомендуется начинать с исходного NVIDIA checkpoint, а не с button-only tiny/full checkpoint. Это снижает начальный перекос и риск забывания второй задачи.

### 7. Проверить multi-task gate до длинного запуска

Минимальный gate:

1. 2–4 конечных и декодируемых эпизода каждой задачи.
2. Ненулевые и физически корректные gripper transitions в basket-эпизодах.
3. Loader smoke для обоих roots.
4. Action normalization round-trip для обоих roots.
5. Короткое обучение без NaN/OOM.
6. Checkpoint reload.
7. Open-loop метрики отдельно для `Click the bell` и `Put the object in the basket`.
8. Проверка того, что language annotation действительно различает задачи.

Текущий `tools/verify_readiness.py` и его checksums относятся к неизменённому single-task baseline `click_bell_sim_baseline_2k`. После добавления данных нужен отдельный launch manifest/preflight, а не игнорирование checksum mismatch.

## Hugging Face

На текущей машине веса уже находятся в локальном cache. Однако launcher намеренно не включает `HF_HUB_OFFLINE=1` или `TRANSFORMERS_OFFLINE=1`: Transformers выполняет metadata lookup для `nvidia/Cosmos-Reason2-2B` во время создания processor. Поэтому при старте требуется доступ к `huggingface.co`, даже если основные веса уже скачаны.

При переносе на другую машину нужно получить доступ к `nvidia/GR00T-N1.7-3B`, авторизовать `hf`, скачать именно revision `2fc962b973bccdd5d8ce4f67cc63b264d6886495` и либо сохранить тот же layout cache, либо изменить `--base-model-path` на новый локальный snapshot.

Не записывать HF token, W&B key или другие secrets в launcher, README, git либо логи.

## Основные пути

| Назначение | Путь |
|---|---|
| Raw dataset | `data/raw/RoboSynChallenge/cobotmagic_Sim_click_bell` |
| Prepared dataset | `data/prepared/cobotmagic_Sim_click_bell__groot_v1` |
| Tiny dataset | `data/tiny/cobotmagic_click_bell_4ep_v1` |
| Modality config | `configs/robosyn_cobotmagic_config.py` |
| Semantic specification | `configs/robosyn_cobotmagic_semantics.json` |
| Full launcher | `runs/click_bell_sim_baseline_2k/command.sh` |
| Full launch manifest | `runs/click_bell_sim_baseline_2k/launch_manifest.json` |
| Tiny training log | `runs/tiny_click_bell_v1/stdout.log` |
| Tiny open-loop metrics | `runs/tiny_click_bell_v1/evaluation/metrics.json` |
| Final readiness report | `reports/READINESS.md` |

## Текущее состояние

- Подготовка: завершена.
- Итоговый preflight single-task baseline: `READY`.
- Full-profile startup smoke: 1/1 optimizer step, batch 32, exit code 0.
- Полный 2 000-шаговый run: не запускался.
- Активного `launch_finetune.py` процесса нет.
- Следующее действие для single-task baseline — выполнить команду из раздела «Быстрый старт».
