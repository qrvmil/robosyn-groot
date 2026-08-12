# 03. Датасеты: реестр, layout и документация

## 1. Каталоги

```text
$WORK_ROOT/data/
├── raw/                    # неизменяемые скачанные/собранные данные
├── prepared/               # GR00T-compatible versions
├── tiny/                   # 1–4 episode subsets
└── manifests/              # hashes, schemas, audit reports
```

Каждая prepared-версия именуется явно:

```text
cobotmagic_Sim_click_bell__groot_v1
cobotmagic_Real_click_bell__groot_v1
cobotmagic_Mix_click_bell__groot_v1
```

Нельзя перезаписывать существующую prepared-версию. Любое изменение schema, annotations, action horizon или фильтрации создаёт `v2`, `v3`, ...

## 2. Ожидаемый LeRobot v2/2.1 layout

```text
dataset_root/
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── tasks.jsonl
│   ├── modality.json          # GR00T-specific
│   ├── stats.json             # генерируется GR00T
│   └── relative_stats.json    # генерируется GR00T при relative actions
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet
└── videos/
    └── chunk-000/
        └── observation.images.<camera>/
            └── episode_000000.mp4
```

Фактический layout проверять по `meta/info.json`; не переименовывать файлы до аудита.

## 3. Что занести в dataset card

Для каждого dataset создать отдельный файл на основе `templates/DATASET_CARD_TEMPLATE.md`:

- exact source/repo ID и revision;
- raw path и prepared path;
- codebase version;
- число episodes/frames/tasks;
- FPS/control dt;
- camera keys, resolution, codec;
- state dimension, slices, units, frame;
- action dimension, slices, units, frame;
- stored action representation: absolute target, delta, velocity или другое;
- gripper semantics/range;
- task instruction source;
- доля pauses/clipping/failed episodes;
- train/validation split по episode IDs;
- SHA256 manifest;
- known issues.

## 4. Нельзя предполагать

Даже если CobotMagic обычно выглядит как две 6-DoF руки + два grippers, запрещено заранее записывать размерность `14`. Нужно проверить:

1. `meta/info.json` feature shape;
2. реальные массивы `observation.state` и `action` в parquet;
3. RoboSyn robot/action config текущего commit;
4. фактическую семантику ordering.

## 5. Split protocol

Разделение только по целым эпизодам. Нельзя случайно разрезать frames одного episode между train и validation.

Минимум:

```text
train: 80–90% episodes
validation: 10–20% episodes
```

Для малых datasets дополнительно держать фиксированные episode IDs для open-loop. Для OOD проверки использовать новые object poses/start states/random seeds, а не frames из тех же trajectories.

## 6. Sim/real mixture

Первый baseline — один dataset, чтобы изолировать bugs. Затем сравнить:

1. sim-only;
2. real-only;
3. sim pretrain → real fine-tune;
4. joint sim+real;
5. joint с балансировкой rare stages/tasks.

GR00T N1.7 поддерживает несколько dataset paths, разделённых `os.pathsep` (`:` на Linux), и `ds_weights_alpha` для power-law weighting по размеру datasets. Это не эквивалент произвольным вручную заданным mix ratios; если нужны точные веса, использовать явный merged/sampled dataset либо минимальное расширение config после baseline.

## 7. Version manifest

```bash
DATASET="$WORK_ROOT/data/raw/RoboSynChallenge/cobotmagic_Sim_click_bell"
find "$DATASET" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$WORK_ROOT/data/manifests/cobotmagic_Sim_click_bell.sha256"
```

Для больших videos hashing займёт время, но это единственный надёжный способ доказать, что два run использовали одинаковые данные.
