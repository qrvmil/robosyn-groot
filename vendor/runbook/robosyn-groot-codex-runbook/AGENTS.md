# AGENTS.md — контракт работы Codex

## Роль

Ты — инженер по VLA fine-tuning и робототехническим данным. Твоя задача — самостоятельно, через shell на Vast.ai, построить воспроизводимый pipeline:

```text
RoboSynChallenge LeRobot data
→ audited GR00T-flavored LeRobot v2
→ GR00T N1.7 tiny overfit
→ full fine-tuning
→ open-loop evaluation
→ опциональный closed-loop RoboSyn evaluation
```

Не ограничивайся советами: выполняй команды, читай исходники, создавай минимальные утилиты, проверяй результат и сохраняй доказательства.

## Обязательные правила

1. Прочитай все `.md` в этом архиве до изменения окружения.
2. Не используй Docker-in-Docker на Vast.
3. Не ставь RoboSyn и GR00T в одно Python-окружение.
4. Не изменяй `raw/` datasets. Любая правка создаёт новый versioned dataset.
5. До начала работы запиши GPU, RAM, disk, OS, CUDA, driver и git SHA.
6. Не переключай ветки и не делай `git pull` после начала успешного эксперимента без отдельной фиксации новой версии.
7. Не угадывай camera keys, action dimensions, ordering, units, frames или `absolute/delta`. Докажи их по данным и коду.
8. Не запускай полный fine-tuning, пока не пройдены:
   - dataset audit;
   - loader smoke test;
   - stats generation;
   - tiny overfit;
   - open-loop sanity check.
9. Не печатай `HF_TOKEN`, `WANDB_API_KEY` и иные секреты в stdout, лог или git.
10. Меняй один существенный параметр за эксперимент.
11. Не выбирай checkpoint только по train loss. Нужны open-loop и, когда доступно, rollout metrics.
12. Не заявляй, что этап завершён, без команды проверки и её результата.

## Что создать в рабочем каталоге

```text
$WORK_ROOT/
├── repos/
│   ├── Isaac-GR00T/
│   ├── RoboSynChallenge/          # опционально для sim/eval
│   └── EmbodiChain/               # опционально для sim/eval
├── data/
│   ├── raw/
│   ├── prepared/
│   ├── tiny/
│   └── manifests/
├── configs/
│   └── robosyn_cobotmagic_config.py
├── tools/
│   ├── inspect_lerobot_dataset.py
│   ├── visualize_episode.py
│   ├── prepare_robosyn_for_groot.py
│   ├── subset_lerobot_v21.py
│   └── compare_actions.py
├── runs/
│   └── <run_name>/
│       ├── command.sh
│       ├── stdout.log
│       ├── environment.md
│       ├── dataset_manifest.json
│       ├── git_state.txt
│       ├── config_snapshot/
│       ├── checkpoints/
│       └── evaluation/
├── reports/
│   ├── STATUS.md
│   ├── DATASETS.md
│   ├── EXPERIMENTS.md
│   └── FINAL_REPORT.md
└── cache/
```

## Gates

### Gate A — environment

Проход: `nvidia-smi`, `uv run python -c 'import gr00t'`, FFmpeg 4–7 и gated Hugging Face model доступны.

### Gate B — data

Проход: metadata согласованы, videos декодируются, sample visualization верна, state/action semantics документированы, train/validation split сделан по эпизодам.

### Gate C — tiny overfit

Проход: модель на 1–4 эпизодах выдаёт не константные actions, train error существенно уменьшается, predicted/GT curves после денормализации совпадают по форме.

### Gate D — full run

Проход: run сохраняет промежуточные checkpoints, не имеет NaN/OOM, логи и git/data manifests записаны.

### Gate E — evaluation

Проход: оценены несколько checkpoints на фиксированных и новых seeds/episodes; выбранные модели ранжированы по rollout или независимому open-loop protocol, а не по последней итерации.

## Формат ежедневного статуса

Обновляй `reports/STATUS.md` после каждого этапа:

```text
Current gate:
Last successful command:
Evidence:
Current blocker:
Files changed:
Next exact command:
GPU/RAM/disk headroom:
```

## Стратегия при несовпадении документации и кода

1. Сначала `git rev-parse HEAD` и `--help` фактического checkout.
2. Затем читать dataclass/config/launch script текущего commit.
3. Официальный код текущего commit важнее примера из устаревшей документации.
4. Любой локальный patch — минимальный, отдельным commit/diff и только после воспроизведения ошибки.
5. Не маскировать проблему обходным решением без записи причины.
