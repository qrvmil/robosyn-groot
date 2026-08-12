# RoboSyn → GR00T N1.7 on Vast.ai: Codex Runbook

Этот архив — рабочая инструкция для Codex, который получает shell-доступ к уже арендованному Vast.ai-инстансу и должен довести пайплайн от данных RoboSynChallenge до воспроизводимого fine-tuning GR00T N1.7.

## Целевая машина

Ожидаемая конфигурация:

- 1× NVIDIA A100 PCIe 80 GB;
- около 16 CPU cores;
- около 256 GB RAM;
- около 900 GB локального диска;
- x86_64, Ubuntu/CUDA-шаблон Vast.ai;
- доступ по SSH, работа от `root` внутри контейнера.

Параметры должны быть перепроверены командами из `00_VAST_INSTANCE_AND_STORAGE.md`; не полагаться на описание оффера вслепую.

## Быстрый маршрут

```text
проверить Vast и хранилище
→ зафиксировать репозитории и версии
→ поднять чистое GR00T-окружение
→ воспроизвести официальный GR00T smoke test
→ скачать один RoboSyn LeRobot 2.1 dataset
→ провести полный data audit
→ добавить GR00T metadata/modality config
→ пересчитать stats
→ tiny overfit на 1–4 эпизодах
→ open-loop проверка
→ первый полноценный run
→ LR/scope/chunk/data sweeps
→ closed-loop rollout в RoboSyn
→ выбрать checkpoints по rollout, а не только loss
→ выгрузить артефакты до уничтожения Vast instance
```

## Порядок чтения

1. `AGENTS.md` — обязательные правила для Codex.
2. `00_VAST_INSTANCE_AND_STORAGE.md` — Vast, диск, кэши, tmux, persistence.
3. `01_REPOSITORIES_AND_VERSIONING.md` — клоны, submodules, SHA и lockfiles.
4. `02_ROBOSYN_SETUP.md` — данные RoboSyn и опциональный симулятор.
5. `03_DATASETS.md` — реестр и документирование датасетов.
6. `04_DATA_AUDIT_AND_GROOT_CONVERSION.md` — проверка и приведение к GR00T LeRobot.
7. `05_GROOT_ENVIRONMENT.md` — установка GR00T N1.7.
8. `06_ROBOSYN_MODALITY_CONFIG.md` — state/action/video/language mapping.
9. `07_TINY_OVERFIT_GATE.md` — обязательный gate перед большим run.
10. `08_FINE_TUNING_RUNBOOK.md` — основной запуск на A100 80 GB.
11. `09_EVALUATION_AND_ROLLOUT.md` — open-loop и RoboSyn closed-loop.
12. `10_EXPERIMENT_RECIPE.md` — порядок экспериментов из VLA cookbook.
13. `11_TROUBLESHOOTING.md` — типовые сбои.
14. `12_BACKUP_AND_HANDOFF.md` — сохранение результатов и отчёт.
15. `SOURCES.md` — первичные источники и правило их приоритета.

## Ключевые решения

- **Не использовать Docker-in-Docker.** Vast instance уже является контейнером. RoboSyn и GR00T ставятся локально в раздельные Python-окружения.
- **Не смешивать зависимости.** GR00T использует Python 3.12 и собственную `.venv`; RoboSyn/EmbodiChain — отдельное Python 3.11 окружение.
- **Для fine-tuning симулятор не нужен.** Сначала скачать готовые RoboSyn datasets и обучить GR00T. Полный RoboSyn simulator разворачивать только для генерации новых данных или closed-loop evaluation.
- **Raw data неизменяемы.** Любая конвертация идёт в новую versioned-папку.
- **Размерности и семантику action/state не угадывать.** Их нужно вывести из фактических metadata, parquet и RoboSyn config.
- **Большой run запрещён до tiny overfit.**
- **Первый baseline должен быть максимально близок к официальному GR00T workflow.** Рекомендации из VLA cookbook вводятся затем как контролируемые эксперименты по одному параметру.
