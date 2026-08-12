# 01. Репозитории, версии и воспроизводимость

## 1. GR00T

```bash
source "$WORK_ROOT/env.sh"
cd "$WORK_ROOT/repos"
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T
git submodule update --init --recursive
git lfs pull
```

Сразу зафиксировать состояние:

```bash
git rev-parse HEAD | tee "$WORK_ROOT/reports/groot_commit.txt"
git status --short --branch | tee -a "$WORK_ROOT/reports/groot_commit.txt"
git submodule status --recursive | tee "$WORK_ROOT/reports/groot_submodules.txt"
```

После первого успешного baseline не делать `git pull` в этой рабочей копии. Для новой версии создать отдельный worktree/clone.

## 2. RoboSyn и EmbodiChain — только если нужен simulator/data generation

Для training на готовых datasets эти репозитории не обязательны. Если нужны генерация данных или closed-loop eval:

```bash
cd "$WORK_ROOT/repos"
git clone https://github.com/EDEM-AI/RoboSynChallenge.git
git clone https://github.com/DexForce/EmbodiChain.git

git -C RoboSynChallenge rev-parse HEAD | tee "$WORK_ROOT/reports/robosyn_commit.txt"
git -C EmbodiChain rev-parse HEAD | tee "$WORK_ROOT/reports/embodichain_commit.txt"
```

### Не пиновать `EmbodiChain v0.2.3` вслепую

Документация RoboSyn исторически рекомендовала `v0.2.3`, но текущий RoboSyn код импортирует `embodichain_tasks`. Поэтому Codex обязан проверить фактическую совместимость:

```bash
find "$WORK_ROOT/repos/EmbodiChain" -maxdepth 2 -type d -name embodichain_tasks -print
```

Если пакет находится внутри current EmbodiChain checkout, он устанавливается отдельно командой `pip install -e ./embodichain_tasks`. Если выбранный tag его не содержит, не пытаться лечить это `PYTHONPATH`; выбрать совместимый commit и зафиксировать SHA.

## 3. Manifest окружения

После установки каждого окружения создать:

```bash
{
  date -Is
  uname -a
  cat /etc/os-release
  nvidia-smi
  nvcc --version || true
  uv --version
  git -C "$WORK_ROOT/repos/Isaac-GR00T" rev-parse HEAD
  git -C "$WORK_ROOT/repos/Isaac-GR00T" submodule status --recursive
} > "$WORK_ROOT/reports/software_manifest.txt"
```

В каждом run дополнительно сохранять:

```bash
git -C "$WORK_ROOT/repos/Isaac-GR00T" diff > "$RUN_DIR/groot.diff"
git -C "$WORK_ROOT/repos/Isaac-GR00T" status --porcelain=v1 > "$RUN_DIR/groot.status"
uv pip freeze > "$RUN_DIR/python_freeze.txt"
```

## 4. Правило source priority

1. фактический код и `--help` зафиксированного commit;
2. официальная документация того же commit;
3. этот runbook;
4. сторонние советы.

Если документация обещает CLI flag, которого нет в `uv run python ... --help`, не передавать flag и не скрывать несовпадение. Записать discrepancy в `reports/STATUS.md`.
