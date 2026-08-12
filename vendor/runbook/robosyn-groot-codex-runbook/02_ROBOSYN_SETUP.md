# 02. RoboSynChallenge: данные и опциональный simulator

## Рекомендуемый путь для Vast fine-tuning

Для обучения GR00T не нужен запущенный EmbodiChain. Начать с готового RoboSyn LeRobot 2.1 dataset. Это снижает число зависимостей и отделяет data/model bugs от simulator bugs.

## 1. Скачать готовые данные

Пример для одной задачи:

```bash
source "$WORK_ROOT/env.sh"
mkdir -p "$WORK_ROOT/data/raw/RoboSynChallenge"

# Команда hf должна быть доступна после установки GR00T либо через отдельный uv tool.
hf download RoboSynChallenge/cobotmagic_Sim_click_bell \
  --repo-type dataset \
  --local-dir "$WORK_ROOT/data/raw/RoboSynChallenge/cobotmagic_Sim_click_bell"

hf download RoboSynChallenge/cobotmagic_Real_click_bell \
  --repo-type dataset \
  --local-dir "$WORK_ROOT/data/raw/RoboSynChallenge/cobotmagic_Real_click_bell"
```

Не скачивать все задачи сразу. Сначала довести `click_bell` до tiny overfit и первого baseline.

## 2. Доступные task names

```text
click_bell
handle_basket
water_pouring
table_rearrangement
items_handover
drawer_open_place
mixer_operating
item_assembly
manipulate_pipette
sample_loading
open_pan
```

Имена Hugging Face datasets:

```text
RoboSynChallenge/cobotmagic_Sim_<task_name>
RoboSynChallenge/cobotmagic_Real_<task_name>
```

## 3. Генерация новых sim trajectories — опционально

RoboSyn умеет собирать LeRobot 3.0 или 2.1:

```bash
bash launch/run_task.sh <task_name> [random|clear] [3_0|2_1] --max_episodes <N>
```

Для GR00T предпочтителен итоговый LeRobot 2.1/v2 layout. При большом сборе разумно сначала писать 3.0 батчами, затем merge и conversion в 2.1, но это отдельный этап после рабочего baseline.

## 4. Local simulator installation на Vast

Только если действительно нужен data generation или closed-loop eval.

### Отдельное Python 3.11 окружение

```bash
source "$WORK_ROOT/env.sh"
cd "$WORK_ROOT/repos"
uv venv --python 3.11 "$WORK_ROOT/.venvs/robosyn"
source "$WORK_ROOT/.venvs/robosyn/bin/activate"

cd "$WORK_ROOT/repos/EmbodiChain"
uv pip install -e . \
  --extra-index-url http://pyp.open3dv.site:2345/simple/ \
  --trusted-host pyp.open3dv.site
uv pip install 'numpy<2.0'

if [ -d embodichain_tasks ]; then
  uv pip install -e ./embodichain_tasks
fi

cd "$WORK_ROOT/repos/RoboSynChallenge"
uv pip install -e .
```

Проверка:

```bash
python - <<'PY'
import dexsim
import embodichain
import robosynchallenge
from embodichain_tasks.tableware.base_agent_env import BaseAgentEnv
print('RoboSyn imports OK')
PY
```

### Scripted smoke test

```bash
cd "$WORK_ROOT/repos/RoboSynChallenge"
unset DISPLAY
bash launch/run_task.sh click_bell clear 2_1 --max_episodes 1
```

Если headless flag управляется config/script иначе, сначала посмотреть `bash launch/run_task.sh -h` и код текущего commit. Не переходить к GR00T adapter, пока scripted environment не проходит самостоятельно.

## 5. Почему simulator и trainer должны жить отдельно

- RoboSyn/EmbodiChain ориентированы на Python 3.11 и нативный DexSim stack.
- GR00T N1.7 dGPU workflow ориентирован на Python 3.12, PyTorch 2.7+, flash-attn и torchcodec.
- Смешивание приводит к трудно диагностируемым конфликтам `torch`, `transformers`, LeRobot, video backend и native libraries.

Связь между ними должна быть через **dataset files** и позднее через policy adapter/process boundary, а не через общий site-packages.
