# 06. Modality config для RoboSyn CobotMagic

## Цель

Создать `$WORK_ROOT/configs/robosyn_cobotmagic_config.py`, который связывает фактический `meta/modality.json` dataset с GR00T processor.

## Базовая форма

```python
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

ACTION_HORIZON = 16  # заменить на значение, выбранное по фактическому FPS

robosyn_cobotmagic_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "cam_high",
            "cam_left_wrist",
            "cam_right_wrist",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm",
            "left_gripper",
            "right_arm",
            "right_gripper",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=[
            "left_arm",
            "left_gripper",
            "right_arm",
            "right_gripper",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="left_arm",
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="left_gripper",
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="right_arm",
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="right_gripper",
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(
    robosyn_cobotmagic_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
```

Это **template**, не готовый config. Codex обязан удалить отсутствующие камеры и изменить keys/order по фактическому dataset.

## Как выбрать action representation

- Arm joint targets: начать с `RELATIVE`, только если dataset хранит absolute state и absolute target action.
- Gripper: обычно `ABSOLUTE`, особенно для open/close target.
- Если action — velocity/delta, `RELATIVE` поверх него применять нельзя.
- Если state key и action key называются по-разному, обязательно указать `state_key`.
- CobotMagic joint-space — `ActionType.NON_EEF`. Не использовать EEF config без Cartesian pose dataset.

## Как выбрать horizon

Сначала считать FPS/control frequency из metadata и timestamps.

```text
baseline_horizon ≈ round(0.5 s × control_frequency)
```

При 25 Hz это 12–13 steps; shipped custom-embodiment example использует 16. Разумный первый вариант — 16, если dataset около 25–30 Hz, затем sweep 8/16/32.

После каждого изменения `delta_indices` обязательно пересчитать `meta/relative_stats.json` и `meta/stats.json`.

## Sin/cos state embedding

Для joint angles в radians можно отдельно проверить `sin_cos_embedding_keys`, но не включать это до первого рабочего baseline. Такое изменение удваивает representation dimensions и должно быть отдельным experiment.

## Config validation

```bash
cd "$WORK_ROOT/repos/Isaac-GR00T"
uv run python - <<PY
import importlib.util
p = "$WORK_ROOT/configs/robosyn_cobotmagic_config.py"
spec = importlib.util.spec_from_file_location("robosyn_cfg", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print("config imported:", p)
PY
```

Затем loader smoke test должен взять один episode и вывести shapes всех modalities. Если config imports, но loader выдаёт flat/empty actions, gate не пройден.

## Augmentation safety

- Не применять horizontal flip, если одновременно не отражаются actions, camera semantics и left/right labels.
- Начинать со слабых crop/resize и color jitter.
- Сильные геометрические transformations легко ломают точную manipulation geometry.
