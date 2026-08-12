# Modality Config Review

## Dataset evidence

- `meta/info.json` path/checksum:
- Representative parquet path:
- RoboSyn action config path/commit:
- Measured action/state shapes:

## Video mapping

| GR00T key | Original LeRobot key | Evidence | Included? |
|---|---|---|---|
| | | | |

## State mapping

| GR00T key | Slice | Semantics | Units/frame | Evidence |
|---|---|---|---|---|
| | | | | |

## Action mapping

| GR00T key | Slice | Stored semantics | GR00T rep | State reference | Evidence |
|---|---|---|---|---|---|
| | | | | | |

## Language mapping

- Parquet column:
- `modality.json` key:
- Python modality key:
- `tasks.jsonl` sample:

## Temporal config

- Dataset FPS:
- Model action horizon:
- Horizon duration:
- Rollout execution horizon:

## Safety checks

- [ ] No double relative conversion.
- [ ] Gripper semantics checked visually.
- [ ] Left/right cameras and arms not swapped.
- [ ] All keys exist in `modality.json`.
- [ ] `action_configs` order equals `modality_keys` order.
- [ ] Stats regenerated after the final horizon/config.
- [ ] Tiny overfit passed with this exact checksum.
