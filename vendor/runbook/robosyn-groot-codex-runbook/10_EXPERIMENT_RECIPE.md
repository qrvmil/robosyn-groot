# 10. Порядок экспериментов

Этот файл адаптирует приложенный `reference/vla-training-cookbook.md` к фактическому GR00T N1.7 API.

## Общий маршрут

```text
официальный environment smoke
→ RoboSyn data audit
→ tiny overfit
→ official-scope baseline
→ open-loop/rollout
→ LR sweep
→ trainable scope sweep
→ chunk sweep
→ data sampling
→ augmentation
→ выбор нескольких checkpoints
```

## 1. Initialization priority

1. официальный GR00T N1.7 base checkpoint;
2. существующий checkpoint для того же CobotMagic embodiment/action space, если он реально доступен и лицензия позволяет;
3. checkpoint близкого bimanual joint-space robot;
4. общий `nvidia/GR00T-N1.7-3B`;
5. с нуля — только как крайний ablation.

Размер модели вторичен относительно camera/action/frequency compatibility.

## 2. Первый scope

Cookbook предлагает action head + LoRA/adapters при frozen VLM. В official GR00T N1.7 fine-tune API документированы другие knobs:

```text
tune_diffusion_model
tune_projector
tune_llm
tune_visual
```

Поэтому ближайший поддерживаемый baseline:

```text
diffusion action model + projector trainable
LLM + visual frozen
```

Не объявлять это LoRA. Отдельную LoRA реализацию добавлять только после baseline, если текущий official repo действительно содержит поддерживаемый путь или созданный patch проходит tests.

## 3. Sweep 1 — learning rate

При base LR `1e-4`:

```text
3e-5
1e-4
3e-4
```

Остальные параметры фиксированы. Сравнивать несколько checkpoints каждого run.

## 4. Sweep 2 — trainable scope

В доступном official API:

1. diffusion model only (`tune_projector=false`);
2. projector + diffusion model — baseline;
3. baseline + visual;
4. baseline + LLM;
5. LLM + visual + projector + diffusion — full scope, только если VRAM/overfitting оправданы.

На 1× A100 80 GB для пунктов 3–5 снижать batch и тщательно следить за VRAM. Не размораживать backbone, пока diagnostics не показывают, что frozen features недостаточны.

## 5. Weight decay

- Official GR00T example/default: `1e-5`.
- VLA cookbook starting point: `0.01`.

Использовать `1e-5` в воспроизводимом baseline, затем отдельный run `0.01`. Не смешивать это с LR/scope change.

## 6. Sweep 3 — action chunk

Если baseline horizon 16:

```text
8
16
32
```

Каждый раз:

- изменить `delta_indices` в modality config;
- создать новую prepared/data-config version;
- пересчитать stats/relative stats;
- проверить tiny overfit;
- отдельно выбрать execution horizon в rollout.

Короткий chunk реактивнее; длинный плавнее, но хуже исправляет ошибку closed-loop.

## 7. Sweep 4 — data sampling

Проверить:

- sim-only;
- real-only;
- sim → real staged fine-tuning;
- joint sim+real;
- уменьшение доли пауз;
- баланс задач/стадий;
- oversampling rare stages;
- recovery/correction episodes;
- rollout data добавлять только с provenance и quality filters.

Удалять только явно повреждённые trajectories. Осмысленные corrections/recovery могут быть полезнее идеально гладких demonstrations.

## 8. Sweep 5 — augmentation

После рабочего baseline:

- слабый crop/resize;
- слабый brightness/contrast/color jitter;
- небольшие camera perturbations;
- paraphrases instruction только при неизменном смысле.

Запрещён horizontal flip без одновременного зеркалирования actions и left/right semantics.

## 9. Минимальный dashboard

- train loss;
- held-out open-loop error;
- action error per dimension and chunk position;
- learning rate;
- gradient norm;
- prediction mean/std/range/clipping after denormalization;
- gripper metric;
- rollout success per task/stage;
- latency and inference frequency;
- failure taxonomy.

## 10. Симптом → следующее действие

| Симптом | Следующее действие |
|---|---|
| Loss не падает | вернуться к tiny overfit; masks, data load, LR, checkpoint, trainable params |
| Train улучшается, held-out хуже | меньше steps/scope, больше data/weak augmentation |
| Offline хорошо, rollout плохо | preprocessing, frequency, execution horizon, latency, controller |
| Policy почти не двигается | pause rate, normalization, gripper/action mask |
| Среднее движение | data balance, multimodality, instruction/context |
| Осцилляция | уменьшить execution horizon, latency check, temporal ensemble как отдельный experiment |
| Конец chunk плохой | уменьшить model horizon или изменить horizon weighting отдельным patch |
| Instruction игнорируется | shuffled-instruction test, task balance, annotations |
| Object не используется | crop/cameras/visual features; только затем tune visual |
| Ошибка только gripper | отдельная metric, action slice/config, loss weighting отдельным patch |
| Full scope хуже default | overfitting/destruction of pretrained features |
| NaN/огромный MSE | stats, units, relative conversion, outliers/clipping |

## 11. Выбор финальной модели

1. Зафиксировать protocol/seeds.
2. Оценить несколько checkpoints каждого сильного run.
3. Отобрать 3–5 по rollout success.
4. Повторить на новых seeds/start states.
5. Проверить averaging/ensemble только если это допускает цель и latency.
6. Повторить inference из чистого process/environment.

Экспериментальная таблица:

```text
run/checkpoint | data version | commit | scope | LR | WD | horizon | augmentation | open-loop | rollout | latency | failure modes
```
