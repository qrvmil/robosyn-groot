# VLA Competition Recipe

Практический рецепт для команды, которая получила датасет и должна быстро обучить сильную VLA-policy.

## 0. Маршрут

```text
воспроизвести baseline → выбрать ближайший checkpoint → проверить данные → tiny overfit → первый run → rollout → sweep LR → sweep trainable scope → sweep chunk/data → ensemble лучших checkpoints
```

---

## 1. От чего инициализироваться

Приоритет такой:

1. официальный checkpoint и config соревнования;
2. pretrained VLA для того же робота и action space;
3. checkpoint для похожего робота и той же action representation;
4. общий pretrained VLA;
5. обучение с нуля — только если остальное невозможно.

Чем ближе checkpoint по камерам, action representation, частоте и типу задач, тем лучше. Размер модели вторичен: маленький подходящий checkpoint часто полезнее большого, но несовместимого.

Первым делом запустите официальный baseline **без изменений** и сохраните его результат.

---

## 2. Проверить датасет

Перед обучением:

- визуализировать `image + instruction + state + action`;
- проверить синхронизацию observation/action;
- проверить оси, units, frames и `absolute/delta`;
- проверить normalize → denormalize;
- разбить train/validation по эпизодам;
- посчитать долю пауз, clipping и неудачных эпизодов;
- посмотреть распределение задач, объектов и стартовых состояний.

Удаляйте только очевидно повреждённые данные. Recovery и осмысленные коррекции полезнее идеально гладких демонстраций.

---

## 3. Tiny overfit

Возьмите 1–4 коротких эпизода. Модель должна почти запомнить target actions.

Если не запоминает, проверьте:

- action masks;
- padding;
- normalization;
- порядок осей;
- learning rate;
- загрузку checkpoint;
- список trainable parameters.

Пока tiny overfit не работает, полный run запускать нельзя.

---

## 4. Первый рабочий run

Начальная конфигурация:

| Параметр | Стартовое значение |
|---|---|
| Initialization | ближайший pretrained checkpoint |
| Trainable modules | action head + LoRA/adapters в action expert |
| Vision-language backbone | frozen |
| Optimizer | AdamW |
| LR: новые heads/adapters | `1e-4` |
| LR: размороженные pretrained weights | `1e-5` |
| Weight decay | `0.01` |
| Gradient clipping | `1.0` |
| Schedule | cosine decay |
| Warmup | `5%` шагов |
| Precision | BF16, если поддерживается |
| Batch | максимально стабильный; увеличить через gradient accumulation |
| Action chunk | официальный default или примерно `0.5 s` движения |
| Augmentation | слабый crop/resize + слабый color jitter |

Это стартовые точки, а не универсальные оптимумы. Сначала меняйте только один параметр за эксперимент.

Не используйте horizontal flip, если вместе с изображением не отражаются actions и left/right semantics.

---

## 5. Что смотреть во время обучения

Минимальный dashboard:

- train и validation loss;
- loss по action dimensions и позиции внутри chunk;
- learning rate и gradient norm;
- target/predicted actions после денормализации;
- среднее, std и clipping предсказанных actions;
- validation success по каждой задаче;
- rollout success, latency и тип отказа.

Сохраняйте несколько checkpoints по ходу run. Минимальный validation loss не гарантирует лучший rollout.

---

## 6. Первый rollout

Проверьте:

- совпадают ли preprocessing в train и inference;
- совпадает ли control frequency;
- правильно ли исполняется action chunk;
- нет ли задержки, осцилляции или постоянного clipping;
- использует ли policy изображение и инструкцию;
- на какой стадии задачи она ломается.

Для длинной задачи считайте success по стадиям, а не только итоговый успех.

---

## 7. Порядок гиперпараметрических экспериментов

### Sweep 1. Learning rate

Запустите:

```text
0.3 × base LR
1.0 × base LR
3.0 × base LR
```

Это обычно самый дешёвый и полезный первый sweep.

### Sweep 2. Trainable scope

Сравните:

1. только action head;
2. head + LoRA/adapters;
3. head + последние блоки action expert;
4. весь action expert;
5. full fine-tuning.

Vision-language backbone размораживайте, только если model явно не извлекает нужную визуальную или языковую информацию.

### Sweep 3. Action chunk

Сравните примерно:

```text
0.5 × default
1.0 × default
2.0 × default
```

Короткий chunk лучше реагирует на ошибки, но требует частого inference. Длинный даёт плавность, но хуже корректируется в closed loop.

### Sweep 4. Data sampling

Попробуйте:

- уменьшить долю пауз;
- балансировать задачи;
- чаще выбирать редкие стадии;
- добавить recovery/correction episodes;
- смешать исходные и rollout-данные.

### Sweep 5. Augmentation

После появления рабочего baseline проверьте:

- crop/resize;
- brightness/contrast/color jitter;
- небольшие camera perturbations;
- paraphrases инструкций, если смысл точно сохраняется.

Сильная augmentation легко ломает точную геометрию manipulation.

---

## 8. Как реагировать на симптомы

| Симптом | Следующее действие |
|---|---|
| Loss не падает | вернуться к tiny overfit; проверить masks, LR и checkpoint |
| Train падает, validation растёт | меньше шагов, больше данных/augmentation, меньше trainable weights |
| Offline хорошо, rollout плохо | проверить timing, chunk execution, latency и controller |
| Policy почти не двигается | уменьшить долю пауз; проверить action normalization |
| Policy выдаёт среднее движение | балансировать данные; добавить контекст; проверить мультимодальность actions |
| Осцилляция | уменьшить chunk, проверить задержку, попробовать temporal ensemble |
| Дальний конец chunk плохой | уменьшить horizon или изменить weighting по horizon |
| Не использует instruction | shuffled-instruction test; баланс задач и формулировок |
| Не реагирует на объект | проверить crop, камеры и visual backbone |
| Ошибка только в gripper | отдельная метрика и отдельный вес gripper loss |
| LoRA упёрлась в плато | разморозить последние блоки action expert |
| Full fine-tuning хуже LoRA | переобучение или разрушение pretrained features |

---

## 9. Как выбрать submission

1. Зафиксировать локальный evaluation protocol и seeds.
2. Оценить несколько checkpoints каждого сильного run.
3. Отобрать 3–5 моделей по rollout success, а не по train loss.
4. Повторить evaluation на новых seeds и стартовых состояниях.
5. Если правила разрешают — проверить checkpoint averaging или ensemble.
6. Перед submission повторить inference из чистого окружения.

В таблице экспериментов храните:

```text
checkpoint | data version | trainable modules | LR | chunk | augmentation | offline metric | rollout success | failure modes
```

---

## 10. Приоритет времени

Если времени мало:

1. воспроизвести baseline;
2. выбрать близкий pretrained checkpoint;
3. добиться tiny overfit;
4. обучить head + LoRA;
5. сделать LR sweep;
6. проверить 3–5 checkpoints в rollout;
7. улучшить sampling и recovery data;
8. только потом размораживать больше модели и менять loss.

Экзотические objectives, большие архитектурные изменения и обучение с нуля оставляйте на конец: они дороже и труднее диагностируются.
