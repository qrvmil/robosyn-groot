# 11. Troubleshooting

## `GatedRepoError` / 401 для Cosmos-Reason2-2B

Причина: нет принятого access или token не авторизован.

```bash
uv run hf auth whoami
```

Проверить доступ к `nvidia/Cosmos-Reason2-2B`. Не менять model code.

## `torchcodec` не загружается

Проверить FFmpeg major version:

```bash
ffmpeg -version | head -1
ldconfig -p | grep -E 'libavcodec|libavformat' | head
```

GR00T-pinned torchcodec поддерживает FFmpeg 4–7, не 8.

## `CUDA_HOME is unset`

```bash
export CUDA_HOME=/usr/local/cuda
bash scripts/deployment/dgpu/install_deps.sh
```

Проверить `nvcc --version` и существование `$CUDA_HOME/bin/nvcc`.

## OOM на A100 80 GB

1. `nvidia-smi` — нет ли чужого process;
2. LLM/visual должны быть frozen для baseline;
3. batch 32 → 16 → 8;
4. gradient accumulation для effective batch;
5. не включён ли full fine-tuning;
6. записать peak memory и exact command.

Default projector+diffusion training должен укладываться заметно ниже 80 GB; OOM при batch 1–4 указывает на неправильный scope или leak.

## Process killed без Python traceback

```bash
dmesg -T | tail -100
free -h
df -h
```

- host OOM: уменьшить `num_shards_per_epoch`, workers или preload;
- disk full: очистить cache/checkpoints безопасно;
- provider reset: проверить Vast logs/reliability.

## Flat/constant actions

Проверить:

- modality keys и ordering;
- annotation column;
- action mask/padding;
- stats/relative_stats;
- relative conversion;
- checkpoint loaded;
- trainable params;
- pause-heavy data.

## `IndexError` после смены action horizon

Старый `relative_stats.json` имеет shape прошлого horizon. Удалить stats только в **prepared version**, затем пересчитать через `gr00t/data/stats.py`.

## NaN loss или огромные denormalized actions

- NaN/Inf в parquet;
- неверные units;
- q01/q99 collapse;
- constant dimension std=0;
- relative action applied to already-delta action;
- gripper scale существенно отличается от arms.

## Train command принимает не все flags из docs

Считать `uv run python gr00t/experiment/launch_finetune.py --help` источником истины. Не передавать неизвестные flags. Записать mismatch. Validation можно делать отдельным open-loop protocol.

## RoboSyn: `No module named embodichain_tasks`

RoboSyn environment отдельный от GR00T. Проверить:

```bash
source "$WORK_ROOT/.venvs/robosyn/bin/activate"
cd "$WORK_ROOT/repos/EmbodiChain"
find . -maxdepth 2 -type d -name embodichain_tasks
uv pip install -e ./embodichain_tasks
python -c 'from embodichain_tasks.tableware.base_agent_env import BaseAgentEnv; print("OK")'
```

Если folder отсутствует в выбранном tag, нужен совместимый EmbodiChain commit; `PYTHONPATH` не заменяет отсутствующий source.

## RoboSyn native/Vulkan crash на VNC

Обычный VNC X display может не поддерживать нужный GPU/Vulkan context. Для first eval использовать headless. GUI — только после отдельной проверки Vulkan/GL и не должен блокировать model pipeline.

## `uv run` каждый раз пишет про flash-attn install

Для URL-pinned wheel uv может повторно валидировать cache. Это не обязательно rebuild. Не удалять source pin из `pyproject.toml` в baseline checkout без необходимости.

## Vast disk нельзя расширить

Container disk size фиксирован при создании. Если место заканчивается, остановить training до corruption checkpoint, выгрузить артефакты и перенести workload на новый instance/volume.
