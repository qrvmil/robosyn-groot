# 00. Vast.ai instance, storage и базовая организация

## 1. Снять фактические характеристики

```bash
set -euo pipefail
uname -a
cat /etc/os-release
nvidia-smi
nvcc --version || true
python3 --version || true
free -h
df -h
lsblk -f || true
mount | sed -n '1,160p'
```

Сохранить вывод в `$WORK_ROOT/reports/instance_report.txt` после выбора `$WORK_ROOT`.

## 2. Выбрать рабочий диск

Vast container storage сохраняется при stop, но удаляется при destroy. Local volume, если он был прикреплён, обычно монтируется в `/data` и переживает уничтожение instance только на том же физическом host.

Использовать:

```bash
if mountpoint -q /data && [ -w /data ]; then
  export WORK_ROOT=/data/robosyn-groot
else
  export WORK_ROOT=/workspace/robosyn-groot
fi

mkdir -p "$WORK_ROOT"/{repos,data/{raw,prepared,tiny,manifests},configs,tools,runs,reports,cache,backups}
printf 'WORK_ROOT=%s\n' "$WORK_ROOT"
df -h "$WORK_ROOT"
```

Не использовать `/tmp` для datasets/checkpoints.

## 3. Постоянный env-файл

Создать `$WORK_ROOT/env.sh`:

```bash
cat > "$WORK_ROOT/env.sh" <<EOF
export WORK_ROOT="$WORK_ROOT"
export HF_HOME="$WORK_ROOT/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$WORK_ROOT/cache/huggingface/hub"
export TORCH_HOME="$WORK_ROOT/cache/torch"
export UV_CACHE_DIR="$WORK_ROOT/cache/uv"
export WANDB_DIR="$WORK_ROOT/runs/wandb"
export CUDA_HOME="/usr/local/cuda"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
EOF

source "$WORK_ROOT/env.sh"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$UV_CACHE_DIR" "$WANDB_DIR"
```

Не записывать секреты в `env.sh`. `HF_TOKEN` и `WANDB_API_KEY` должны приходить из Vast secrets или экспортироваться только в текущую shell-сессию.

## 4. Системные пакеты

GR00T dGPU workflow требует Python 3.12 через `uv`, Git submodules, Git LFS и FFmpeg. `torchcodec==0.8.0` поддерживает FFmpeg 4–7.

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git git-lfs curl ca-certificates build-essential pkg-config \
  ffmpeg jq tmux htop rsync unzip zip tree aria2

git lfs install
ffmpeg -version | head -1
```

Если FFmpeg major version равен 8, не продолжать: установить runtime `<8` и убедиться, что именно его shared libraries видит Python.

## 5. Установить uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

Добавить `$HOME/.local/bin` в `PATH` текущей SSH-сессии и `~/.bashrc`, если Vast template этого не сделал.

## 6. Работать через tmux

```bash
tmux new -s groot
```

Минимум два окна:

- `train` — training process;
- `monitor` — `watch -n 1 nvidia-smi`, `htop`, `df -h`.

## 7. Важное ограничение Vast

Vast instance уже является Docker container. Docker-in-Docker не поддерживается и не нужен. Поэтому официальный RoboSyn Docker image здесь не запускать: использовать local installation в отдельном Python 3.11 окружении либо вообще не ставить simulator на training-only instance.

## 8. Storage budget

Рекомендуемое распределение для диска около 900 GB:

```text
120–180 GB  Hugging Face/model/uv caches
100–300 GB  RoboSyn raw + prepared datasets
100–250 GB  checkpoints и optimizer states
20–50 GB    logs, plots, videos
остаток      safety margin
```

Перед каждым большим run:

```bash
df -h "$WORK_ROOT"
du -sh "$WORK_ROOT"/* 2>/dev/null | sort -h
```

Не начинать run, если свободного места меньше ожидаемого размера всех checkpoints × 2 плюс 50 GB запаса.
