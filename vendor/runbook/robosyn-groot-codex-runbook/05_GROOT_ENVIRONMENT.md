# 05. Установка GR00T N1.7 на Vast A100

## 1. Предусловия

```bash
source "$WORK_ROOT/env.sh"
nvidia-smi
ffmpeg -version | head -1
```

A100 80 GB подходит для standard fine-tuning projector + diffusion action head. Размораживание VLM (`tune_llm`/`tune_visual`) требует заметно больше VRAM и должно идти отдельным experiment с уменьшенным batch.

## 2. Установка

```bash
cd "$WORK_ROOT/repos/Isaac-GR00T"
uv sync --python 3.12
uv run python -c "import gr00t; print('GR00T installed successfully')"
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```

Если `CUDA_HOME is unset`:

```bash
export CUDA_HOME=/usr/local/cuda
bash scripts/deployment/dgpu/install_deps.sh
```

Не запускать platform scripts для Thor/Spark/Orin на A100 dGPU.

## 3. Hugging Face access

GR00T N1.7 загружает gated backbone `nvidia/Cosmos-Reason2-2B`, даже если base checkpoint — `nvidia/GR00T-N1.7-3B`. Пользователь должен заранее получить доступ к gated model.

```bash
# HF_TOKEN должен быть задан безопасно, без echo.
uv run hf auth login --token "$HF_TOKEN"
uv run hf auth whoami
```

Smoke download/load делать до большого run. Ошибка `GatedRepoError`/401 — не training bug.

## 4. Проверить CLI фактического commit

```bash
uv run python gr00t/experiment/launch_finetune.py --help \
  | tee "$WORK_ROOT/reports/launch_finetune_help.txt"

uv run python gr00t/eval/open_loop_eval.py --help \
  | tee "$WORK_ROOT/reports/open_loop_help.txt"
```

Не использовать flag, которого нет в сохранённом help.

## 5. Официальный smoke test до RoboSyn

Перед custom dataset желательно воспроизвести shipped `demo_data/cube_to_bowl_5`/SO100 pipeline с коротким run. Это отделяет installation/model-access bugs от RoboSyn conversion bugs.

Минимально:

```bash
cd "$WORK_ROOT/repos/Isaac-GR00T"
CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path ./demo_data/cube_to_bowl_5 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/SO100/so100_config.py \
  --num-gpus 1 \
  --output-dir "$WORK_ROOT/runs/so100_smoke" \
  --save-total-limit 2 \
  --save-steps 10 \
  --max-steps 10 \
  --global-batch-size 4 \
  --dataloader-num-workers 2
```

Gate проходит, если модель/processor/data loader созданы, сделан optimizer step, loss finite и checkpoint сохраняется.

## 6. Freeze environment

```bash
cd "$WORK_ROOT/repos/Isaac-GR00T"
uv pip freeze > "$WORK_ROOT/reports/groot_python_freeze.txt"
uv lock --check
```

Не ставить случайные packages в `.venv` без записи причины. Для utility scripts предпочитать зависимости, уже присутствующие в GR00T environment.
