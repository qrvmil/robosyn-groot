# Sources and provenance

Дата подготовки runbook: 2026-08-12.

## Primary official sources

### NVIDIA Isaac GR00T N1.7

- Repository: https://github.com/NVIDIA/Isaac-GR00T
- Main README and installation: https://github.com/NVIDIA/Isaac-GR00T/blob/main/README.md
- Hardware recommendations: https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/hardware_recommendation.md
- Data preparation: https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md
- Modality configuration: https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_config.md
- Custom embodiment fine-tuning: https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/finetune_new_embodiment.md
- Fine-tune wrapper: https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/finetune.sh
- Fine-tune entrypoint: https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/experiment/launch_finetune.py
- Fine-tune dataclass: https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/configs/finetune_config.py
- Training defaults: https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/configs/training/training_config.py
- Stats generation: https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/data/stats.py

### RoboSynChallenge / EmbodiChain

- RoboSynChallenge repository: https://github.com/EDEM-AI/RoboSynChallenge
- Installation: https://github.com/EDEM-AI/RoboSynChallenge/blob/main/docs/getting_started/installation.md
- Downloaded datasets: https://github.com/EDEM-AI/RoboSynChallenge/blob/main/docs/tutorials/download_data.md
- Data collection: https://github.com/EDEM-AI/RoboSynChallenge/blob/main/docs/tutorials/collect_data.md
- LeRobot 3.0 → 2.1 conversion: https://github.com/EDEM-AI/RoboSynChallenge/blob/main/scripts/convert_lerobot3.0_to_2.1.py
- EmbodiChain repository: https://github.com/DexForce/EmbodiChain

### Vast.ai

- Instances overview: https://docs.vast.ai/guides/instances/overview
- Storage types: https://docs.vast.ai/guides/instances/storage/types
- Volumes: https://docs.vast.ai/guides/instances/storage/volumes
- Templates: https://docs.vast.ai/guides/instances/choosing/templates
- Instances FAQ / Docker-in-Docker and persistence: https://docs.vast.ai/guides/reference/faq/instances

## User-provided source

- `reference/vla-training-cookbook.md` — копия приложенного пользователем файла с практическим VLA training recipe.

## Important provenance rule

GitHub main branches меняются. Codex должен записать фактические `git rev-parse HEAD` и сохранить `--help` перед работой. Этот runbook описывает подтверждённую структуру на дату выше, но exact CLI current checkout всегда имеет приоритет.
