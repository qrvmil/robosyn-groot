# GR00T Fine-Tuning Readiness

**Result:** READY

Run: `click_bell_sim_baseline_2k`

| Check | Result | Detail |
|---|---:|---|
| `launch_manifest` | PASS | /workspace/challenge/robosyn-groot/runs/click_bell_sim_baseline_2k/launch_manifest.json |
| `tiny_overfit` | PASS | 500-step training, reload, and open-loop evidence |
| `full_startup_smoke` | PASS | batch 32, full dataset, Cosmos metadata, and one optimizer step |
| `config_checksum` | PASS | expected=a92e1d5a1cd47c1a93e60e22179976b46b7c1a313bdea1078fbe0bff724b29f8 |
| `provenance_checksums` | PASS | matched |
| `command_executable` | PASS | /workspace/challenge/robosyn-groot/runs/click_bell_sim_baseline_2k/command.sh |
| `cli_flags` | PASS | all flags in pinned help |
| `launcher_environment` | PASS | Hub metadata lookup is enabled |
| `launch_scope` | PASS | missing=[]; unsafe=[] |
| `full_run_unlaunched` | PASS | /workspace/challenge/robosyn-groot/runs/click_bell_sim_baseline_2k/checkpoints |
| `hardware` | PASS | {"bf16": true, "cuda": true, "ffmpeg": true, "gpu": "NVIDIA A100 80GB PCIe", "torch": "2.9.0+cu128", "torch_cuda": "12.8"} |
| `ffmpeg` | PASS | /usr/bin/ffmpeg |
| `disk_reserve` | PASS | free_bytes=262807097344; required=161061273600 |
| `model_access` | PASS | revision=2fc962b973bccdd5d8ce4f67cc63b264d6886495; shards=2 |
| `groot_revision` | PASS | 376ba890cff8c9de64d71d982772a9c36185fdd7 |
| `official_smoke` | PASS | SO100 train+reload |
| `source_revision` | PASS | dd1f0de495cfec3240beaf2714f239168d5b3fae |
| `raw_hashes` | PASS | verified_files=4006 |
| `audit_reconciliation` | PASS | episodes=1000; frames=74000 |
| `visual_review` | PASS | videos=5; stills=15 |
| `prepared_dataset` | PASS | /workspace/challenge/robosyn-groot/data/prepared/cobotmagic_Sim_click_bell__groot_v1 |
| `loader_stats_roundtrip` | PASS | roundtrip_max=5.551115123125783e-17 |
| `semantic_horizons` | PASS | {"action_delta_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "action_horizon_seconds": 0.52, "action_horizon_steps": 13, "fps": 25, "state_delta_indices": [0], "video_and_parquet_aligned": true, "video_delta_indices": [0]} |

Full training was not launched by this verifier.
