# Prepared Dataset

Status: **PASS**

- Source: `RoboSynChallenge/cobotmagic_Sim_click_bell` at revision `dd1f0de495cfec3240beaf2714f239168d5b3fae`.
- Prepared root: `data/prepared/cobotmagic_Sim_click_bell__groot_v1`.
- Raw integrity: all 4,006 source files passed SHA256 recheck; raw files retain no write bits.
- Prepared contents: 1,000 parquet files, 74,000 rows, 3,000 AV1 videos.
- Language: `task_index` copied to aligned `annotation.human.task_description`; every value resolves to exact task text `Click the bell`.
- Split: deterministic whole episodes, seed 17; 800 train and 200 validation with no overlap.
- Modality: 14-D state/action split as 6+1+6+1; three cameras; action horizon 13 at 25 FPS.
- Full prepared audit: all counts reconcile; every video is 640x480 AV1, 25 FPS, 74 frames, 2.96 s; zero probe/decode failures.
- Generated config import: PASS against pinned Isaac-GR00T environment.
- Prepared file manifest: 8,016 files hashed before stats generation.

Evidence:

- `configs/robosyn_cobotmagic_config.py`
- `data/manifests/cobotmagic_Sim_click_bell__groot_v1.audit.json`
- `data/manifests/cobotmagic_Sim_click_bell__groot_v1.sha256.json`
- `reports/raw_hash_recheck.txt`
