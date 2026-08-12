# 12. Backup, handoff и завершение Vast instance

## 1. Что считать критичными артефактами

```text
configs/
tools/
reports/
runs/*/command.sh
runs/*/stdout.log
runs/*/environment.md
runs/*/dataset_manifest.json
runs/*/git_state.txt
runs/*/evaluation/
selected checkpoints
prepared dataset metadata + conversion script
```

Raw datasets можно повторно скачать, если зафиксированы repo/revision/checksum. Уникальные generated datasets нужно выгружать целиком.

## 2. Финальный manifest run

Для каждого strong run:

```bash
RUN_DIR=<path>
{
  date -Is
  nvidia-smi
  git -C "$WORK_ROOT/repos/Isaac-GR00T" rev-parse HEAD
  git -C "$WORK_ROOT/repos/Isaac-GR00T" status --porcelain=v1
  sha256sum "$RUN_DIR/command.sh"
} > "$RUN_DIR/environment.md"
```

Добавить:

- data version и SHA manifest;
- modality config checksum;
- base model ID/revision;
- exact checkpoint path;
- action/model/execution horizons;
- metrics и seeds;
- known failure modes.

## 3. Упаковать handoff

```bash
cd "$WORK_ROOT"
tar --zstd -cf backups/robosyn_groot_handoff_$(date +%Y%m%d_%H%M).tar.zst \
  configs tools reports \
  runs/*/command.sh runs/*/stdout.log runs/*/environment.md \
  runs/*/evaluation
```

Checkpoints лучше выгружать отдельно из-за размера.

## 4. External backup

До `Destroy` обязательно одно из:

- Hugging Face private model/dataset repo;
- S3/Backblaze/rclone;
- SCP/rsync на лабораторный сервер;
- Vast local volume, если он действительно создан и прикреплён.

Проверить backup чтением/скачиванием хотя бы одного файла, а не только exit code upload command.

## 5. Final report

`reports/FINAL_REPORT.md` должен содержать:

1. hardware/software versions;
2. source commits;
3. dataset cards и conversion summary;
4. tiny overfit evidence;
5. таблицу runs;
6. open-loop/rollout results;
7. выбранные checkpoints;
8. failure analysis;
9. точную команду воспроизведения;
10. location/checksum backups;
11. что осталось не проверено.

## 6. Перед остановкой/уничтожением

```bash
sync
df -h "$WORK_ROOT"
find "$WORK_ROOT/backups" -maxdepth 2 -type f -printf '%s %p\n' | sort -n | tail
```

Stop сохраняет container storage, но billing за storage продолжается. Destroy удаляет container storage безвозвратно. Local volume переживает destroy, но привязан к физическому host.
