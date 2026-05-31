# ChromaDB Backup Automation

This project keeps backup and restore business logic in `src/utils/backup_manager.py`.
The files under `scripts/` are thin CLI entry points.

## Manual Commands

Create a backup:

```bash
python scripts/backup_db.py
```

Restore the newest backup:

```bash
python scripts/restore_db.py
```

Restore a specific backup:

```bash
python scripts/restore_db.py --file chromadb_backup_20260531_030000.tar.gz
```

## Consistency Checks

Before creating a `tar.gz` archive, the backup manager:

1. Creates an operation lock in the vector DB directory.
2. Waits until the vector DB file manifest is stable for the configured quiet period.
3. Compares the manifest again after archive creation.
4. Deletes the incomplete archive if files changed during backup.
5. Writes a `.sha256` sidecar for the archive.

Restore verifies the `.sha256` sidecar when it exists, extracts the archive, and runs
diagnostics that exercise both collection count and a real query path.

For production backups, schedule the job during a maintenance window or another period
where ingestion and bulk writes are paused. If files keep changing, the backup fails
instead of producing a potentially inconsistent archive.

## Crontab Example

Run a daily backup at 03:00 and keep logs in `logs/backup_cron.log`:

```cron
0 3 * * * cd /path/to/context-rag-chatbot && /path/to/venv/bin/python scripts/backup_db.py >> logs/backup_cron.log 2>&1
```

On Linux, use `flock` to avoid overlapping scheduler executions:

```cron
0 3 * * * cd /path/to/context-rag-chatbot && flock -n /tmp/context-rag-chatbot-backup.lock /path/to/venv/bin/python scripts/backup_db.py >> logs/backup_cron.log 2>&1
```

Windows Task Scheduler can run the same entry point with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-Location 'C:\path\to\context-rag-chatbot'; .\.venv\Scripts\python.exe scripts\backup_db.py"
```
