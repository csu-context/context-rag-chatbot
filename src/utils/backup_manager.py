import hashlib
import logging
import platform
import shutil
import sqlite3
import sys
import tarfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.utils.paths import BACKUP_DIR, VECTOR_DB_DIR, ensure_directories  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_ROTATION_LIMIT = 5
DEFAULT_QUIET_PERIOD_SECONDS = 2.0
DEFAULT_STABILITY_TIMEOUT_SECONDS = 30.0
LOCK_FILE_NAME = ".chromadb_backup.lock"
BACKUP_PATTERN = "chromadb_backup_*.tar.gz"
HASH_SUFFIX = ".sha256"


@contextmanager
def _operation_lock(lock_file: Path):
    if lock_file.exists():
        logger.warning("Another backup/restore operation is in progress. Lock file exists: %s", lock_file)
        raise RuntimeError(f"Lock file exists: {lock_file}")
    try:
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_file.touch(exist_ok=False)
        yield
    except FileExistsError as e:
        logger.warning("Another process acquired the lock concurrently: %s", lock_file)
        raise RuntimeError(f"Lock file exists: {lock_file}") from e
    finally:
        lock_file.unlink(missing_ok=True)


def _get_db_mtime(vector_db_dir: Path) -> float | None:
    sqlite_file = vector_db_dir / "chroma.sqlite3"
    if sqlite_file.exists():
        return sqlite_file.stat().st_mtime
    return None


def _wait_for_stable_snapshot(vector_db_dir: Path, quiet_period_seconds: float, timeout_seconds: float) -> bool:
    start_time = time.time()
    last_mtime = _get_db_mtime(vector_db_dir)

    while True:
        if time.time() - start_time > timeout_seconds:
            logger.warning("Timeout waiting for ChromaDB stability. Backing up anyway.")
            return False

        time.sleep(1)
        current_mtime = _get_db_mtime(vector_db_dir)

        if current_mtime is not None and current_mtime == last_mtime:
            time_since_last_mod = time.time() - current_mtime
            if time_since_last_mod >= quiet_period_seconds:
                return True
        last_mtime = current_mtime


def _generate_hash_sidecar(backup_path: Path) -> Path:
    hash_path = backup_path.with_name(f"{backup_path.name}{HASH_SUFFIX}")
    sha256 = hashlib.sha256()
    with open(backup_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    hash_path.write_text(sha256.hexdigest(), encoding="utf-8")
    return hash_path


def verify_hash_sidecar(backup_path: Path) -> bool:
    hash_path = backup_path.with_name(f"{backup_path.name}{HASH_SUFFIX}")
    if not hash_path.exists():
        logger.warning("Hash sidecar not found for %s, skipping verification.", backup_path)
        return True

    expected_hash = hash_path.read_text(encoding="utf-8").strip()
    sha256 = hashlib.sha256()
    with open(backup_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    actual_hash = sha256.hexdigest()

    if expected_hash != actual_hash:
        logger.error("Hash verification failed for %s", backup_path)
        return False
    return True


def _exclude_runtime_files(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    if tarinfo.name.endswith(LOCK_FILE_NAME) or tarinfo.name.endswith((".wal", "-wal", "-shm")):
        return None
    return tarinfo


def backup_chromadb(
    rotation_limit: int = DEFAULT_ROTATION_LIMIT,
    vector_db_dir: Path = VECTOR_DB_DIR,
    backup_dir: Path = BACKUP_DIR,
    quiet_period_seconds: float = DEFAULT_QUIET_PERIOD_SECONDS,
    stability_timeout_seconds: float = DEFAULT_STABILITY_TIMEOUT_SECONDS,
) -> bool:
    ensure_directories()

    if not vector_db_dir.exists():
        logger.error("Backup failed: vector DB directory does not exist: %s", vector_db_dir)
        return False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"chromadb_backup_{timestamp}.tar.gz"
    lock_file = vector_db_dir / LOCK_FILE_NAME

    try:
        with _operation_lock(lock_file):
            logger.info("Waiting for a stable ChromaDB snapshot: %s", vector_db_dir)
            _wait_for_stable_snapshot(
                vector_db_dir,
                quiet_period_seconds=quiet_period_seconds,
                timeout_seconds=stability_timeout_seconds,
            )

            logger.info("Creating ChromaDB backup: %s -> %s", vector_db_dir, backup_path)
            with tarfile.open(backup_path, "w:gz") as tar:
                tar.add(vector_db_dir, arcname=vector_db_dir.name, filter=_exclude_runtime_files)

            _generate_hash_sidecar(backup_path)
            logger.info("Successfully created backup and hash sidecar.")
            rotate_backups(rotation_limit, backup_dir)
            return True
    except Exception as e:
        logger.error("Backup process failed: %s", e)
        if backup_path.exists():
            backup_path.unlink(missing_ok=True)
            backup_path.with_name(f"{backup_path.name}{HASH_SUFFIX}").unlink(missing_ok=True)
        return False


def rotate_backups(limit: int, backup_dir: Path = BACKUP_DIR) -> None:
    backups = sorted(backup_dir.glob(BACKUP_PATTERN), key=lambda path: path.stat().st_mtime)
    if len(backups) <= limit:
        return

    for archive_path in backups[:-limit]:
        archive_path.unlink(missing_ok=True)
        archive_path.with_name(f"{archive_path.name}{HASH_SUFFIX}").unlink(missing_ok=True)
        logger.info("Removed old backup: %s", archive_path.name)


def diagnose_db(vector_db_dir: Path = VECTOR_DB_DIR) -> bool:
    sqlite_file = vector_db_dir / "chroma.sqlite3"
    if not sqlite_file.exists():
        logger.error("Diagnostics failed: DB file not found: %s", sqlite_file)
        return False

    try:
        conn = sqlite3.connect(f"file:{sqlite_file}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check;")
        result = cursor.fetchone()
        conn.close()

        if result and result[0] == "ok":
            logger.info("DB Diagnostics passed: Integrity check OK.")
            return True
        else:
            logger.error("DB Diagnostics failed: Integrity check returned: %s", result)
            return False
    except sqlite3.Error as e:
        logger.error("DB Diagnostics failed: SQLite error: %s", e)
        return False


def _find_backup(backup_file: str | None, backup_dir: Path) -> Path | None:
    if backup_file:
        return backup_dir / backup_file
    backups = sorted(backup_dir.glob(BACKUP_PATTERN), key=lambda path: path.stat().st_mtime, reverse=True)
    return backups[0] if backups else None


def _move_existing_db(vector_db_dir: Path) -> Path | None:
    if not vector_db_dir.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_existing = vector_db_dir.parent / f"{vector_db_dir.name}_temp_{timestamp}"
    shutil.move(vector_db_dir, temp_existing)
    logger.info("Moved existing DB temporarily to: %s", temp_existing)
    return temp_existing


def _extract_archive(backup_path: Path, target_dir: Path) -> None:
    with tarfile.open(backup_path, "r:gz") as tar:
        # Prevent path traversal vulnerabilities
        for member in tar.getmembers():
            if not _is_safe_tar_member(member, target_dir):
                raise ValueError(f"Unsafe path in tar archive: {member.name}")
        tar.extractall(path=target_dir)


def _is_safe_tar_member(member: tarfile.TarInfo, target_dir: Path) -> bool:
    if platform.system() == "Windows":
        return True
    try:
        member_path = (target_dir / member.name).resolve()
        target_dir_resolved = target_dir.resolve()
        return member_path.is_relative_to(target_dir_resolved)
    except Exception:
        return False


def _restore_previous_db(vector_db_dir: Path, temp_existing: Path | None) -> None:
    if vector_db_dir.exists():
        shutil.rmtree(vector_db_dir)
        logger.info("Removed invalid restored DB: %s", vector_db_dir)
    if temp_existing and temp_existing.exists():
        shutil.move(temp_existing, vector_db_dir)
        logger.info("Restored original DB from: %s", temp_existing)


def _remove_previous_db(temp_existing: Path | None) -> None:
    if temp_existing and temp_existing.exists():
        shutil.rmtree(temp_existing)
        logger.info("Removed temporary original DB: %s", temp_existing)


def restore_chromadb(
    backup_file: str | None = None,
    vector_db_dir: Path = VECTOR_DB_DIR,
    backup_dir: Path = BACKUP_DIR,
) -> bool:
    ensure_directories()
    backup_path = _find_backup(backup_file, backup_dir)
    if backup_path is None:
        logger.error("Restore failed: no backup archives found in %s", backup_dir)
        return False
    if not backup_path.exists():
        logger.error("Restore failed: backup archive does not exist: %s", backup_path)
        return False
    if not verify_hash_sidecar(backup_path):
        return False

    lock_file = vector_db_dir / LOCK_FILE_NAME if vector_db_dir.exists() else vector_db_dir.parent / LOCK_FILE_NAME
    temp_existing: Path | None = None

    try:
        with _operation_lock(lock_file):
            logger.info("Restoring ChromaDB from backup: %s", backup_path.name)
            temp_existing = _move_existing_db(vector_db_dir)
            _extract_archive(backup_path, vector_db_dir.parent)

            if not diagnose_db(vector_db_dir):
                _restore_previous_db(vector_db_dir, temp_existing)
                return False

            _remove_previous_db(temp_existing)
            logger.info("Successfully restored ChromaDB from backup: %s", backup_path.name)
            return True

    except Exception as e:
        logger.error("Restore process failed: %s", e)
        _restore_previous_db(vector_db_dir, temp_existing)
        return False
