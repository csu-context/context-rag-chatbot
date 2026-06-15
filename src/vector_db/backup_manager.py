import hashlib
import json
import logging
import os
import platform
import shutil
import sqlite3
import tarfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from src.common.config import settings
from src.utils.paths import BACKUP_DIR, VECTOR_DB_DIR, ensure_directories

logger = logging.getLogger(__name__)

DEFAULT_ROTATION_LIMIT = 5
DEFAULT_QUIET_PERIOD_SECONDS = 2.0
DEFAULT_STABILITY_TIMEOUT_SECONDS = 30.0
LOCK_FILE_NAME = ".chromadb_backup.lock"
BACKUP_PATTERN = "chromadb_backup_*.tar.gz"
HASH_SUFFIX = ".sha256"
STATUS_FILE_NAME = "backup_status.json"


def _update_status(status: str, error: str | None = None, target: str | None = None) -> None:
    try:
        status_file = BACKUP_DIR / STATUS_FILE_NAME
        status_file.parent.mkdir(parents=True, exist_ok=True)
        data = {"status": status, "last_update": time.time(), "error": error, "target": target}
        status_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.error("Failed to write backup status: %s", e)


@contextmanager
def _operation_lock(lock_file: Path):
    if lock_file.exists():
        logger.warning("다른 백업 또는 복원 작업이 진행 중입니다. 잠금 파일이 존재합니다: %s", lock_file)
        raise RuntimeError(f"잠금 파일이 존재합니다: {lock_file}")
    try:
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_file.touch(exist_ok=False)
        yield
    except FileExistsError as e:
        logger.warning("다른 프로세스가 동시에 잠금을 획득했습니다: %s", lock_file)
        raise RuntimeError(f"잠금 파일이 존재합니다: {lock_file}") from e
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
            logger.warning("ChromaDB 안정화 대기 시간 초과. 백업을 계속 진행합니다.")
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
        logger.warning("%s에 대한 해시 검증용 파일을 찾을 수 없어 검증을 건너뜁니다.", backup_path)
        return True

    expected_hash = hash_path.read_text(encoding="utf-8").strip()
    sha256 = hashlib.sha256()
    with open(backup_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    actual_hash = sha256.hexdigest()

    if expected_hash != actual_hash:
        logger.error("%s에 대한 해시 검증에 실패했습니다.", backup_path)
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
    _update_status("running_backup")

    if not vector_db_dir.exists():
        logger.error("백업 실패: 벡터 DB 디렉터리가 존재하지 않습니다: %s", vector_db_dir)
        _update_status("failed", error=f"벡터 DB 디렉터리가 존재하지 않습니다: {vector_db_dir}")
        return False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"chromadb_backup_{timestamp}.tar.gz"
    lock_file = vector_db_dir / LOCK_FILE_NAME

    try:
        with _operation_lock(lock_file):
            logger.info("안정적인 ChromaDB 스냅샷 생성 대기 중: %s", vector_db_dir)
            _wait_for_stable_snapshot(
                vector_db_dir,
                quiet_period_seconds=quiet_period_seconds,
                timeout_seconds=stability_timeout_seconds,
            )

            logger.info("ChromaDB 백업 작성 중: %s -> %s", vector_db_dir, backup_path)
            with tarfile.open(backup_path, "w:gz") as tar:
                tar.add(vector_db_dir, arcname=vector_db_dir.name, filter=_exclude_runtime_files)

            _generate_hash_sidecar(backup_path)
            logger.info("백업 파일 및 해시 검증 파일을 성공적으로 생성했습니다.")
            _update_status("completed", target=backup_path.name)
            rotate_backups(rotation_limit, backup_dir)
            return True
    except Exception as e:
        logger.error("백업 프로세스 실패: %s", e)
        _update_status("failed", error=str(e))
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
        logger.info("오래된 백업 삭제 완료: %s", archive_path.name)


def diagnose_db(vector_db_dir: Path = VECTOR_DB_DIR) -> bool:
    # 1. SQLite 파일 무결성 검사
    sqlite_file = vector_db_dir / "chroma.sqlite3"
    if not sqlite_file.exists():
        logger.error("진단 실패: DB 파일을 찾을 수 없습니다: %s", sqlite_file)
        return False

    try:
        conn = sqlite3.connect(f"file:{sqlite_file}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check;")
        result = cursor.fetchone()
        conn.close()

        if not (result and result[0] == "ok"):
            logger.error("DB 진단 실패: 무결성 검사 결과: %s", result)
            return False
        logger.info("DB 진단 통과: 무결성 검사 통과")
    except sqlite3.Error as e:
        logger.error("DB 진단 실패: SQLite 오류: %s", e)
        return False

    # 2. ChromaDB 클라이언트 실제 쿼리 검증
    try:
        # 시스템 설정의 임베딩 모델 사용
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBEDDING_MODEL_NAME,
            device="cpu",  # 진단용이므로 가볍게 CPU 사용
        )
        client = chromadb.PersistentClient(
            path=str(vector_db_dir), settings=chromadb.Settings(anonymized_telemetry=False)
        )
        collection = client.get_or_create_collection(
            name="diagnostic_collection", embedding_function=embedding_function
        )
        collection.add(ids=["test_id"], documents=["test document"])
        results = collection.query(query_texts=["test"], n_results=1)
        client.delete_collection(name="diagnostic_collection")

        if not results or not results.get("ids"):
            logger.error("DB 진단 실패: ChromaDB 쿼리 결과가 없습니다.")
            return False
        logger.info("DB 진단 통과: ChromaDB 쿼리 정상 작동")
    except Exception as e:
        logger.error("DB 진단 실패: ChromaDB 클라이언트 오류: %s", e)
        return False

    return True


def _find_backup(backup_file: str | None, backup_dir: Path) -> Path | None:
    if backup_file:
        return backup_dir / backup_file
    backups = sorted(backup_dir.glob(BACKUP_PATTERN), key=lambda path: path.stat().st_mtime, reverse=True)
    return backups[0] if backups else None


def _release_chromadb_locks() -> None:
    chroma_host = os.environ.get("CHROMA_SERVER_HOST")
    if not chroma_host:
        return
    try:
        client = chromadb.HttpClient(host=chroma_host, port=os.environ.get("CHROMA_SERVER_PORT", "8000"))
        client.reset()
        logger.info("ChromaDB 서버 리셋을 통해 파일 락(Lock)을 해제했습니다.")
    except Exception as e:
        logger.warning(f"ChromaDB 서버 리셋 시도 중 예외 발생 (무시됨): {e}")


def _move_existing_db(vector_db_dir: Path) -> Path | None:
    if not vector_db_dir.exists() or not any(vector_db_dir.iterdir()):
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_existing = vector_db_dir.parent / f"{vector_db_dir.name}_temp_{timestamp}"
    temp_existing.mkdir(parents=True, exist_ok=True)
    for item in vector_db_dir.iterdir():
        try:
            shutil.move(str(item), str(temp_existing / item.name))
        except Exception as e:
            logger.warning(f"임시 이동 실패 (무시됨): {item} - {e}")
    logger.info("기존 DB 내용을 임시 위치로 이동했습니다: %s", temp_existing)
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
        for item in vector_db_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)
        logger.info("잘못 복원된 DB 내용 제거 완료: %s", vector_db_dir)
    if temp_existing and temp_existing.exists():
        for item in temp_existing.iterdir():
            try:
                shutil.move(str(item), str(vector_db_dir / item.name))
            except Exception as e:
                logger.warning(f"원본 복원 중 실패 (무시됨): {item} - {e}")
        shutil.rmtree(temp_existing, ignore_errors=True)
        logger.info("원본 DB 복원 완료: %s", temp_existing)


def _remove_previous_db(temp_existing: Path | None) -> None:
    if temp_existing and temp_existing.exists():
        shutil.rmtree(temp_existing, ignore_errors=True)
        logger.info("임시 보관된 원본 DB 제거 완료: %s", temp_existing)


def restore_chromadb(
    backup_file: str | None = None,
    vector_db_dir: Path = VECTOR_DB_DIR,
    backup_dir: Path = BACKUP_DIR,
) -> bool:
    ensure_directories()
    backup_path = _find_backup(backup_file, backup_dir)
    if backup_path is None:
        logger.error("복원 실패: %s 디렉터리에서 백업 보관 파일을 찾을 수 없습니다.", backup_dir)
        _update_status("failed", error="백업 보관 파일을 찾을 수 없습니다.")
        return False
    if not backup_path.exists():
        logger.error("복원 실패: 백업 보관 파일이 존재하지 않습니다: %s", backup_path)
        _update_status("failed", error=f"백업 보관 파일이 존재하지 않습니다: {backup_path.name}")
        return False
    if not verify_hash_sidecar(backup_path):
        _update_status("failed", error=f"해시 검증에 실패했습니다: {backup_path.name}")
        return False

    _update_status("running_restore", target=backup_path.name)
    lock_file = vector_db_dir / LOCK_FILE_NAME if vector_db_dir.exists() else vector_db_dir.parent / LOCK_FILE_NAME
    temp_existing: Path | None = None

    try:
        with _operation_lock(lock_file):
            logger.info("백업에서 ChromaDB 복원 중: %s", backup_path.name)
            _release_chromadb_locks()
            temp_existing = _move_existing_db(vector_db_dir)
            _extract_archive(backup_path, vector_db_dir.parent)

            if not diagnose_db(vector_db_dir):
                _restore_previous_db(vector_db_dir, temp_existing)
                _update_status("failed", error="복원 후 데이터베이스 진단에 실패했습니다.", target=backup_path.name)
                return False

            _remove_previous_db(temp_existing)
            logger.info("백업에서 ChromaDB를 성공적으로 복원했습니다: %s", backup_path.name)
            _update_status("completed", target=backup_path.name)
            return True

    except Exception as e:
        logger.error("복원 프로세스 실패: %s", e)
        try:
            _restore_previous_db(vector_db_dir, temp_existing)
        except Exception as inner_e:
            logger.error("이전 DB 복구 중 추가 오류 발생: %s", inner_e)
        _update_status("failed", error=str(e), target=backup_path.name)
        return False
