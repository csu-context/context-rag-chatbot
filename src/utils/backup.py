"""Issue 28: ChromaDB 자동 백업/복원 모듈 (tarball, 최신 5개 순환삭제).

Docker 환경(CHROMA_SERVER_HOST 설정 시)에서는 ChromaDB HTTP API를 통해
컬렉션 데이터를 JSON으로 백업합니다. 로컬 환경에서는 VECTOR_DB_DIR을
tarball로 직접 압축합니다.
"""

import json
import logging
import os
import tarfile
from datetime import datetime
from pathlib import Path

from src.utils.paths import VECTOR_DB_DIR

logger = logging.getLogger(__name__)

_BACKUP_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "backups"
_MAX_BACKUPS = 5


def _is_remote_chroma() -> bool:
    """CHROMA_SERVER_HOST가 설정되어 있으면 원격(Docker) 모드."""
    return bool(os.getenv("CHROMA_SERVER_HOST"))


def _backup_via_http() -> Path | None:
    """ChromaDB HTTP API를 통해 컬렉션 데이터를 JSON으로 백업합니다."""
    _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = _BACKUP_DIR / f"chromadb_{timestamp}.json.gz"

    try:
        import gzip

        import chromadb
        from chromadb.config import Settings

        host = os.getenv("CHROMA_SERVER_HOST", "localhost")
        port = int(os.getenv("CHROMA_SERVER_PORT", "8000"))
        client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=Settings(anonymized_telemetry=False),
        )

        collections_data = {}
        for col in client.list_collections():
            data = col.get(include=["documents", "metadatas", "embeddings"])
            embeddings = data.get("embeddings")
            if embeddings is not None:
                embeddings = [e.tolist() if hasattr(e, "tolist") else e for e in embeddings]

            collections_data[col.name] = {
                "ids": data["ids"],
                "documents": data.get("documents"),
                "metadatas": data.get("metadatas"),
                # 임베딩은 용량이 크므로 포함 여부 선택
                "embeddings": embeddings,
            }

        with gzip.open(backup_path, "wt", encoding="utf-8") as f:
            json.dump(collections_data, f, ensure_ascii=False)

        size_kb = backup_path.stat().st_size // 1024
        total_docs = sum(len(c["ids"]) for c in collections_data.values())
        logger.info(f"ChromaDB HTTP 백업 완료: {backup_path} ({size_kb} KB, {total_docs} docs)")
        _rotate_backups()
        return backup_path
    except Exception as e:
        logger.error(f"ChromaDB HTTP 백업 실패: {e}")
        if backup_path.exists():
            backup_path.unlink(missing_ok=True)
        return None


def _backup_local() -> Path | None:
    """로컬 VECTOR_DB_DIR을 tarball로 압축합니다."""
    if not VECTOR_DB_DIR.exists():
        logger.warning(f"백업 대상 디렉토리가 존재하지 않습니다: {VECTOR_DB_DIR}")
        return None

    # 빈 디렉토리 경고
    if not any(VECTOR_DB_DIR.iterdir()):
        logger.warning(f"백업 대상 디렉토리가 비어 있습니다: {VECTOR_DB_DIR}")
        return None

    _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = _BACKUP_DIR / f"chromadb_{timestamp}.tar.gz"

    try:
        with tarfile.open(backup_path, "w:gz") as tar:
            tar.add(VECTOR_DB_DIR, arcname=VECTOR_DB_DIR.name)
        logger.info(f"ChromaDB 백업 완료: {backup_path} ({backup_path.stat().st_size // 1024} KB)")
        _rotate_backups()
        return backup_path
    except Exception as e:
        logger.error(f"ChromaDB 백업 실패: {e}")
        if backup_path.exists():
            backup_path.unlink(missing_ok=True)
        return None


def create_backup() -> Path | None:
    """ChromaDB 데이터를 백업합니다. 환경에 따라 HTTP API 또는 로컬 tarball 방식을 사용."""
    if _is_remote_chroma():
        return _backup_via_http()
    return _backup_local()


def _restore_via_http(backup_path: Path) -> bool:
    """`.json.gz` 논리 백업을 ChromaDB HTTP API로 복원합니다 (임베딩 그대로 upsert)."""
    try:
        import gzip

        import chromadb
        from chromadb.config import Settings

        host = os.getenv("CHROMA_SERVER_HOST", "localhost")
        port = int(os.getenv("CHROMA_SERVER_PORT", "8000"))
        client = chromadb.HttpClient(host=host, port=port, settings=Settings(anonymized_telemetry=False))

        with gzip.open(backup_path, "rt", encoding="utf-8") as f:
            collections_data = json.load(f)

        for name, payload in collections_data.items():
            ids = payload.get("ids") or []
            if not ids:
                continue
            collection = client.get_or_create_collection(name=name)
            upsert_kwargs = {"ids": ids}
            for key in ("documents", "metadatas", "embeddings"):
                if payload.get(key) is not None:
                    upsert_kwargs[key] = payload[key]
            collection.upsert(**upsert_kwargs)

        logger.info(f"ChromaDB HTTP 복원 완료: {backup_path}")
        return True
    except Exception as e:
        logger.error(f"ChromaDB HTTP 복원 실패: {e}")
        return False


def _restore_local(backup_path: Path, target_dir: Path | None = None) -> bool:
    """`.tar.gz` 백업을 VECTOR_DB_DIR로 압축 해제하여 복원합니다."""
    target = target_dir or VECTOR_DB_DIR.parent
    try:
        with tarfile.open(backup_path, "r:gz") as tar:
            tar.extractall(path=target, filter="data")  # 경로 traversal 멤버 거부
        logger.info(f"ChromaDB 복원 완료: {backup_path} → {target}")
        return True
    except Exception as e:
        logger.error(f"ChromaDB 복원 실패: {e}")
        return False


def restore_backup(backup_path: Path, target_dir: Path | None = None) -> bool:
    """백업에서 ChromaDB 데이터를 복원합니다.

    `.json.gz`(원격/HTTP 논리 백업)는 HTTP API로, `.tar.gz`(로컬)는 압축 해제로 복원.
    """
    if not backup_path.exists():
        logger.error(f"백업 파일이 존재하지 않습니다: {backup_path}")
        return False
    if backup_path.name.endswith(".json.gz"):
        return _restore_via_http(backup_path)
    return _restore_local(backup_path, target_dir)


def list_backups() -> list[Path]:
    """최신 순으로 정렬된 백업 파일 목록을 반환합니다."""
    if not _BACKUP_DIR.exists():
        return []
    return sorted(
        list(_BACKUP_DIR.glob("chromadb_*.tar.gz")) + list(_BACKUP_DIR.glob("chromadb_*.json.gz")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _rotate_backups() -> None:
    """최신 MAX_BACKUPS개만 보존하고 나머지 삭제."""
    backups = list_backups()
    for old in backups[_MAX_BACKUPS:]:
        try:
            old.unlink()
            logger.info(f"오래된 백업 삭제: {old.name}")
        except Exception as e:
            logger.warning(f"백업 삭제 실패 {old.name}: {e}")
