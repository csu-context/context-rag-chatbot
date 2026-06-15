import tarfile
import time
from unittest.mock import patch

import pytest

from src.vector_db.backup_manager import (
    HASH_SUFFIX,
    _generate_hash_sidecar,
    backup_chromadb,
    diagnose_db,
    restore_chromadb,
    rotate_backups,
)


@pytest.fixture
def mock_backup_dir(tmp_path):
    backup_path = tmp_path / "data" / "backups"
    backup_path.mkdir(parents=True)
    return backup_path


@pytest.fixture
def mock_vector_db_dir(tmp_path):
    db_path = tmp_path / "vector_db"
    db_path.mkdir(parents=True)
    (db_path / "chroma.sqlite3").write_text("dummy data", encoding="utf-8")
    return db_path


def test_backup_chromadb_success(mock_vector_db_dir, mock_backup_dir):
    success = backup_chromadb(
        rotation_limit=5,
        vector_db_dir=mock_vector_db_dir,
        backup_dir=mock_backup_dir,
        quiet_period_seconds=0,
        stability_timeout_seconds=1,
    )

    assert success is True

    backups = list(mock_backup_dir.glob("chromadb_backup_*.tar.gz"))
    assert len(backups) == 1
    assert backups[0].with_name(f"{backups[0].name}{HASH_SUFFIX}").exists()

    with tarfile.open(backups[0], "r:gz") as tar:
        names = tar.getnames()
        assert "vector_db/chroma.sqlite3" in names
        assert "vector_db/.chromadb_backup.lock" not in names


def test_rotate_backups_removes_archives_and_hashes(mock_backup_dir):
    for i in range(7):
        archive = mock_backup_dir / f"chromadb_backup_2026010{i}_000000.tar.gz"
        archive.write_text("dummy", encoding="utf-8")
        archive.with_name(f"{archive.name}{HASH_SUFFIX}").write_text("hash", encoding="utf-8")

    rotate_backups(limit=5, backup_dir=mock_backup_dir)

    remaining = sorted(mock_backup_dir.glob("chromadb_backup_*.tar.gz"))
    remaining_hashes = sorted(mock_backup_dir.glob(f"chromadb_backup_*.tar.gz{HASH_SUFFIX}"))
    assert len(remaining) == 5
    assert len(remaining_hashes) == 5


def test_backup_proceeds_even_on_timeout(mock_vector_db_dir, mock_backup_dir, monkeypatch):
    from src.vector_db import backup_manager

    def mock_get_db_mtime(*args, **kwargs):
        return time.time()

    monkeypatch.setattr(backup_manager, "_get_db_mtime", mock_get_db_mtime)

    success = backup_chromadb(
        vector_db_dir=mock_vector_db_dir,
        backup_dir=mock_backup_dir,
        quiet_period_seconds=2,
        stability_timeout_seconds=1,
    )

    assert success is True
    assert len(list(mock_backup_dir.glob("chromadb_backup_*.tar.gz"))) == 1


@pytest.fixture
def mock_dirs(tmp_path):
    backup_dir = tmp_path / "data" / "backups"
    backup_dir.mkdir(parents=True)

    vector_db_dir = tmp_path / "vector_db"
    vector_db_dir.mkdir(parents=True)
    (vector_db_dir / "chroma.sqlite3").write_text("old data", encoding="utf-8")

    return backup_dir, vector_db_dir


def _create_backup(backup_dir, source_parent):
    source_db = source_parent / "source_db"
    source_db.mkdir()
    (source_db / "chroma.sqlite3").write_text("restored data", encoding="utf-8")

    backup_file = backup_dir / "chromadb_backup_test.tar.gz"
    with tarfile.open(backup_file, "w:gz") as tar:
        tar.add(source_db, arcname="vector_db")
    _generate_hash_sidecar(backup_file)
    return backup_file


def test_restore_chromadb_success(mock_dirs):
    backup_dir, vector_db_dir = mock_dirs
    backup_file = _create_backup(backup_dir, vector_db_dir.parent)

    with patch("src.vector_db.backup_manager.diagnose_db", return_value=True):
        success = restore_chromadb(
            backup_file=backup_file.name,
            vector_db_dir=vector_db_dir,
            backup_dir=backup_dir,
        )

    assert success is True
    assert (vector_db_dir / "chroma.sqlite3").read_text(encoding="utf-8") == "restored data"


def test_update_status_persists_note(tmp_path):
    """복원 성공 시 'chromadb 재시작 필요' 안내가 상태 파일의 note에 기록된다."""
    import json

    from src.vector_db import backup_manager

    with patch.object(backup_manager, "BACKUP_DIR", tmp_path):
        backup_manager._update_status("completed", target="x.tar.gz", note="chromadb 재시작 필요")
        data = json.loads((tmp_path / "backup_status.json").read_text(encoding="utf-8"))

    assert data["status"] == "completed"
    assert data["target"] == "x.tar.gz"
    assert data["note"] == "chromadb 재시작 필요"


def test_restore_rejects_hash_mismatch(mock_dirs):
    backup_dir, vector_db_dir = mock_dirs
    backup_file = _create_backup(backup_dir, vector_db_dir.parent)
    backup_file.write_bytes(backup_file.read_bytes() + b"corrupt")

    success = restore_chromadb(
        backup_file=backup_file.name,
        vector_db_dir=vector_db_dir,
        backup_dir=backup_dir,
    )

    assert success is False
    assert (vector_db_dir / "chroma.sqlite3").read_text(encoding="utf-8") == "old data"


def _write_chroma_sqlite(db_file, collection_count=1, embedding_count=3):
    """diagnose_db 검증용 최소 chroma sqlite(collections·embeddings 테이블) 생성."""
    import sqlite3

    conn = sqlite3.connect(str(db_file))
    conn.execute("CREATE TABLE collections (id TEXT)")
    conn.execute("CREATE TABLE embeddings (id INTEGER)")
    conn.executemany("INSERT INTO collections VALUES (?)", [(str(i),) for i in range(collection_count)])
    conn.executemany("INSERT INTO embeddings VALUES (?)", [(i,) for i in range(embedding_count)])
    conn.commit()
    conn.close()


def test_diagnose_db_success(tmp_path):
    """read-only sqlite로 collections/embeddings 존재를 확인해 진단 통과한다."""
    db_dir = tmp_path / "vector_db"
    db_dir.mkdir()
    _write_chroma_sqlite(db_dir / "chroma.sqlite3", collection_count=1, embedding_count=5)

    assert diagnose_db(vector_db_dir=db_dir) is True


def test_diagnose_db_fails_on_sqlite_integrity(mock_vector_db_dir):
    with patch("sqlite3.connect") as mock_sqlite_connect:
        mock_conn = mock_sqlite_connect.return_value
        mock_cursor = mock_conn.cursor.return_value
        mock_cursor.fetchone.return_value = ("corrupt",)

        success = diagnose_db(vector_db_dir=mock_vector_db_dir)
        assert success is False


def test_diagnose_db_fails_when_no_collections(tmp_path):
    """무결성은 통과하지만 복원된 DB에 컬렉션이 없으면 진단 실패한다."""
    db_dir = tmp_path / "vector_db"
    db_dir.mkdir()
    _write_chroma_sqlite(db_dir / "chroma.sqlite3", collection_count=0, embedding_count=0)

    assert diagnose_db(vector_db_dir=db_dir) is False
