import tarfile
from unittest.mock import patch

import pytest

from src.vector_db.backup_manager import _generate_hash_sidecar, diagnose_db, restore_chromadb


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


def test_diagnose_db_success(tmp_path):
    import sqlite3

    vector_db_dir = tmp_path / "vector_db"
    vector_db_dir.mkdir()
    conn = sqlite3.connect(str(vector_db_dir / "chroma.sqlite3"))
    conn.execute("CREATE TABLE collections (id TEXT)")
    conn.execute("CREATE TABLE embeddings (id INTEGER)")
    conn.execute("INSERT INTO collections VALUES ('c1')")
    conn.execute("INSERT INTO embeddings VALUES (1)")
    conn.commit()
    conn.close()

    assert diagnose_db(vector_db_dir=vector_db_dir) is True


def test_diagnose_db_failure_integrity_check(mock_dirs):
    _, vector_db_dir = mock_dirs

    with patch("sqlite3.connect") as mock_connect:
        mock_conn = mock_connect.return_value
        mock_cursor = mock_conn.cursor.return_value
        mock_cursor.fetchone.return_value = ("corrupt",)

        success = diagnose_db(vector_db_dir=vector_db_dir)

        assert success is False


def test_diagnose_db_failure_missing_file(tmp_path):
    vector_db_dir = tmp_path / "empty_vector_db"
    vector_db_dir.mkdir()

    success = diagnose_db(vector_db_dir=vector_db_dir)

    assert success is False
