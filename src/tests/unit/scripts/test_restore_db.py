import tarfile
from unittest.mock import MagicMock, patch

import pytest

from src.utils.backup_manager import diagnose_db, restore_chromadb, write_hash_sidecar


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
    write_hash_sidecar(backup_file)
    return backup_file


def test_restore_chromadb_success(mock_dirs):
    backup_dir, vector_db_dir = mock_dirs
    backup_file = _create_backup(backup_dir, vector_db_dir.parent)

    with patch("src.utils.backup_manager.diagnose_db", return_value=True):
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


def test_diagnose_db_success():
    mock_manager = MagicMock()
    mock_manager.get_count.return_value = 10
    mock_manager.search.return_value = [{"content": "test"}]

    success = diagnose_db(manager_factory=lambda: mock_manager)

    assert success is True
    mock_manager.get_count.assert_called_once()
    mock_manager.search.assert_called_once_with(query_text="diagnostic query", k=1)


def test_diagnose_db_failure_when_query_returns_no_results():
    mock_manager = MagicMock()
    mock_manager.get_count.return_value = 10
    mock_manager.search.return_value = []

    success = diagnose_db(manager_factory=lambda: mock_manager)

    assert success is False


def test_diagnose_db_failure_on_exception():
    mock_manager = MagicMock()
    mock_manager.get_count.side_effect = Exception("DB Error")

    success = diagnose_db(manager_factory=lambda: mock_manager)

    assert success is False
