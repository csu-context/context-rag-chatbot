import gzip
import json
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.utils.backup import create_backup, restore_backup


@pytest.fixture
def mock_db_manager():
    manager = MagicMock()
    manager.collection.get.return_value = {
        "ids": ["doc1"],
        "metadatas": [{"source": "test.pdf"}],
        "documents": ["Test document content"],
    }
    return manager


def test_create_backup_success(tmp_path):
    # Test _backup_local
    db_dir = tmp_path / "vector_db"
    db_dir.mkdir(parents=True)
    # Create a dummy file so iterdir() is not empty
    (db_dir / "dummy.txt").write_text("data")

    with (
        patch("src.utils.backup.VECTOR_DB_DIR", db_dir),
        patch("src.utils.backup._BACKUP_DIR", tmp_path / "backups"),
        patch("src.utils.backup._is_remote_chroma", return_value=False),
    ):
        backup_file = create_backup()
        assert backup_file is not None
        assert backup_file.exists()
        assert str(backup_file).endswith(".tar.gz")


def test_create_backup_with_host_env(mock_db_manager, tmp_path):
    # Test _backup_via_http
    with (
        patch("src.utils.backup._BACKUP_DIR", tmp_path / "backups"),
        patch("src.utils.backup._is_remote_chroma", return_value=True),
        patch("chromadb.HttpClient") as mock_client,
    ):
        mock_col = MagicMock()
        mock_col.name = "test_col"
        mock_col.get.return_value = {
            "ids": ["doc1"],
            "metadatas": [{"source": "test.pdf"}],
            "documents": ["Test document content"],
            "embeddings": None,
        }
        mock_client.return_value.list_collections.return_value = [mock_col]

        backup_file = create_backup()

        assert backup_file is not None
        assert backup_file.exists()
        assert str(backup_file).endswith(".json.gz")

        with gzip.open(backup_file, "rt") as f:
            data = json.load(f)
            assert "test_col" in data


def test_create_backup_failure():
    with (
        patch("src.utils.backup._is_remote_chroma", return_value=False),
        patch("src.utils.backup.VECTOR_DB_DIR", Path("/nonexistent/path")),
    ):
        backup_file = create_backup()
        assert backup_file is None


def test_restore_backup_via_http(tmp_path):
    # .json.gz 논리 백업 → HTTP API 복원
    backup_file = tmp_path / "chromadb_20260101_000000.json.gz"
    with gzip.open(backup_file, "wt", encoding="utf-8") as f:
        json.dump(
            {
                "test_col": {
                    "ids": ["doc1"],
                    "documents": ["내용"],
                    "metadatas": [{"source": "test.pdf"}],
                    "embeddings": [[0.1, 0.2, 0.3]],
                }
            },
            f,
            ensure_ascii=False,
        )

    with patch("chromadb.HttpClient") as mock_client:
        mock_col = MagicMock()
        mock_client.return_value.get_or_create_collection.return_value = mock_col

        ok = restore_backup(backup_file)

        assert ok is True
        mock_col.upsert.assert_called_once()
        _, kwargs = mock_col.upsert.call_args
        assert kwargs["ids"] == ["doc1"]
        assert kwargs["embeddings"] == [[0.1, 0.2, 0.3]]


def test_restore_backup_local(tmp_path):
    # .tar.gz 백업 → 로컬 압축 해제 복원
    src_dir = tmp_path / "vector_db"
    src_dir.mkdir()
    (src_dir / "dummy.txt").write_text("data")
    backup_file = tmp_path / "chromadb_20260101_000000.tar.gz"
    with tarfile.open(backup_file, "w:gz") as tar:
        tar.add(src_dir, arcname="vector_db")

    target = tmp_path / "restore_target"
    target.mkdir()
    ok = restore_backup(backup_file, target_dir=target)

    assert ok is True
    assert (target / "vector_db" / "dummy.txt").exists()


def test_restore_backup_missing_file(tmp_path):
    assert restore_backup(tmp_path / "nope.tar.gz") is False
