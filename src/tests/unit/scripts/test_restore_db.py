import tarfile
from unittest.mock import MagicMock, patch

import pytest

from scripts.restore_db import diagnose_db, restore_chromadb


@pytest.fixture
def mock_dirs(tmp_path):
    backup_dir = tmp_path / "data" / "backups"
    backup_dir.mkdir(parents=True)

    vector_db_dir = tmp_path / "vector_db"
    vector_db_dir.mkdir(parents=True)

    return backup_dir, vector_db_dir


def test_restore_chromadb_success(mock_dirs):
    backup_dir, vector_db_dir = mock_dirs

    # 더미 백업 파일 생성
    source_db = vector_db_dir.parent / "source_db"
    source_db.mkdir()
    (source_db / "chroma.sqlite3").write_text("restored data")

    backup_file = backup_dir / "chromadb_backup_test.tar.gz"
    with tarfile.open(backup_file, "w:gz") as tar:
        tar.add(source_db, arcname="vector_db")

    # restore_db.py 내부의 상수를 패치
    with (
        patch("scripts.restore_db.BACKUP_DIR", backup_dir),
        patch("scripts.restore_db.VECTOR_DB_DIR", vector_db_dir),
        patch("scripts.restore_db.diagnose_db", return_value=True),
        patch("shutil.rmtree"),
    ):
        # 실행
        success = restore_chromadb(backup_file=backup_file.name)

    assert success is True
    # 압축 해제 확인 (vector_db_dir 내부에 파일이 생겼는지)
    assert (vector_db_dir / "chroma.sqlite3").exists()


@patch("scripts.restore_db.ChromaDBManager")
def test_diagnose_db_success(mock_manager_class):
    mock_manager = MagicMock()
    mock_manager.get_count.return_value = 10
    mock_manager.search.return_value = [{"content": "test"}]
    mock_manager_class.return_value = mock_manager

    success = diagnose_db()

    assert success is True
    mock_manager.get_count.assert_called_once()
    mock_manager.search.assert_called_once()


@patch("scripts.restore_db.ChromaDBManager")
def test_diagnose_db_failure(mock_manager_class):
    mock_manager = MagicMock()
    mock_manager.get_count.side_effect = Exception("DB Error")
    mock_manager_class.return_value = mock_manager

    success = diagnose_db()

    assert success is False
