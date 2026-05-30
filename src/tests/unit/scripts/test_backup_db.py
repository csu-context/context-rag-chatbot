import os
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.backup_db import backup_chromadb, rotate_backups
from src.utils.paths import BACKUP_DIR, VECTOR_DB_DIR


@pytest.fixture
def mock_backup_dir(tmp_path):
    """임시 백업 디렉토리 생성"""
    backup_path = tmp_path / "data" / "backups"
    backup_path.mkdir(parents=True)
    return backup_path


@pytest.fixture
def mock_vector_db_dir(tmp_path):
    """임시 벡터 DB 디렉토리 및 파일 생성"""
    db_path = tmp_path / "vector_db"
    db_path.mkdir(parents=True)
    (db_path / "chroma.sqlite3").write_text("dummy data")
    return db_path


def test_backup_chromadb_success(mock_vector_db_dir, mock_backup_dir):
    with patch("scripts.backup_db.BACKUP_DIR", mock_backup_dir), \
         patch("scripts.backup_db.VECTOR_DB_DIR", mock_vector_db_dir):
        
        # mock_backup_dir에 대한 glob 모킹이 필요할 수 있음 (rotate_backups용)
        # 하지만 실제 디렉토리이므로 glob()이 정상 작동함.

        # 실행
        success = backup_chromadb(rotation_limit=5)

    assert success is True
    # 백업 파일 생성 확인
    backups = list(mock_backup_dir.glob("chromadb_backup_*.tar.gz"))
    assert len(backups) == 1
    
    # 압축 내용 확인
    with tarfile.open(backups[0], "r:gz") as tar:
        # arcname=VECTOR_DB_DIR.name 으로 했으므로 vector_db/ 가 최상위여야 함
        assert "vector_db/chroma.sqlite3" in tar.getnames()


def test_rotate_backups(mock_backup_dir):
    # 더미 백업 파일들 생성 (시간차를 두어 정렬 보장)
    for i in range(7):
        f = mock_backup_dir / f"chromadb_backup_2026010{i}_000000.tar.gz"
        f.write_text("dummy")
        # mtime 설정을 통해 순서 보장 (optional, 기본적으로 생성 순서)

    with patch("scripts.backup_db.BACKUP_DIR", mock_backup_dir):
        # 실행: 5개만 남기기
        rotate_backups(limit=5)

    # 결과 확인
    remaining = sorted(list(mock_backup_dir.glob("chromadb_backup_*.tar.gz")))
    assert len(remaining) == 5


def test_backup_fails_when_db_locked(mock_vector_db_dir, mock_backup_dir):
    db_file = mock_vector_db_dir / "chroma.sqlite3"
    
    with patch("scripts.backup_db.VECTOR_DB_DIR", mock_vector_db_dir), \
         patch("scripts.backup_db.BACKUP_DIR", mock_backup_dir):
        
        # 파일을 다른 "프로세스"가 열고 있는 것처럼 시뮬레이션
        # backup_db.py의 open(db_file, "r+b") 시 side_effect 발생
        with patch("builtins.open", side_effect=IOError("File locked")):
            success = backup_chromadb()
            assert success is False
