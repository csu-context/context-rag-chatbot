import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# FastAPI 애플리케이션 임포트
from src.api.main import app

client = TestClient(app)


@pytest.fixture
def mock_backup_dir(tmp_path):
    # 테스트용 백업 디렉토리 생성
    backup_path = tmp_path / "data" / "backups"
    backup_path.mkdir(parents=True)
    return backup_path


@pytest.fixture
def mock_backup_files(mock_backup_dir):
    # 테스트용 백업 파일 생성
    file1 = mock_backup_dir / "chromadb_backup_20230101_120000.tar.gz"
    file2 = mock_backup_dir / "chromadb_backup_20230102_130000.tar.gz"
    file3 = mock_backup_dir / "chromadb_backup_20230103_140000.tar.gz"

    file1.write_text("dummy1")
    time.sleep(0.01)  # Ensure different mtime
    file2.write_text("dummy2")
    time.sleep(0.01)
    file3.write_text("dummy3")

    return [file1, file2, file3]


def test_get_backups(mock_backup_dir, mock_backup_files):
    with patch("src.utils.paths.BACKUP_DIR", mock_backup_dir):
        response = client.get("/api/admin/backups")
        assert response.status_code == 200
        data = response.json()
        assert "backups" in data
        assert len(data["backups"]) == 3

        # 최신 파일이 먼저 오는지 확인 (mtime 기준 내림차순)
        assert data["backups"][0]["filename"] == "chromadb_backup_20230103_140000.tar.gz"
        assert data["backups"][1]["filename"] == "chromadb_backup_20230102_130000.tar.gz"
        assert data["backups"][2]["filename"] == "chromadb_backup_20230101_120000.tar.gz"


def test_create_backup():
    with patch("src.vector_db.backup_manager.backup_chromadb") as mock_backup_chromadb:
        response = client.post("/api/admin/backups")
        assert response.status_code == 200
        assert response.json() == {"message": "Backup job started in the background."}
        mock_backup_chromadb.assert_called_once()


def test_restore_backup():
    test_filename = "test_backup.tar.gz"
    with patch("src.vector_db.backup_manager.restore_chromadb") as mock_restore_chromadb:
        response = client.post(f"/api/admin/backups/restore?filename={test_filename}")
        assert response.status_code == 200
        assert response.json() == {"message": f"Restore job for {test_filename} started in the background."}
        mock_restore_chromadb.assert_called_once_with(test_filename)
