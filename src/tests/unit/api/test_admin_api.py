import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


@pytest.fixture
def mock_backup_dir(tmp_path):
    backup_path = tmp_path / "data" / "backups"
    backup_path.mkdir(parents=True)
    return backup_path


@pytest.fixture
def mock_backup_files(mock_backup_dir):
    files = []
    for i in range(3):
        file = mock_backup_dir / f"chromadb_backup_2023010{i + 1}_120000.tar.gz"
        file.write_text(f"dummy{i + 1}")
        time.sleep(0.01)
        files.append(file)
    return files


def test_get_backups(mock_backup_dir, mock_backup_files):
    with patch("src.api.main.BACKUP_DIR", mock_backup_dir):
        response = client.get("/api/admin/backups")
        assert response.status_code == 200
        data = response.json()
        assert "backups" in data
        assert len(data["backups"]) == 3
        assert data["backups"][0]["filename"] == "chromadb_backup_20230103_120000.tar.gz"


def test_get_backup_status_no_file(mock_backup_dir):
    with patch("src.api.main.BACKUP_DIR", mock_backup_dir):
        response = client.get("/api/admin/backups/status")
        assert response.status_code == 200
        assert response.json() == {"status": "idle"}


def test_get_backup_status_with_file(mock_backup_dir):
    import json
    status_file = mock_backup_dir / "backup_status.json"
    dummy_data = {
        "status": "running_backup",
        "last_update": 123456.0,
        "error": None,
        "target": "chromadb_backup_20230101_120000.tar.gz"
    }
    status_file.write_text(json.dumps(dummy_data), encoding="utf-8")

    with patch("src.api.main.BACKUP_DIR", mock_backup_dir):
        response = client.get("/api/admin/backups/status")
        assert response.status_code == 200
        assert response.json() == dummy_data


def test_create_backup():
    # TestClient는 BackgroundTasks를 비동기가 아닌 동기적으로 즉시 실행합니다.
    # 따라서 override 할 필요 없이 실제 함수(backup_chromadb)가 호출되었는지만 검증하면 됩니다.
    with patch("src.api.main.backup_chromadb") as mock_backup_chromadb:
        response = client.post("/api/admin/backups")

        assert response.status_code == 200
        assert response.json() == {"message": "Backup job started in the background."}

        # 백업 함수가 1회 정상 호출되었는지 확인
        mock_backup_chromadb.assert_called_once()


def test_restore_backup():
    test_filename = "test_backup.tar.gz"

    # 마찬가지로 override 코드 제거 후, 실제 함수 호출만 검증합니다.
    with patch("src.api.main.restore_chromadb") as mock_restore_chromadb:
        response = client.post(f"/api/admin/backups/restore?filename={test_filename}")

        assert response.status_code == 200
        assert response.json() == {"message": f"Restore job for {test_filename} started in the background."}

        # 복원 함수가 파일명 파라미터와 함께 1회 정상 호출되었는지 확인
        mock_restore_chromadb.assert_called_once_with(test_filename)
