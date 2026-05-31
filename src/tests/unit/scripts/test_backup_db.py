import tarfile

import pytest

from src.utils.backup_manager import HASH_SUFFIX, backup_chromadb, rotate_backups


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


def test_backup_fails_when_db_changes_during_archive(mock_vector_db_dir, mock_backup_dir, monkeypatch):
    from src.utils import backup_manager

    original = backup_manager._ensure_snapshot_unchanged

    def mutate_then_check(before, directory):
        (directory / "chroma.sqlite3").write_text("changed", encoding="utf-8")
        original(before, directory)

    monkeypatch.setattr(backup_manager, "_ensure_snapshot_unchanged", mutate_then_check)

    success = backup_chromadb(
        vector_db_dir=mock_vector_db_dir,
        backup_dir=mock_backup_dir,
        quiet_period_seconds=0,
        stability_timeout_seconds=1,
    )

    assert success is False
    assert list(mock_backup_dir.glob("chromadb_backup_*.tar.gz")) == []
