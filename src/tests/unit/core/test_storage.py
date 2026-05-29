import tempfile
from pathlib import Path

import pytest

from src.core.storage import StorageManager


@pytest.fixture
def temp_storage_dirs():
    with tempfile.TemporaryDirectory() as tmp_proc_dir, tempfile.TemporaryDirectory() as tmp_cache_dir:
        yield Path(tmp_proc_dir), Path(tmp_cache_dir)


def test_storage_manager_operations(temp_storage_dirs):
    proc_dir, cache_dir = temp_storage_dirs
    manager = StorageManager(processed_dir=proc_dir, cache_dir=cache_dir)

    source_id = "test_doc_123"
    cache_data = {"raw_markdown": "hello world", "metadata": {"source": "test"}}

    # 1. 캐시 존재하지 않음 확인
    assert not manager.has_cache(source_id)

    # 2. 캐시 저장 및 확인
    manager.save_cache(source_id, cache_data)
    assert manager.has_cache(source_id)

    # 3. 캐시 로드
    loaded_cache = manager.load_cache(source_id)
    assert loaded_cache == cache_data

    # 4. JSON 가공 데이터 저장
    json_data = [{"chunk_id": "c1", "text": "chunk1"}]
    json_path = manager.save_processed_data(source_id, json_data)
    assert json_path.exists()

    # 5. 가공 데이터 로드
    loaded_json = manager.load_processed_file(json_path)
    assert loaded_json == json_data

    # 6. 목록 스캔
    files = manager.scan_processed_files()
    assert json_path in files

    # 7. 일괄 삭제
    deleted = manager.delete_processed_and_cache(source_id)
    assert deleted
    assert not manager.has_cache(source_id)
    assert not json_path.exists()
