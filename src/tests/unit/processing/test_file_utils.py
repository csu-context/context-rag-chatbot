import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.utils.file_utils import generate_file_hash


def test_generate_file_hash_normal(tmp_path):
    # 정상 파일 해싱 테스트
    file_path = tmp_path / "test.txt"
    file_path.write_bytes(b"hello world")

    with patch("src.utils.file_utils.RAW_DATA_DIR", tmp_path):
        h = generate_file_hash(file_path, "manual")
        assert len(h) == 12


def test_generate_file_hash_outside_raw_dir(tmp_path):
    # RAW_DATA_DIR 바깥의 경로에 대해 relative_to 실패로 name으로 대체되는 케이스 검증
    file_path = tmp_path / "test.txt"
    file_path.write_bytes(b"hello world")

    h = generate_file_hash(file_path, "manual")
    assert len(h) == 12


def test_generate_file_hash_fallback(tmp_path):
    # Exception 발생하여 fallback 로직 타는 케이스 검증
    file_path = tmp_path / "test_fallback.txt"
    file_path.write_bytes(b"hello fallback")

    # open을 mock하여 Exception 발생 유도
    with patch("builtins.open", side_effect=PermissionError("no read permission")):
        h = generate_file_hash(file_path, "manual")
        assert len(h) == 12
