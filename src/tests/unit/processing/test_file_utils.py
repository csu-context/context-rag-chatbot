from unittest.mock import patch

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


def test_generate_file_hash_includes_doc_type(tmp_path):
    # doc_type이 캐시 키에 반영되어 DOC_TYPE 전환 시 해시가 달라져야 한다(stale 파싱 방지).
    file_path = tmp_path / "doc.pdf"
    file_path.write_bytes(b"same content")
    with patch("src.utils.file_utils.RAW_DATA_DIR", tmp_path):
        h_legal = generate_file_hash(file_path, "docling", doc_type="legal")
        h_general = generate_file_hash(file_path, "docling", doc_type="general")
        assert h_legal != h_general
        # 동일 doc_type은 동일 해시(결정성)
        assert h_legal == generate_file_hash(file_path, "docling", doc_type="legal")


def test_generate_file_hash_fallback(tmp_path):
    # Exception 발생하여 fallback 로직 타는 케이스 검증
    file_path = tmp_path / "test_fallback.txt"
    file_path.write_bytes(b"hello fallback")

    # open을 mock하여 Exception 발생 유도
    with patch("builtins.open", side_effect=PermissionError("no read permission")):
        h = generate_file_hash(file_path, "manual")
        assert len(h) == 12
