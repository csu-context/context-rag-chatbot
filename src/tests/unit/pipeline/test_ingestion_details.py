import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.common.constants import MetadataFields
from src.pipeline.ingestion import (
    _chunk_sections,
    _get_parser_strategy_for_file,
    _safe_invoke_progress,
    _split_table_into_row_chunks,
)
from src.pipeline.strategies import (
    DoclingPDFParserStrategy,
    ManualParserStrategy,
    MarkdownParserStrategy,
)


def test_safe_invoke_progress():
    # 1. 정상 호출 검증
    mock_cb = MagicMock()
    _safe_invoke_progress(mock_cb, 5, 10, "test_file.pdf")
    mock_cb.assert_called_once_with(5, 10, "test_file.pdf")

    # 2. 콜백 내부 예외 발생 시 에러 없이 예외 처리되는지 검증
    mock_cb_err = MagicMock(side_effect=ValueError("callback error"))
    try:
        _safe_invoke_progress(mock_cb_err, 5, 10, "test_file.pdf")
    except Exception as e:
        pytest.fail(f"safe_invoke_progress raised an exception: {e}")


def test_get_parser_strategy_for_file():
    from src.utils.paths import RAW_DATA_DIR

    # 1. Non-PDF 파일은 MarkdownParserStrategy 반환
    strategy = _get_parser_strategy_for_file(RAW_DATA_DIR / "test.md", None)
    assert isinstance(strategy, MarkdownParserStrategy)

    strategy = _get_parser_strategy_for_file(RAW_DATA_DIR / "test.txt", None)
    assert isinstance(strategy, MarkdownParserStrategy)

    # 2. PDF 파일 기본값은 ManualParserStrategy 반환
    strategy = _get_parser_strategy_for_file(RAW_DATA_DIR / "test.pdf", None)
    assert isinstance(strategy, ManualParserStrategy)

    # 3. manifest 설정에 따른 파서 설정 검증
    file_parser_types = {"test.pdf": "docling"}
    # docling 라이브러리 존재 여부 패치
    with patch("importlib.util.find_spec", return_value=MagicMock()) as mock_find:
        strategy = _get_parser_strategy_for_file(RAW_DATA_DIR / "test.pdf", file_parser_types)
        assert isinstance(strategy, DoclingPDFParserStrategy)

    with patch("importlib.util.find_spec", return_value=None):
        strategy = _get_parser_strategy_for_file(RAW_DATA_DIR / "test.pdf", file_parser_types)
        assert isinstance(strategy, ManualParserStrategy)


def test_split_table_into_row_chunks():
    # 1. 정상 마크다운 표 구조 청킹 검증
    table_content = (
        "학부 졸업 이수 요건 표\n\n"
        "| 학년 | 학점 | 비고 |\n"
        "|---|---|---|\n"
        "| 1학년 | 30 | - |\n"
        "| 2학년 | 60 | 필수 |\n"
    )
    parent_id = "parent-123"
    base_meta = {"src_name": "rules.pdf"}

    children = _split_table_into_row_chunks(table_content, parent_id, base_meta)
    assert len(children) == 2
    assert children[0]["chunk_id"] == "parent-123_r0"
    assert children[0]["metadata"][MetadataFields.IS_TABLE] is True
    assert "1학년" in children[0]["text"]
    assert "학부 졸업 이수 요건 표" in children[0]["text"]
    assert "|---|---|---|" in children[0]["text"]

    # 2. 비정상 표 구조 예외 검증
    short_table = "| 학년 | 학점 |\n"
    res = _split_table_into_row_chunks(short_table, parent_id, base_meta)
    assert res == []


def test_chunk_sections_empty():
    assert _chunk_sections([]) == []


def test_chunk_sections_raw_markdown():
    sections = [
        {
            "is_raw_markdown": True,
            "content": "# 대제목\n## 소제목\n내용입니다.",
            "metadata": {"src_name": "test.md"},
        }
    ]
    chunks = _chunk_sections(sections)
    assert len(chunks) > 0


def test_chunk_sections_combined():
    sections = [
        {
            "is_combined": True,
            "content": "대형 텍스트 청크 결합 테스트 내용입니다. " * 50,
            "metadata": {"src_name": "combined.md"},
        }
    ]
    chunks = _chunk_sections(sections)
    assert len(chunks) > 0
