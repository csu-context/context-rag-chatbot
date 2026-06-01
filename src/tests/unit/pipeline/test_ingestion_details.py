from unittest.mock import MagicMock, patch

import pytest

from src.common.constants import MetadataFields
from src.pipeline.ingestion import (
    _safe_invoke_progress,
)
from src.pipeline.strategies import (
    DoclingPDFParserStrategy,
    HwpParserStrategy,
    ManualParserStrategy,
    MarkdownParserStrategy,
    ParserFactory,
)
from src.processing.chunking import _split_table_into_row_chunks, chunk_sections


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
    strategy = ParserFactory.create(RAW_DATA_DIR / "test.md", None)
    assert isinstance(strategy, MarkdownParserStrategy)

    strategy = ParserFactory.create(RAW_DATA_DIR / "test.txt", None)
    assert isinstance(strategy, MarkdownParserStrategy)

    # 2. PDF 파일이면서 file_parser_types가 None인 경우 기본 manual
    strategy = ParserFactory.create(RAW_DATA_DIR / "test.pdf", None)
    assert isinstance(strategy, ManualParserStrategy)

    # 3. PDF 파일이면서 file_parser_types에 manual로 지정된 경우
    strategy = ParserFactory.create(RAW_DATA_DIR / "test.pdf", {"test.pdf": "manual"})
    assert isinstance(strategy, ManualParserStrategy)

    # 4. PDF 파일이면서 file_parser_types에 docling으로 지정된 경우
    with patch("importlib.util.find_spec") as mock_find_spec:
        mock_find_spec.return_value = True  # docling 설치된 것으로 모방
        strategy = ParserFactory.create(RAW_DATA_DIR / "test_docling.pdf", {"test_docling.pdf": "docling"})
        assert isinstance(strategy, DoclingPDFParserStrategy)

    # 5. docling 지정되었으나 docling이 설치되지 않은 경우 fallback to manual
    with patch("importlib.util.find_spec") as mock_find_spec:
        mock_find_spec.return_value = None  # docling 미설치 모방
        strategy = ParserFactory.create(RAW_DATA_DIR / "test_docling.pdf", {"test_docling.pdf": "docling"})
        assert isinstance(strategy, ManualParserStrategy)

    # 6. HWP/HWPX 파일은 HwpParserStrategy 반환 (ParserFactory 라우팅)
    strategy = ParserFactory.create(RAW_DATA_DIR / "test.hwp", None)
    assert isinstance(strategy, HwpParserStrategy)

    strategy = ParserFactory.create(RAW_DATA_DIR / "test.hwpx", None)
    assert isinstance(strategy, HwpParserStrategy)


def test_split_table_into_row_chunks():
    # 1. 정상 마크다운 표 구조 청킹 검증
    table_content = (
        "학부 졸업 이수 요건 표\n\n| 학년 | 학점 | 비고 |\n|---|---|---|\n| 1학년 | 30 | - |\n| 2학년 | 60 | 필수 |\n"
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
    assert chunk_sections([]) == []


def test_chunk_sections_raw_markdown():
    sections = [
        {
            "is_raw_markdown": True,
            "content": "# 테스트\n## 섹션\n본문입니다.",
            "metadata": {"src_name": "test.md"},
        }
    ]
    chunks = chunk_sections(sections)
    assert len(chunks) > 0


def test_chunk_sections_combined():
    sections = [
        {
            "is_combined": True,
            "content": "이것은 텍스트 청크 분할 테스트 문장입니다. " * 50,
            "metadata": {"src_name": "combined.md"},
        }
    ]
    chunks = chunk_sections(sections)
    assert len(chunks) > 0
