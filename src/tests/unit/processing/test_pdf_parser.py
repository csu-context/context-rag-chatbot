from unittest.mock import MagicMock, patch

import pytest

from src.processing.pdf_parser import DoclingPDFParser


def test_docling_pdf_parser_init():
    parser = DoclingPDFParser()
    assert parser._converter is None


def test_docling_pdf_parser_import_error():
    parser = DoclingPDFParser()
    with patch("builtins.__import__", side_effect=ImportError("mock import error")):
        with pytest.raises(ImportError) as exc_info:
            parser._get_converter()
        assert "docling 라이브러리가 필요합니다" in str(exc_info.value)


def test_docling_pdf_parser_parse():
    parser = DoclingPDFParser()

    # Create mock docling objects
    mock_converter = MagicMock()
    mock_result = MagicMock()
    mock_doc = MagicMock()

    mock_converter.convert.return_value = mock_result
    mock_result.document = mock_doc

    # Mock export_to_markdown
    mock_doc.export_to_markdown.return_value = (
        "Header text\nSome content\nFooter text\n<!-- page break -->\nHeader text\nPage two content\nFooter text"
    )

    # Mock doc.pages for get_page_count
    mock_doc.pages = [MagicMock(), MagicMock()]

    # Mock doc.tables for table extraction
    mock_table1 = MagicMock()
    mock_table1.prov = [MagicMock(page_no=1)]
    mock_table1.export_to_markdown.return_value = "| A | B |"
    mock_table1.data.grid = [["A", "B"], ["1", "2"]]

    mock_doc.tables = [mock_table1]

    # Mock settings.PDF_HEADER_FOOTER_THRESHOLD
    # Since they are imported inside setting/helper, let's patch Settings
    mock_fitz_doc = MagicMock()
    mock_fitz_open = MagicMock()
    mock_fitz_open.return_value.__enter__ = MagicMock(return_value=mock_fitz_doc)
    mock_fitz_open.return_value.__exit__ = MagicMock(return_value=False)

    with (
        patch("src.processing.pdf_parser.settings") as mock_settings,
        patch("src.processing.pdf_parser.DoclingPDFParser._get_converter", return_value=mock_converter),
        patch("src.processing.pdf_parser.fitz.open", mock_fitz_open),
    ):
        mock_settings.PDF_HEADER_FOOTER_THRESHOLD = 0.5
        mock_settings.PDF_NOISE_PATTERNS = [r"Page \d+"]

        res = parser.parse("dummy_path.pdf")

        assert res["page_count"] == 2
        assert res["table_count"] == 1
        assert len(res["tables"]) == 1
        assert res["tables"][0]["table_index"] == 0
        assert res["tables"][0]["page"] == 1
        assert res["tables"][0]["markdown"] == "| A | B |"
        assert res["tables"][0]["row_count"] == 2
        assert res["tables"][0]["col_count"] == 2


def test_cells_to_markdown_doc_type_gating():
    # general 모드에서는 표 셀에 legal 규칙(분류어 역순·숫자단위 결합)이 누수되지 않아야 한다.
    cells = [["구분", "제호 2"], ["값", "2 학기"]]
    legal = DoclingPDFParser._cells_to_markdown(cells, "legal")
    general = DoclingPDFParser._cells_to_markdown(cells, "general")
    assert "제2호" in legal and "2학기" in legal  # legal: 도메인 규칙 적용
    assert "제호 2" in general and "2 학기" in general  # general: 원형 보존


def test_clean_pdf_noise_and_remove_repeated_lines():
    parser = DoclingPDFParser()

    # Check fallback path when pages < 2
    res_single = parser._remove_repeated_lines("Only one page", threshold=0.5)
    assert "Only one page" in res_single

    # Test clean pdf noise
    with patch("src.processing.pdf_parser.settings") as mock_settings:
        mock_settings.PDF_NOISE_PATTERNS = [r"NoisePattern"]
        cleaned = parser._clean_pdf_noise("Some text NoisePattern and excess newlines\n\n\n\nmore text")
        assert "NoisePattern" not in cleaned
        assert "\n\n\n" not in cleaned
