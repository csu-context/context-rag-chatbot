import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
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
        "Header text\n"
        "Some content\n"
        "Footer text\n"
        "<!-- page break -->\n"
        "Header text\n"
        "Page two content\n"
        "Footer text"
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
    with (
        patch("src.processing.pdf_parser.settings") as mock_settings,
        patch("src.processing.pdf_parser.DoclingPDFParser._get_converter", return_value=mock_converter)
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
