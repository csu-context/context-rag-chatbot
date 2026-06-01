from pathlib import Path
from unittest.mock import MagicMock, patch

from src.common.config import settings
from src.pipeline.strategies import (
    DoclingPDFParserStrategy,
    ManualParserStrategy,
    MarkdownParserStrategy,
    _safe_relative_to,
)


def test_safe_relative_to():
    base = Path("/workspace/data/raw")
    path_inside = Path("/workspace/data/raw/sub/file.txt")
    path_outside = Path("/tmp/outside.txt")

    assert _safe_relative_to(path_inside, base) == Path("sub/file.txt")
    assert _safe_relative_to(path_outside, base) == path_outside


def test_manual_parser_strategy():
    strategy = ManualParserStrategy()
    mock_parser = MagicMock()
    mock_parser.parse.return_value = [{"content": "manual parsed"}]

    with (
        patch("src.pipeline.strategies.RAW_DATA_DIR", Path("/raw")),
        patch("src.pipeline.strategies.ManualParser", return_value=mock_parser) as mock_class,
    ):
        res = strategy.parse(Path("/raw/test.pdf"))
        mock_class.assert_called_once_with("test.pdf", parser_type="manual", doc_type=settings.DOC_TYPE)
        assert res == [{"content": "manual parsed"}]


def test_markdown_parser_strategy(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("# Heading\nSome text content here.", encoding="utf-8")

    strategy = MarkdownParserStrategy()
    with (
        patch("src.pipeline.strategies.RAW_DATA_DIR", tmp_path),
        patch("src.pipeline.strategies.generate_file_hash", return_value="hash123"),
    ):
        res = strategy.parse(md_file)
        assert len(res) == 1
        assert res[0]["is_raw_markdown"] is True
        assert "Some text content here." in res[0]["content"]
        assert res[0]["metadata"]["source_id"] == "hash123"


def test_docling_pdf_parser_strategy(tmp_path):
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4...")

    mock_pdf_parser = MagicMock()
    # Mock docling return
    mock_pdf_parser.parse.return_value = {
        "table_count": 1,
        "tables": [{"page": 1, "table_index": 0, "markdown": "| A | B |\n|---|---|\n| 1 | 2 |"}],
    }

    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 1
    mock_page = MagicMock()
    mock_page.get_text.side_effect = lambda mode, **kw: (
        {"blocks": []} if mode == "rawdict" else "Page Header\nSome regular non-table text\n"
    )
    mock_page.find_tables.return_value.tables = []
    mock_doc.__getitem__.return_value = mock_page

    with (
        patch("src.pipeline.strategies.DoclingPDFParser") as mock_pdf_parser_class,
        patch("src.pipeline.strategies.RAW_DATA_DIR", tmp_path),
        patch("src.pipeline.strategies.generate_file_hash", return_value="hash_docling"),
        patch("src.pipeline.strategies.ManualParser") as mock_manual_parser_class,
        patch("fitz.open", return_value=mock_doc),
    ):
        mock_pdf_parser_class.return_value = mock_pdf_parser
        mock_pdf_parser_class.build_table_content.return_value = "Page Header\n\n| A | B |\n|---|---|\n| 1 | 2 |"
        # mock manual parser return
        mock_manual_inst = MagicMock()
        mock_manual_inst.parse.return_value = [{"content": "manual text", "metadata": {}}]
        mock_manual_parser_class.return_value = mock_manual_inst
        mock_manual_parser_class.get_table_bboxes.return_value = []
        mock_manual_parser_class.clean_text.return_value = ""

        strategy = DoclingPDFParserStrategy()
        res = strategy.parse(pdf_file)

        # Docling parser should have been called
        mock_pdf_parser.parse.assert_called_once_with(pdf_file)
        # Verify both manual text and docling tables are in the result
        assert len(res) == 2
        assert res[0]["content"] == "manual text"
        assert "A | B" in res[1]["content"]
