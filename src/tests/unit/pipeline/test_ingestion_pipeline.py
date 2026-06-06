import json
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.ingestion import IngestionPipeline


@pytest.fixture
def mock_pipeline_setup(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()
    yield raw_dir, processed_dir


def test_ingestion_pipeline_scan_files(mock_pipeline_setup):
    raw_dir, processed_dir = mock_pipeline_setup
    (raw_dir / "doc1.pdf").write_bytes(b"pdf data")
    (raw_dir / "doc2.md").write_text("md data")
    (raw_dir / "._hidden.pdf").write_bytes(b"hidden")
    (raw_dir / "sub").mkdir()
    (raw_dir / "sub" / "doc3.pdf").write_bytes(b"sub pdf")

    mock_strategy = MagicMock()
    with patch("src.pipeline.ingestion.ChromaDBManager"):
        pipeline = IngestionPipeline(strategy=mock_strategy, raw_dir=raw_dir, processed_dir=processed_dir)
        files = pipeline.scan_files()
        assert len(files) == 3
        basenames = [f.name for f in files]
        assert "doc1.pdf" in basenames
        assert "doc2.md" in basenames
        assert "doc3.pdf" in basenames
        assert "._hidden.pdf" not in basenames


def test_ingestion_pipeline_process_and_chunk_single(mock_pipeline_setup):
    raw_dir, processed_dir = mock_pipeline_setup
    pdf_file = raw_dir / "test.pdf"
    pdf_file.write_bytes(b"pdf data")

    mock_strategy = MagicMock()
    # parse returns sections
    mock_sections = [
        {
            "chapter": "제1장",
            "article": "제1조",
            "content": "이것은 본문입니다. " * 30,
            "metadata": {"src_name": "test.pdf", "source_id": "test_id", "doc_type": "pdf"},
        }
    ]
    mock_strategy.parse.return_value = mock_sections

    with (
        patch("src.pipeline.ingestion.ChromaDBManager"),
        patch("src.pipeline.ingestion.ParserFactory.create", return_value=mock_strategy),
        patch("src.pipeline.ingestion.RAW_DATA_DIR", raw_dir),
        patch("src.data.parser.RAW_DATA_DIR", raw_dir),
        patch("src.utils.file_utils.RAW_DATA_DIR", raw_dir),
        patch("src.pipeline.strategies.RAW_DATA_DIR", raw_dir),
        patch("src.utils.paths.RAW_DATA_DIR", raw_dir),
    ):
        pipeline = IngestionPipeline(strategy=mock_strategy, raw_dir=raw_dir, processed_dir=processed_dir)
        result = pipeline.process_and_chunk([pdf_file])
        assert len(result) == 1
        assert result[0]["parent_text"].startswith("이것은 본문입니다.")
        assert len(result[0]["children"]) > 0


def test_ingestion_pipeline_save_and_upsert(mock_pipeline_setup):
    raw_dir, processed_dir = mock_pipeline_setup

    data = [
        {
            "parent_id": "p1",
            "parent_text": "parent text",
            "metadata": {"source_id": "sid_1", "src_name": "test.pdf"},
            "children": [
                {"chunk_id": "p1_c1", "text": "child text", "metadata": {"source_id": "sid_1", "src_name": "test.pdf"}}
            ],
        }
    ]

    mock_db = MagicMock()
    with patch("src.pipeline.ingestion.ChromaDBManager", return_value=mock_db):
        pipeline = IngestionPipeline(strategy=MagicMock(), raw_dir=raw_dir, processed_dir=processed_dir)
        pipeline.upsert_to_db(data)
        mock_db.upsert_documents.assert_called_once()

        saved = pipeline.save_processed_data(data)
        assert len(saved) == 1
        assert (processed_dir / "sid_1.json").exists()


def test_ingestion_pipeline_cleanup(mock_pipeline_setup):
    raw_dir, processed_dir = mock_pipeline_setup

    # Pre-populate processed file
    sid_file = processed_dir / "sid_1.json"
    sid_file.write_text(json.dumps([{"metadata": {"relative_path": "rules.pdf"}}]))

    mock_db = MagicMock()
    with patch("src.pipeline.ingestion.ChromaDBManager", return_value=mock_db):
        pipeline = IngestionPipeline(strategy=MagicMock(), raw_dir=raw_dir, processed_dir=processed_dir)

        # mock db query
        mock_db.collection.get.return_value = {"metadatas": [{"source_id": "sid_1"}]}

        pipeline.cleanup_db(filenames_to_delete=["rules.pdf"], relative_paths_to_delete=["rules.pdf"])

        # Verify db delete and physical delete
        mock_db.delete_documents.assert_called()
        assert not sid_file.exists()
