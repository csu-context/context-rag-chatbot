from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.orchestrator import PipelineOrchestrator


@pytest.fixture
def orchestrator():
    # Reset singleton
    PipelineOrchestrator._instance = None
    with (
        patch("src.pipeline.orchestrator.IngestionPipeline"),
        patch("src.pipeline.orchestrator.StorageManager"),
        patch("src.pipeline.orchestrator.SemanticCache"),
    ):
        orch = PipelineOrchestrator()
        orch.ingestion_pipeline = MagicMock()
        orch.storage_manager = MagicMock()
        orch.cache = MagicMock()
        yield orch


class TestPipelineOrchestratorExt:
    def test_reset_all_data(self, orchestrator):
        orchestrator.storage_manager.scan_processed_files.return_value = [Path("file1.json")]

        with patch("src.pipeline.orchestrator.CACHE_DIR") as mock_cache_dir:
            mock_cache_dir.glob.return_value = [Path("file1_parsed.pkl")]
            orchestrator._reset_all_data()

            orchestrator.ingestion_pipeline.db_manager.reset_collection.assert_called_once()
            assert orchestrator.storage_manager.delete_file.call_count == 2

    def test_update_parser_if_changed(self, orchestrator):
        orchestrator.parser_type = "manual"

        with patch.object(orchestrator, "_get_parser_strategy") as mock_get_strategy:
            mock_get_strategy.return_value = "docling_strategy"
            orchestrator._update_parser_if_changed("docling")

            assert orchestrator.parser_type == "docling"
            assert orchestrator.strategy == "docling_strategy"
            assert orchestrator.ingestion_pipeline.strategy == "docling_strategy"

    def test_update_file_parser(self, orchestrator):
        with (
            patch.object(orchestrator, "_load_manifest") as mock_load,
            patch.object(orchestrator, "_save_manifest") as mock_save,
            patch.object(orchestrator, "_find_relative_path") as mock_find,
            patch.object(orchestrator, "run_ingestion") as mock_run,
        ):
            mock_load.return_value = {"files": {"test.pdf": {"hash": "h1"}}}
            mock_find.return_value = "test.pdf"

            orchestrator.update_file_parser("test.pdf", "docling")

            orchestrator.ingestion_pipeline.cleanup_db.assert_called_once_with(
                source_ids_to_delete=["h1"], filenames_to_delete=["test.pdf"]
            )

            mock_save.assert_called_once()
            saved_manifest = mock_save.call_args[0][0]
            assert saved_manifest["files"]["test.pdf"]["parser_type"] == "docling"
            assert saved_manifest["files"]["test.pdf"]["hash"] == ""

            mock_run.assert_called_once()

    def test_update_multiple_file_parsers(self, orchestrator):
        with (
            patch.object(orchestrator, "_load_manifest") as mock_load,
            patch.object(orchestrator, "_save_manifest") as mock_save,
            patch.object(orchestrator, "_find_relative_path") as mock_find,
            patch.object(orchestrator, "run_ingestion") as mock_run,
        ):
            mock_load.return_value = {"files": {"test1.pdf": {"hash": "h1"}, "test2.pdf": {"hash": "h2"}}}
            mock_find.side_effect = ["test1.pdf", "test2.pdf", "test1.pdf", "test2.pdf"]

            orchestrator.update_multiple_file_parsers({"test1.pdf": "docling", "test2.pdf": "manual"})

            orchestrator.ingestion_pipeline.cleanup_db.assert_called_once_with(
                source_ids_to_delete=["h1", "h2"], filenames_to_delete=["test1.pdf", "test2.pdf"]
            )

            mock_save.assert_called_once()
            saved_manifest = mock_save.call_args[0][0]
            assert saved_manifest["files"]["test1.pdf"]["parser_type"] == "docling"
            assert saved_manifest["files"]["test2.pdf"]["parser_type"] == "manual"

            mock_run.assert_called_once()

    def test_process_single_file(self, orchestrator):
        with (
            patch("src.pipeline.orchestrator.Path.exists", return_value=True),
            patch.object(orchestrator, "_process_changes") as mock_process,
            patch.object(orchestrator, "_load_manifest") as mock_load,
            patch.object(orchestrator, "_save_manifest") as mock_save,
            patch("src.pipeline.orchestrator.generate_file_hash") as mock_hash,
        ):
            mock_load.return_value = {"files": {}}
            mock_hash.return_value = "new_hash"

            # Use a path that is strictly relative to RAW_DATA_DIR to avoid ValueError
            from src.utils.paths import RAW_DATA_DIR

            test_path = RAW_DATA_DIR / "test.pdf"

            res = orchestrator.process_single_file(test_path)

            assert res is True
            mock_process.assert_called_once()
            mock_save.assert_called_once()
            saved_manifest = mock_save.call_args[0][0]
            assert "test.pdf" in saved_manifest["files"]
            assert saved_manifest["files"]["test.pdf"]["hash"] == "new_hash"

    def test_process_single_file_not_exists(self, orchestrator):
        with patch("src.pipeline.orchestrator.Path.exists", return_value=False):
            res = orchestrator.process_single_file(Path("fake.pdf"))
            assert res is False

    def test_cleanup_db(self, orchestrator):
        session = MagicMock()
        with patch.object(orchestrator, "_load_manifest") as mock_load:
            mock_load.return_value = {"files": {"test.pdf": {"hash": "h1"}}}

            orchestrator._cleanup_db(session, ["h1"], ["test.pdf"])

            orchestrator.ingestion_pipeline.cleanup_db.assert_called_once_with(
                source_ids_to_delete=["h1"], relative_paths_to_delete=["test.pdf"], filenames_to_delete=["test.pdf"]
            )
