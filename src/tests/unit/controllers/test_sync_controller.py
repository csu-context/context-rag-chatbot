from unittest.mock import MagicMock, patch

import pytest

from src.controllers.sync_controller import SyncController, SyncJobState


@pytest.fixture
def mock_st_session_state():
    with patch("streamlit.session_state", new_callable=dict) as mock_state:
        yield mock_state


class TestSyncJobState:
    def test_initial_state(self):
        job = SyncJobState()
        assert job.running is True
        assert job.current == 0
        assert job.total == 0
        assert job.file_name == ""
        assert job.percent == 0
        assert job.completed is False
        assert job.cancelled is False
        assert job.error is None
        assert not job.should_cancel

    def test_request_cancel(self):
        job = SyncJobState()
        job.request_cancel()
        assert job.should_cancel is True

    def test_update_progress(self):
        job = SyncJobState()
        job.update_progress(5, 10, "test.pdf")
        assert job.current == 5
        assert job.total == 10
        assert job.file_name == "test.pdf"
        assert job.percent == 50

        # Test edge case: total = 0
        job.update_progress(0, 0, "test.pdf")
        assert job.percent == 0

    def test_complete(self):
        job = SyncJobState()
        job.complete()
        assert job.running is False
        assert job.completed is True
        assert job.percent == 100

    def test_fail(self):
        job = SyncJobState()
        job.fail("Error occurred")
        assert job.running is False
        assert job.error == "Error occurred"

    def test_mark_cancelled(self):
        job = SyncJobState()
        job.mark_cancelled()
        assert job.running is False
        assert job.cancelled is True

    def test_snapshot(self):
        job = SyncJobState()
        job.update_progress(1, 2, "a.pdf")
        snapshot = job.snapshot()
        assert snapshot["running"] is True
        assert snapshot["current"] == 1
        assert snapshot["total"] == 2
        assert snapshot["file_name"] == "a.pdf"
        assert snapshot["percent"] == 50
        assert snapshot["completed"] is False
        assert snapshot["cancelled"] is False
        assert snapshot["error"] is None


class TestSyncController:
    @patch("src.controllers.sync_controller.threading.Thread")
    @patch("src.controllers.sync_controller.PipelineOrchestrator")
    @patch("src.controllers.sync_controller.st")
    def test_trigger_sync_background(self, mock_st, mock_orchestrator, mock_thread):
        # Setup session state
        mock_st.session_state = MagicMock()
        mock_st.session_state.get.return_value = None

        # Call background sync
        job = SyncController.trigger_sync_background("docling", force=True)

        # Verify job is created and saved
        assert isinstance(job, SyncJobState)
        assert mock_st.session_state._current_sync_job == job

        # Verify thread started
        mock_thread.assert_called_once()
        thread_instance = mock_thread.return_value
        thread_instance.start.assert_called_once()

        # If it's already running, calling again should return the same job
        mock_st.session_state.get.return_value = job
        job2 = SyncController.trigger_sync_background("docling", force=True)
        assert job2 is job

        # Test background thread execution
        run_func = mock_thread.call_args[1]["target"]
        args = mock_thread.call_args[1]["args"]

        with (
            patch("src.utils.health_check.run_full_diagnostics") as mock_diag,
            patch("src.utils.health_check.repair_integrity") as mock_repair,
        ):
            mock_diag.return_value = (True, {"db": {"anomalies": {"missing": True}}})

            # Execute run_func simulating thread run
            run_func(*args)

            mock_diag.assert_called_once_with(silent=True, check_model=False)
            mock_repair.assert_called_once_with({"missing": True}, target_parser="docling")

            # Ensure orchestrator is called
            mock_orchestrator.return_value.run_ingestion.assert_called_once()
            assert args[0].completed is True

    @patch("src.controllers.sync_controller.logger")
    @patch("src.controllers.sync_controller.threading.Thread")
    @patch("src.controllers.sync_controller.PipelineOrchestrator")
    @patch("src.controllers.sync_controller.st")
    def test_background_failure_logs_stacktrace(self, mock_st, mock_orchestrator, mock_thread, mock_logger):
        """백그라운드 동기화 실패 시 logger.error(exc_info=True)로 스택트레이스를 남기고 job에 기록한다. (#201)"""
        mock_st.session_state = MagicMock()
        mock_st.session_state.get.return_value = None
        mock_orchestrator.return_value.run_ingestion.side_effect = RuntimeError("ingestion boom")

        job = SyncController.trigger_sync_background("docling", force=True)
        run_func = mock_thread.call_args[1]["target"]
        args = mock_thread.call_args[1]["args"]

        with (
            patch("src.utils.health_check.run_full_diagnostics") as mock_diag,
            patch("src.utils.health_check.repair_integrity"),
        ):
            mock_diag.return_value = (True, {})
            run_func(*args)

        mock_logger.error.assert_called_once()
        assert mock_logger.error.call_args[1]["exc_info"] is True
        assert job.error == "ingestion boom"
        assert job.running is False
        assert job.completed is False

    @patch("src.controllers.sync_controller.PipelineOrchestrator")
    @patch("src.controllers.sync_controller.st")
    def test_trigger_sync(self, mock_st, mock_orchestrator):
        mock_status = MagicMock()
        mock_st.status.return_value.__enter__.return_value = mock_status
        mock_progress_bar = MagicMock()
        mock_st.progress.return_value = mock_progress_bar

        with (
            patch("src.utils.health_check.run_full_diagnostics") as mock_diag,
            patch("src.utils.health_check.repair_integrity"),
            patch("src.controllers.sync_controller.time.sleep"),
        ):
            mock_diag.return_value = (True, {})

            mock_clear_cache = MagicMock()
            SyncController.trigger_sync("docling", force=False, clear_cache_callback=mock_clear_cache)

            # Test orchestrator
            mock_orchestrator.return_value.run_ingestion.assert_called_once()
            kwargs = mock_orchestrator.return_value.run_ingestion.call_args[1]
            assert kwargs["force"] is False
            assert kwargs["parser_type"] == "docling"

            # Callback test
            cb = kwargs["progress_callback"]
            cb(1, 2, "test.pdf", mock_progress_bar, mock_status)
            mock_progress_bar.progress.assert_called_with(50, text="[1/2] test.pdf 처리 중... (50%)")

            cb(0, 0, "empty", mock_progress_bar, mock_status)
            mock_progress_bar.progress.assert_called_with(0, text="대기 중...")

            # End checks
            mock_clear_cache.assert_called_once()
            mock_st.success.assert_called_once()
            mock_st.rerun.assert_called_once()

    @patch("src.controllers.sync_controller.PipelineOrchestrator")
    @patch("src.controllers.sync_controller.st")
    def test_trigger_single_file_sync(self, mock_st, mock_orchestrator):
        mock_status = MagicMock()
        mock_st.status.return_value.__enter__.return_value = mock_status
        mock_progress_bar = MagicMock()
        mock_st.progress.return_value = mock_progress_bar

        with patch("src.controllers.sync_controller.time.sleep"):
            mock_clear_cache = MagicMock()
            SyncController.trigger_single_file_sync("test.pdf", "docling", clear_cache_callback=mock_clear_cache)

            mock_orchestrator.return_value.update_file_parser.assert_called_once()
            kwargs = mock_orchestrator.return_value.update_file_parser.call_args[1]
            cb = kwargs["progress_callback"]

            cb(1, 1, "test.pdf", mock_progress_bar, mock_status)
            mock_progress_bar.progress.assert_called_with(100, text="[1/1] test.pdf 처리 중... (100%)")

            mock_clear_cache.assert_called_once()
            mock_st.rerun.assert_called_once()

    @patch("src.controllers.sync_controller.PipelineOrchestrator")
    @patch("src.controllers.sync_controller.st")
    def test_trigger_multiple_files_sync(self, mock_st, mock_orchestrator):
        mock_status = MagicMock()
        mock_st.status.return_value.__enter__.return_value = mock_status
        mock_progress_bar = MagicMock()
        mock_st.progress.return_value = mock_progress_bar

        with patch("src.controllers.sync_controller.time.sleep"):
            mock_clear_cache = MagicMock()
            pending_copy = {"file1.pdf": "docling"}
            SyncController.trigger_multiple_files_sync(pending_copy, clear_cache_callback=mock_clear_cache)

            mock_orchestrator.return_value.update_multiple_file_parsers.assert_called_once()
            kwargs = mock_orchestrator.return_value.update_multiple_file_parsers.call_args[1]
            cb = kwargs["progress_callback"]

            cb(1, 1, "file1.pdf", mock_progress_bar, mock_status)
            mock_progress_bar.progress.assert_called_with(100, text="[1/1] file1.pdf 처리 중... (100%)")

            mock_clear_cache.assert_called_once()
            mock_st.rerun.assert_called_once()
