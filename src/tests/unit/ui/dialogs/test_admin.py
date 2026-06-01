from unittest.mock import MagicMock, patch

import pytest

from src.ui.dialogs.admin import _render_sync_progress, reset_admin_active, show_admin_dialog


@pytest.fixture
def mock_st():
    with patch("src.ui.dialogs.admin.st") as mock_st:
        mock_st.session_state = MagicMock()
        mock_st.session_state.admin_active = True
        mock_st.session_state.get.return_value = None

        def mock_dialog(title, width=None, on_dismiss=None):
            def wrapper(func):
                return func

            return wrapper

        def mock_fragment(run_every=None):
            def wrapper(func):
                return func

            return wrapper

        def mock_columns(spec):
            cols = []
            spec_iter = range(spec) if isinstance(spec, int) else spec
            for _ in spec_iter:
                m = MagicMock()
                m.button.return_value = False
                cols.append(m)
            return cols

        mock_st.dialog = mock_dialog
        mock_st.fragment = mock_fragment
        mock_st.columns.side_effect = mock_columns

        yield mock_st


class TestAdminDialog:
    def test_reset_admin_active(self, mock_st):
        reset_admin_active()
        assert mock_st.session_state.admin_active is False

    def test_show_admin_dialog_basic(self, mock_st):
        db_manager = MagicMock()
        init_cb = MagicMock()

        with (
            patch("src.ui.dialogs.admin.PipelineOrchestrator") as mock_orch,
            patch("src.ui.dialogs.admin.RAW_DATA_DIR") as mock_raw_dir,
        ):
            mock_orch.return_value._load_manifest.return_value = {"files": {}}

            mock_file = MagicMock()
            mock_file.name = "test.pdf"
            mock_file.relative_to.return_value = "test.pdf"
            mock_file.stat.return_value.st_size = 1000

            mock_raw_dir.glob.return_value = [mock_file]

            # Button states: all False
            mock_st.button.return_value = False

            # Use __wrapped__ if available, otherwise just call (if it's already unwrapped by some chance)
            func = getattr(show_admin_dialog, "__wrapped__", show_admin_dialog)
            func(db_manager, init_cb)

            mock_st.subheader.assert_called()
            mock_st.file_uploader.assert_called()

    def test_show_admin_dialog_buttons(self, mock_st):
        db_manager = MagicMock()
        init_cb = MagicMock()

        with (
            patch("src.ui.dialogs.admin.PipelineOrchestrator") as mock_orch,
            patch("src.ui.dialogs.admin.RAW_DATA_DIR") as mock_raw_dir,
            patch("src.ui.dialogs.admin.run_full_diagnostics") as mock_diag,
            patch("src.ui.dialogs.admin.repair_integrity"),
            patch("src.ui.dialogs.admin.SyncController"),
        ):
            mock_orch.return_value._load_manifest.return_value = {
                "files": {"test.pdf": {"hash": "1", "parser_type": "docling"}}
            }

            mock_file = MagicMock()
            mock_file.name = "test.pdf"
            mock_file.relative_to.return_value = "test.pdf"
            mock_file.exists.return_value = True
            mock_file.stat.return_value.st_size = 1000

            mock_raw_dir.glob.return_value = [mock_file]
            db_manager.get_source_chunks.return_value = [{"id": "1"}]

            # Simulate clicking all buttons to hit coverage
            def mock_button(*args, **kwargs):
                return True

            mock_st.button.side_effect = mock_button
            mock_st.file_uploader.return_value = None

            mock_diag.return_value = (True, {"db": {"anomalies": {"ghost_chunks": ["file"]}}})

            # Set state for diagnostic report
            mock_st.session_state.health_report = {"db": {"anomalies": {"ghost_chunks": ["file"]}}}

            func = getattr(show_admin_dialog, "__wrapped__", show_admin_dialog)
            func(db_manager, init_cb)

    def test_show_admin_dialog_upload_files(self, mock_st, tmp_path):
        db_manager = MagicMock()
        init_cb = MagicMock()

        with (
            patch("src.ui.dialogs.admin.PipelineOrchestrator") as mock_orch,
            patch("src.ui.dialogs.admin.RAW_DATA_DIR", tmp_path),
            patch("src.ui.dialogs.admin.SyncController") as mock_sync,
        ):
            mock_orch.return_value._load_manifest.return_value = {"files": {}}

            # Button returns True
            def mock_button(*args, **kwargs):
                return True

            mock_st.button.side_effect = mock_button

            # Setup file uploader
            f1 = MagicMock()
            f1.name = "test.pdf"
            f1.size = 1000
            f1.getbuffer.return_value = b"test content"

            f2 = MagicMock()
            f2.name = "invalid.pdf"
            f2.size = 0

            mock_st.file_uploader.return_value = [f1, f2]

            # Setup parser type
            mock_st.radio.return_value = "docling"

            # Setup auto_sync True
            mock_st.checkbox.return_value = True

            func = getattr(show_admin_dialog, "__wrapped__", show_admin_dialog)
            func(db_manager, init_cb)

            # Since auto_sync is True, it should trigger sync background
            mock_sync.trigger_sync_background.assert_called()

            # Verify file was written
            assert (tmp_path / "test.pdf").exists()
            assert (tmp_path / "test.pdf").read_bytes() == b"test content"

    def test_show_admin_dialog_empty_files(self, mock_st):
        db_manager = MagicMock()
        init_cb = MagicMock()

        with (
            patch("src.ui.dialogs.admin.PipelineOrchestrator") as mock_orch,
            patch("src.ui.dialogs.admin.RAW_DATA_DIR") as mock_raw_dir,
            patch("src.ui.dialogs.admin.SyncController"),
        ):
            mock_orch.return_value._load_manifest.return_value = {"files": {}}
            mock_raw_dir.glob.return_value = []

            # Upload empty list
            mock_st.button.return_value = True
            mock_st.file_uploader.return_value = []

            func = getattr(show_admin_dialog, "__wrapped__", show_admin_dialog)
            func(db_manager, init_cb)

    def test_show_admin_dialog_parser_change(self, mock_st):
        db_manager = MagicMock()
        init_cb = MagicMock()

        with (
            patch("src.ui.dialogs.admin.PipelineOrchestrator") as mock_orch,
            patch("src.ui.dialogs.admin.RAW_DATA_DIR") as mock_raw_dir,
            patch("src.ui.dialogs.admin.SyncController") as mock_sync,
        ):
            mock_orch.return_value._load_manifest.return_value = {
                "files": {"test.pdf": {"hash": "1", "parser_type": "old_parser"}}
            }

            mock_file = MagicMock()
            mock_file.name = "test.pdf"
            mock_file.exists.return_value = True
            mock_file.stat.return_value.st_size = 1000
            mock_raw_dir.glob.return_value = [mock_file]

            # Setup pending change
            mock_st.session_state.get.side_effect = lambda k, d=None: (
                {"test.pdf": "new_parser"} if k == "parser_change_pending" else d
            )
            mock_st.button.return_value = True

            # This should hit "예, 변경 및 재색인 실행" button
            func = getattr(show_admin_dialog, "__wrapped__", show_admin_dialog)
            func(db_manager, init_cb)

            # SyncController should be called for the pending file
            mock_sync.trigger_multiple_files_sync.assert_called()

    def test_show_admin_dialog_file_actions(self, mock_st):
        db_manager = MagicMock()
        init_cb = MagicMock()

        with (
            patch("src.ui.dialogs.admin.PipelineOrchestrator") as mock_orch,
            patch("src.ui.dialogs.admin.RAW_DATA_DIR") as mock_raw_dir,
            patch("src.ui.dialogs.admin.SyncController") as mock_sync,
        ):
            mock_orch.return_value._load_manifest.return_value = {
                "files": {"test.pdf": {"hash": "1", "parser_type": "docling"}}
            }

            mock_file = MagicMock()
            mock_file.name = "test.pdf"
            mock_file.exists.return_value = True
            mock_file.stat.return_value.st_size = 1000

            # Return file only for pdf to prevent duplicate hits in loop
            mock_raw_dir.glob.side_effect = lambda ext: [mock_file] if "pdf" in ext else []

            db_manager.get_source_chunks.return_value = [{"id": "1"}]

            # Simulate column buttons returning True to test sync and delete
            def mock_columns(spec):
                cols = []
                spec_iter = range(spec) if isinstance(spec, int) else spec
                for _ in spec_iter:
                    m = MagicMock()
                    m.button.return_value = True  # Return True for r_col7 (sync) and r_col8 (delete)
                    cols.append(m)
                return cols

            mock_st.columns.side_effect = mock_columns

            # Only trigger sync button to test pending clear
            mock_st.session_state.parser_change_pending = {"test.pdf": "docling"}

            func = getattr(show_admin_dialog, "__wrapped__", show_admin_dialog)
            func(db_manager, init_cb)

            mock_sync.trigger_single_file_sync.assert_called()
            mock_file.unlink.assert_called_once()

    def test_show_admin_dialog_health_report(self, mock_st):
        db_manager = MagicMock()
        init_cb = MagicMock()

        with (
            patch("src.ui.dialogs.admin.PipelineOrchestrator") as mock_orch,
            patch("src.ui.dialogs.admin.RAW_DATA_DIR") as mock_raw_dir,
            patch("src.ui.dialogs.admin.SyncController"),
        ):
            mock_orch.return_value._load_manifest.return_value = {"files": {}}
            mock_raw_dir.glob.return_value = []

            mock_st.session_state = MagicMock()
            mock_st.session_state.__contains__.return_value = True
            mock_st.session_state.get.return_value = None
            mock_st.session_state.health_report = {
                "db": {
                    "anomalies": {
                        "ghost_chunks": ["ghost1"],
                        "mismatched_hash": ["mismatch1"],
                        "duplicate_parsers": ["dup1"],
                    }
                }
            }
            mock_st.button.return_value = False

            func = getattr(show_admin_dialog, "__wrapped__", show_admin_dialog)
            func(db_manager, init_cb)

            mock_st.error.assert_any_call("유령 청크 감지: 1개 파일의 데이터가 DB에 남아있습니다.")

    def test_show_admin_dialog_health_repair(self, mock_st):
        db_manager = MagicMock()
        init_cb = MagicMock()

        with (
            patch("src.ui.dialogs.admin.PipelineOrchestrator") as mock_orch,
            patch("src.ui.dialogs.admin.RAW_DATA_DIR") as mock_raw_dir,
            patch("src.ui.dialogs.admin.repair_integrity") as mock_repair,
            patch("src.ui.dialogs.admin.SyncController"),
        ):
            mock_orch.return_value._load_manifest.return_value = {"files": {}}
            mock_raw_dir.glob.return_value = []

            mock_st.session_state = MagicMock()
            mock_st.session_state.__contains__.return_value = True
            mock_st.session_state.get.return_value = None
            mock_st.session_state.health_report = {"db": {"anomalies": {"ghost_chunks": ["ghost1"]}}}

            # simulate button click for Repair only
            def mock_button(*args, **kwargs):
                return "Repair" in args[0]

            mock_st.button.side_effect = mock_button

            func = getattr(show_admin_dialog, "__wrapped__", show_admin_dialog)
            func(db_manager, init_cb)

            mock_repair.assert_called_once()
            mock_st.success.assert_called_with("복구가 완료되었습니다. 상태를 재확인하세요.")

    def test_render_sync_progress(self, mock_st):
        # mock fragment decorator
        job = MagicMock()
        job.snapshot.return_value = {
            "running": True,
            "total": 10,
            "current": 5,
            "file_name": "test.pdf",
            "percent": 50,
            "completed": False,
            "cancelled": False,
            "error": None,
        }
        mock_st.session_state.get.return_value = job
        mock_st.button.return_value = True

        with patch("src.ui.dialogs.admin.time.sleep"):
            _render_sync_progress(MagicMock())
            job.request_cancel.assert_called_once()

            job.snapshot.return_value["running"] = False
            job.snapshot.return_value["completed"] = True
            cb = MagicMock()
            _render_sync_progress(cb)
            cb.assert_called_once()
