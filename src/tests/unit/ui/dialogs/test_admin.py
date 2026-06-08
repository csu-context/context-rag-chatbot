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

        # Mock st.tabs to return context managers
        mock_tab1 = MagicMock()
        mock_tab2 = MagicMock()
        mock_st.tabs.return_value = (mock_tab1, mock_tab2)

        yield mock_st


class TestAdminDialog:
    def test_reset_admin_active(self, mock_st):
        reset_admin_active()
        assert mock_st.session_state.admin_active is False

    def test_show_admin_dialog_basic(self, mock_st):
        db_manager = MagicMock()
        init_cb = MagicMock()

        with patch("src.ui.dialogs.admin.requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"backups": []}

            with (
                patch("src.ui.dialogs.admin.PipelineOrchestrator") as mock_orch,
                patch("src.ui.dialogs.admin.RAW_DATA_DIR") as mock_raw_dir,
            ):
                mock_orch.return_value._load_manifest.return_value = {"files": {}}
                mock_raw_dir.glob.return_value = []
                mock_st.button.return_value = False

                func = getattr(show_admin_dialog, "__wrapped__", show_admin_dialog)
                func(db_manager, init_cb)

                mock_st.subheader.assert_called()
                mock_get.assert_called_with("http://localhost:8000/api/admin/backups")

    def test_show_admin_dialog_backup_and_restore(self, mock_st):
        db_manager = MagicMock()
        init_cb = MagicMock()

        with patch("src.ui.dialogs.admin.requests") as mock_requests:
            mock_requests.get.return_value.status_code = 200
            mock_requests.get.return_value.json.return_value = {"backups": [{"filename": "backup1.tar.gz"}]}
            mock_requests.post.return_value.status_code = 200

            def button_side_effect(label, key=None, **kwargs):
                if "수동 백업 실행" in label:
                    return True
                if key == "restore_backup1.tar.gz":
                    mock_st.session_state.get.return_value = True
                    return True
                return "예, 복원합니다" in label

            mock_st.button.side_effect = button_side_effect

            with (
                patch("src.ui.dialogs.admin.PipelineOrchestrator"),
                patch("src.ui.dialogs.admin.RAW_DATA_DIR") as mock_raw_dir,
            ):
                mock_raw_dir.glob.return_value = []
                func = getattr(show_admin_dialog, "__wrapped__", show_admin_dialog)
                func(db_manager, init_cb)

                mock_requests.post.assert_any_call("http://localhost:8000/api/admin/backups")
                mock_requests.post.assert_any_call(
                    "http://localhost:8000/api/admin/backups/restore?filename=backup1.tar.gz"
                )

    def test_render_sync_progress(self, mock_st):
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
