from unittest.mock import MagicMock, patch

import pytest

from src.ui.dialogs.admin import (
    API_URL,
    _render_sync_progress,
    _render_sync_result,
    reset_admin_active,
    show_admin_dialog,
)


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
                mock_get.assert_called_with(f"{API_URL}/api/admin/backups")

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

                mock_requests.post.assert_any_call(f"{API_URL}/api/admin/backups")
                mock_requests.post.assert_any_call(f"{API_URL}/api/admin/backups/restore?filename=backup1.tar.gz")

    def test_render_sync_progress_running_cancel(self, mock_st):
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

        _render_sync_progress()
        job.request_cancel.assert_called_once()

    def test_render_sync_progress_terminal_handoff(self, mock_st):
        """가동 종료 시 결과 패널로 1회 핸드오프(st.rerun)하고 자동으로 닫지 않는다. (#201)"""
        job = MagicMock()
        job.snapshot.return_value = {
            "running": False,
            "total": 10,
            "current": 10,
            "file_name": "test.pdf",
            "percent": 100,
            "completed": True,
            "cancelled": False,
            "error": None,
        }
        mock_st.session_state.get.return_value = job

        _render_sync_progress()

        mock_st.rerun.assert_called_once()
        # 자동 닫기(admin_active=False)·job 소거를 하지 않아 결과가 부모로 인계된다.
        assert mock_st.session_state.admin_active is True

    def test_render_sync_result_completed(self, mock_st):
        """완료 결과는 성공 메시지 + RAG 재초기화 1회. 확인 전까지 패널 유지. (#201)"""
        job = MagicMock()
        job.snapshot.return_value = {
            "running": False,
            "completed": True,
            "cancelled": False,
            "error": None,
            "file_name": "",
            "current": 1,
            "total": 1,
            "percent": 100,
        }
        cb = MagicMock()
        mock_st.button.return_value = False  # 확인 미클릭 → 유지

        _render_sync_result(job, cb)

        cb.assert_called_once()
        mock_st.success.assert_called_once()
        mock_st.rerun.assert_not_called()

    def test_render_sync_result_error_keeps_panel(self, mock_st):
        """오류 결과는 사유 + 마지막 처리 파일을 표시하고, 재초기화하지 않는다. (#201)"""
        job = MagicMock()
        job.snapshot.return_value = {
            "running": False,
            "completed": False,
            "cancelled": False,
            "error": "ingestion boom",
            "file_name": "broken.hwp",
            "current": 0,
            "total": 3,
            "percent": 0,
        }
        cb = MagicMock()
        mock_st.button.return_value = False

        _render_sync_result(job, cb)

        mock_st.error.assert_called_once()
        assert "ingestion boom" in mock_st.error.call_args[0][0]
        mock_st.caption.assert_called_once()
        cb.assert_not_called()

    def test_render_sync_result_ack_closes(self, mock_st):
        """'확인' 클릭 시 job을 비우고 rerun으로 결과 패널을 닫는다. (#201)"""
        job = MagicMock()
        job.snapshot.return_value = {
            "running": False,
            "completed": True,
            "cancelled": False,
            "error": None,
            "file_name": "",
            "current": 1,
            "total": 1,
            "percent": 100,
        }
        mock_st.button.return_value = True  # 확인 클릭

        _render_sync_result(job, MagicMock())

        assert mock_st.session_state._current_sync_job is None
        mock_st.rerun.assert_called_once()
