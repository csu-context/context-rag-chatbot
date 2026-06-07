from unittest.mock import patch

import streamlit as st

from src.ui.session import init_session_state
from src.utils.cookie import get_cookie_session_id, set_cookie_session_id


class TestSessionCookie:
    def test_get_cookie_session_id_success(self):
        """st_session_id가 존재할 때 정상 반환하는지 검증"""
        with patch("src.utils.cookie.CookieController") as mock_controller_cls:
            mock_controller = mock_controller_cls.return_value
            mock_controller.get.return_value = "test-uuid-1234"
            assert get_cookie_session_id() == "test-uuid-1234"

    def test_get_cookie_session_id_missing(self):
        """st_session_id가 없을 때 None을 반환하는지 검증"""
        with patch("src.utils.cookie.CookieController") as mock_controller_cls:
            mock_controller = mock_controller_cls.return_value
            mock_controller.get.return_value = None
            assert get_cookie_session_id() is None

    def test_get_cookie_session_id_exception(self):
        """예외 발생 시 None을 반환하는지 검증"""
        with patch("src.utils.cookie.CookieController", side_effect=Exception("error")):
            assert get_cookie_session_id() is None

    def test_set_cookie_session_id(self):
        """set_cookie_session_id 호출 시 set이 올바른 파라미터로 호출되는지 검증"""
        with patch("src.utils.cookie.CookieController") as mock_controller_cls:
            mock_controller = mock_controller_cls.return_value
            set_cookie_session_id("new-uuid-5678")
            mock_controller.set.assert_called_once_with(
                "st_session_id", "new-uuid-5678", max_age=604800, same_site="lax"
            )

    def test_init_session_state_uuid(self):
        """init_session_state 실행 시 session_uuid가 없는 경우 새로 생성하는지 검증"""
        # st.session_state 비우기
        for k in list(st.session_state.keys()):
            del st.session_state[k]

        assert "session_uuid" not in st.session_state
        init_session_state()
        assert "session_uuid" in st.session_state
        assert len(st.session_state["session_uuid"]) > 0
