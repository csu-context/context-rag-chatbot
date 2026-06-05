import logging

import streamlit as st

logger = logging.getLogger(__name__)


def get_cookie_session_id() -> str | None:
    """Streamlit WebSocket 헤더 정보에서 브라우저 쿠키를 읽어 세션 ID를 식별합니다."""
    try:
        from streamlit.web.server.websocket_headers import _get_websocket_headers
    except ImportError:
        logger.debug("streamlit websocket_headers 모듈 임포트 실패 (테스트/CLI 환경)")
        return None

    if _get_websocket_headers is None:
        return None

    try:
        headers = _get_websocket_headers()
    except Exception as e:
        logger.debug(f"웹소켓 헤더 조회 실패: {e}")
        return None

    if not headers:
        return None

    cookie_header = headers.get("Cookie", "")
    if not cookie_header:
        return None

    for cookie in cookie_header.split(";"):
        cookie = cookie.strip()
        if cookie.startswith("st_session_id="):
            try:
                val = cookie.split("=", 1)[1]
                if val:
                    return val
            except IndexError:
                pass
    return None


def set_cookie_session_id(session_id: str) -> None:
    """HTML/JS 컴포넌트 주입을 통해 브라우저 쿠키에 고유 세션 ID를 생성/갱신합니다.

    유효기간은 7일로 설정합니다.
    """
    js_script = f"""
    <script>
    document.cookie = "st_session_id={session_id}; path=/; max-age=604800; SameSite=Lax";
    </script>
    """
    st.components.v1.html(js_script, height=0)
    logger.debug(f"브라우저 쿠키에 세션 ID 설정: {session_id}")
