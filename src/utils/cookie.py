import logging

import streamlit as st

logger = logging.getLogger(__name__)

# 쿠키 설정 관련 표준 상수 정의
COOKIE_NAME = "st_session_id"
COOKIE_MAX_AGE_SEC = 604800  # 7일
COOKIE_SAME_SITE = "Lax"


def get_cookie_session_id() -> str | None:
    """Streamlit 1.57+ 공식 st.context.cookies API를 사용하여 세션 ID를 식별합니다."""
    try:
        cookies = st.context.cookies
        return cookies.get(COOKIE_NAME)
    except Exception as e:
        logger.debug(f"st.context.cookies 조회 실패 (테스트/CLI 환경 가능성): {e}")
        return None


def set_cookie_session_id(session_id: str) -> None:
    """Streamlit 공식 st.html(unsafe_allow_javascript=True) API를 사용하여 브라우저 쿠키에 고유 세션 ID를 기록합니다.

    st.components.v1.html은 1.56.0부터 공식적으로 deprecated 지정되었으므로, iframe 격리 없이
    메인 페이지 DOM에 직접 스크립트를 주입하여 쿠키를 설정할 수 있도록 st.html 방식으로 리팩토링하여 적용합니다.
    """
    try:
        cookie_value = f"{COOKIE_NAME}={session_id}; path=/; max-age={COOKIE_MAX_AGE_SEC}; SameSite={COOKIE_SAME_SITE}"

        js_script = f"""
        <script>
        (function() {{
            try {{
                // st.html은 iframe 외부 메인 문서에 직접 삽입되어 parent window 없이
                // document.cookie에 직접 기록이 가능합니다.
                document.cookie = "{cookie_value}";
            }} catch (e) {{}}
        }})();
        </script>
        """
        st.html(js_script, unsafe_allow_javascript=True)
        logger.debug("브라우저 쿠키에 세션 ID 설정 완료 (st.html 적용)")
    except Exception as e:
        logger.warning(f"쿠키 저장 주입 실패: {e}")
