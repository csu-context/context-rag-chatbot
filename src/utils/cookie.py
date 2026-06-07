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
    """Streamlit HTML/JS 컴포넌트 주입을 통해 브라우저 쿠키에 고유 세션 ID를 생성/갱신합니다.

    브라우저의 iframe 타사 쿠키 제약 정책을 우회하기 위해 parent window 및 직접 주입 방식을 결합해 적용합니다.
    """
    try:
        cookie_value = f"{COOKIE_NAME}={session_id}; path=/; max-age={COOKIE_MAX_AGE_SEC}; SameSite={COOKIE_SAME_SITE}"

        js_script = f"""
        <script>
        (function() {{
            try {{
                if (window.parent && window.parent.document) {{
                    window.parent.document.cookie = "{cookie_value}";
                }}
            }} catch (e) {{}}
            document.cookie = "{cookie_value}";
        }})();
        </script>
        """
        st.components.v1.html(js_script, height=0)
        logger.debug("브라우저 쿠키에 세션 ID 설정 완료")
    except Exception as e:
        logger.warning(f"쿠키 저장 주입 실패: {e}")
