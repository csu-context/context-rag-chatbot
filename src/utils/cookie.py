import logging

from streamlit_cookies_controller import CookieController

logger = logging.getLogger(__name__)


def get_cookie_session_id() -> str | None:
    """streamlit-cookies-controller를 사용하여 브라우저 쿠키의 세션 ID를 식별합니다."""
    try:
        controller = CookieController()
        return controller.get("st_session_id")
    except Exception as e:
        logger.debug(f"CookieController 쿠키 조회 실패 (테스트/CLI 환경 가능성): {e}")
        return None


def set_cookie_session_id(session_id: str) -> None:
    """streamlit-cookies-controller를 사용하여 브라우저 쿠키에 고유 세션 ID를 생성/갱신합니다.

    유효기간은 7일(604800초)로 설정합니다.
    """
    try:
        controller = CookieController()
        controller.set("st_session_id", session_id, max_age=604800, same_site="lax")
        logger.debug("브라우저 쿠키에 세션 ID 설정 완료")
    except Exception as e:
        logger.warning(f"CookieController 쿠키 저장 실패: {e}")
