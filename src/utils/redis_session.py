"""Issue 12: Redis 기반 세션 스토어 — Stateless 수평 확장 지원.

REDIS_URL 환경변수 설정 시 Redis에 세션 데이터를 저장합니다.
미설정 시 아무 동작도 하지 않아 기존 Streamlit in-memory 동작을 유지합니다.

사용법:
    # .env 또는 환경변수
    REDIS_URL=redis://localhost:6379/0

    # app.py 에서 메시지 저장 시
    RedisSessionStore.save_messages(session_id, messages)

    # 앱 시작 시 복원
    messages = RedisSessionStore.load_messages(session_id)
"""

import json
import logging
from typing import Any

from src.common.config import settings

logger = logging.getLogger(__name__)

_redis_client = None
_redis_available = False


def _get_client():
    global _redis_client, _redis_available
    if _redis_client is not None:
        return _redis_client if _redis_available else None

    if not settings.REDIS_URL:
        _redis_available = False
        return None

    try:
        import redis  # type: ignore[import-untyped]

        _redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        _redis_client.ping()
        _redis_available = True
        logger.info(f"Redis 세션 스토어 연결 성공: {settings.REDIS_URL}")
    except ImportError:
        logger.warning("redis 패키지 미설치 — pip install redis. in-memory 폴백 사용.")
        _redis_available = False
    except Exception as e:
        logger.warning(f"Redis 연결 실패 ({settings.REDIS_URL}): {e}. in-memory 폴백 사용.")
        _redis_client = None  # 다음 호출에서 재시도 허용
        _redis_available = False

    return _redis_client if _redis_available else None


class RedisSessionStore:
    """Streamlit 세션 데이터를 Redis에 저장/복원.

    수평 확장 시나리오: 로드밸런서가 동일 사용자를 다른 Streamlit 인스턴스로 라우팅해도
    Redis에서 세션을 복원하여 대화 연속성 유지.
    """

    @staticmethod
    def _key(session_id: str, field: str) -> str:
        return f"rag:session:{session_id}:{field}"

    @classmethod
    def save(cls, session_id: str, field: str, data: Any) -> bool:
        client = _get_client()
        if client is None:
            return False
        try:
            key = cls._key(session_id, field)
            client.setex(key, settings.REDIS_SESSION_TTL_SEC, json.dumps(data, ensure_ascii=False))
            return True
        except Exception as e:
            logger.debug(f"Redis 저장 실패 ({field}): {e}")
            return False

    @classmethod
    def load(cls, session_id: str, field: str, default: Any = None) -> Any:
        client = _get_client()
        if client is None:
            return default
        try:
            key = cls._key(session_id, field)
            raw = client.get(key)
            return json.loads(raw) if raw else default
        except Exception as e:
            logger.debug(f"Redis 로드 실패 ({field}): {e}")
            return default

    @classmethod
    def save_messages(cls, session_id: str, messages: list[dict]) -> bool:
        return cls.save(session_id, "messages", messages)

    @classmethod
    def load_messages(cls, session_id: str) -> list[dict]:
        return cls.load(session_id, "messages", default=[])

    @classmethod
    def delete_session(cls, session_id: str) -> None:
        client = _get_client()
        if client is None:
            return
        try:
            pattern = cls._key(session_id, "*")
            keys = client.keys(pattern)
            if keys:
                client.delete(*keys)
        except Exception as e:
            logger.debug(f"Redis 세션 삭제 실패: {e}")

    @staticmethod
    def is_available() -> bool:
        return _get_client() is not None
