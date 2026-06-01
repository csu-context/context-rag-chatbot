import json
import sys
from unittest.mock import MagicMock, patch

import pytest

mock_redis = MagicMock()
sys.modules["redis"] = mock_redis

from src.utils.redis_session import RedisSessionStore  # noqa: E402


@pytest.fixture
def redis_store():
    # Setup _get_client mock
    with patch("src.utils.redis_session._get_client", return_value=mock_redis.Redis.return_value):
        yield RedisSessionStore


def test_redis_session_is_available():
    with patch("src.utils.redis_session._get_client", return_value=mock_redis.Redis.return_value):
        assert RedisSessionStore.is_available() is True


def test_save_messages(redis_store):
    session_data = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi", "citations": [{"id": 1}]},
    ]

    redis_store.save_messages("session_123", session_data)

    saved_call_args = mock_redis.Redis.return_value.setex.call_args[0]
    saved_json = saved_call_args[2]
    saved_dict = json.loads(saved_json)

    assert saved_dict == session_data


def test_load_messages(redis_store):
    expected_data = [{"role": "user", "content": "hello"}]
    mock_redis.Redis.return_value.get.return_value = json.dumps(expected_data).encode("utf-8")

    loaded_data = redis_store.load_messages("session_123")
    assert loaded_data == expected_data


def test_delete_session(redis_store):
    redis_store.delete_session("session_123")
    assert mock_redis.Redis.return_value.delete.call_count >= 1


def test_redis_retry_decorator_failure(redis_store):
    mock_redis.Redis.return_value.get.side_effect = Exception("Connection lost")
    result = redis_store.load_messages("test_sess")
    assert result == []
