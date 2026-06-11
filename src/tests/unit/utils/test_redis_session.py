import json
import sys
from unittest.mock import MagicMock, patch

import pytest

mock_redis = MagicMock()
sys.modules["redis"] = mock_redis

from src.utils.redis_session import RedisSessionStore  # noqa: E402


@pytest.fixture
def redis_store():
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


def test_redis_retry_decorator_failure(redis_store):
    mock_redis.Redis.return_value.get.side_effect = Exception("Connection lost")
    result = redis_store.load_messages("test_sess")
    assert result == []


@patch("src.utils.redis_session._get_client")
def test_delete_session_success(mock_get_client):
    mock_redis = MagicMock()
    mock_get_client.return_value = mock_redis

    test_session_id = "test_user_123"
    expected_pattern = f"rag:session:{test_session_id}:*"
    mock_keys_to_delete = [f"rag:session:{test_session_id}:messages", f"rag:session:{test_session_id}:docs"]

    mock_redis.keys.return_value = mock_keys_to_delete

    RedisSessionStore.delete_session(test_session_id)

    mock_redis.keys.assert_called_once_with(expected_pattern)
    mock_redis.delete.assert_called_once_with(*mock_keys_to_delete)


@patch("src.utils.redis_session._get_client")
def test_delete_session_no_keys(mock_get_client):
    mock_redis = MagicMock()
    mock_get_client.return_value = mock_redis

    test_session_id = "empty_user_session"

    mock_redis.keys.return_value = []

    RedisSessionStore.delete_session(test_session_id)

    mock_redis.delete.assert_not_called()
