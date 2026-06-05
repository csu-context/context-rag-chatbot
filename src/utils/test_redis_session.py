from unittest.mock import MagicMock, patch

from src.utils.redis_session import RedisSessionStore


@patch("src.utils.redis_session._get_client")
def test_delete_session_success(mock_get_client):
    """Redis 세션 삭제 기능(패턴 검색 후 전체 삭제)이 정상 동작하는지 검증합니다."""
    # 1. 모의(Mock) Redis 클라이언트 및 반환값 설정
    mock_redis = MagicMock()
    mock_get_client.return_value = mock_redis

    test_session_id = "test_user_123"
    expected_pattern = f"rag:session:{test_session_id}:*"
    mock_keys_to_delete = [
        f"rag:session:{test_session_id}:messages",
        f"rag:session:{test_session_id}:docs"
    ]

    # client.keys()가 실행될 때 가짜 키 리스트를 반환하도록 세팅
    mock_redis.keys.return_value = mock_keys_to_delete

    # 2. 실제 삭제 기능 호출
    RedisSessionStore.delete_session(test_session_id)

    # 3. 검증 (올바른 패턴으로 검색하고, 검색된 키들을 전부 삭제했는지 확인)
    mock_redis.keys.assert_called_once_with(expected_pattern)
    mock_redis.delete.assert_called_once_with(*mock_keys_to_delete)


@patch("src.utils.redis_session._get_client")
def test_delete_session_no_keys(mock_get_client):
    """삭제할 키가 없는 경우 delete 메서드가 호출되지 않는지 검증합니다."""
    mock_redis = MagicMock()
    mock_get_client.return_value = mock_redis

    test_session_id = "empty_user_session"

    # client.keys()가 실행될 때 빈 리스트를 반환하도록 세팅
    mock_redis.keys.return_value = []

    # 삭제 기능 호출
    RedisSessionStore.delete_session(test_session_id)

    # 검증
    mock_redis.delete.assert_not_called()