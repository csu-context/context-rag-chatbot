import pytest


@pytest.fixture(autouse=True)
def _force_chroma_persistent_mode(monkeypatch):
    """단위 테스트가 외부 ChromaDB 서버에 의존하지 않도록 항상 로컬(Persistent) 모드를 강제합니다.

    .env에 CHROMA_SERVER_HOST가 설정된 개발 환경에서는 ChromaDBManager가 서버 모드(HttpClient)로
    전환되어 PersistentClient mock이 우회되고 실 서버에 접속합니다. 그 결과 단위 테스트가
    비결정적으로 실패하므로(예: test_chroma_db_manager_operations), 서버 모드 환경 변수를 제거해
    격리성을 보장합니다. 통합/e2e 테스트는 이 conftest 범위 밖이라 영향을 받지 않습니다.
    """
    monkeypatch.delenv("CHROMA_SERVER_HOST", raising=False)
    monkeypatch.delenv("CHROMA_SERVER_PORT", raising=False)
