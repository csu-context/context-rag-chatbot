from unittest.mock import MagicMock, patch

import pytest

from src.models.base import LLMResponse
from src.models.factory import LLMFactory
from src.models.llm_ollama import OllamaModel


@pytest.fixture
def mock_chat_ollama():
    """ChatOllama 인스턴스를 모의(Mocking)하여 외부 API 호출 없이 로직을 테스트합니다."""
    with patch("src.models.llm_ollama.ChatOllama") as mock:
        mock_inst = mock.return_value
        # Mock invoke response
        mock_response = MagicMock()
        mock_response.content = "로컬 모델 응답입니다."
        mock_response.usage_metadata = {"input_token_count": 10, "output_token_count": 20}
        mock_inst.invoke.return_value = mock_response
        # metadata는 딕셔너리여야 함
        mock_response.response_metadata = {}
        yield mock


def test_ollama_model_initialization(mock_chat_ollama):
    """인스턴스 초기화 시 하이퍼파라미터(num_ctx 등)가 ChatOllama에 정확히 매핑되는지 검증"""
    with patch.object(OllamaModel, "check_health", return_value=True):
        model = OllamaModel(model_name="llama3", base_url="http://test:11434")

    assert model.model_name == "llama3"
    assert model.base_url == "http://test:11434"
    mock_chat_ollama.assert_called_once_with(
        model="llama3",
        base_url="http://test:11434",
        temperature=0.1,
        num_ctx=2048,  # 실제 구현부에 추가된 컨텍스트 길이 파라미터 반영
        num_predict=512,
        repeat_penalty=1.2,
    )


def test_ollama_model_invoke(mock_chat_ollama):
    """동기 호출(invoke) 메서드가 정상적으로 텍스트와 토큰 사용량을 반환하는지 검증"""
    with patch.object(OllamaModel, "check_health", return_value=True):
        model = OllamaModel(model_name="llama3")

    response = model.invoke("안녕")

    assert isinstance(response, LLMResponse)
    assert response.content == "로컬 모델 응답입니다."
    assert response.usage["input_tokens"] == 10
    assert response.cost == 0.0  # 로컬 모델은 항상 0원
    assert response.model_name == "llama3"


def test_factory_ollama_creation(mock_chat_ollama):
    """LLMFactory를 통한 OllamaModel 인스턴스 생성 검증"""
    with patch.object(OllamaModel, "check_health", return_value=True):
        model = LLMFactory().get_model(model_type="ollama", model_name="solar")

    assert isinstance(model, OllamaModel)
    assert model.model_name == "solar"


def test_ollama_is_model_available_success(mock_chat_ollama):
    """로컬에 다운로드된 모델 목록(/api/tags)을 조회하여 타겟 모델이 존재하는지 확인"""
    with patch.object(OllamaModel, "check_health", return_value=True):
        model = OllamaModel(model_name="gemma2:2b")

    mock_response = MagicMock()
    mock_response.__enter__.return_value = mock_response
    mock_response.status = 200
    mock_response.read.return_value = b'{"models": [{"name": "gemma2:2b"}]}'

    with patch("urllib.request.urlopen", return_value=mock_response):
        assert model.is_model_available() is True


def test_ollama_is_model_available_failure(mock_chat_ollama):
    """타겟 모델이 로컬에 존재하지 않을 경우 False 반환 검증"""
    with patch.object(OllamaModel, "check_health", return_value=True):
        model = OllamaModel(model_name="gemma2:2b")

    mock_response = MagicMock()
    mock_response.__enter__.return_value = mock_response
    mock_response.status = 200
    mock_response.read.return_value = b'{"models": [{"name": "llama3:latest"}]}'

    with patch("urllib.request.urlopen", return_value=mock_response):
        assert model.is_model_available() is False


def test_ollama_pull_model_progress(mock_chat_ollama):
    """Ollama API를 통한 모델 다운로드 진행 상태 파싱 검증"""
    with patch.object(OllamaModel, "check_health", return_value=True):
        model = OllamaModel(model_name="gemma2:2b")

    mock_response = MagicMock()
    mock_response.iter_lines.return_value = [
        b'{"status": "downloading", "completed": 50, "total": 100}\n',
        b'{"status": "success"}\n',
    ]

    with patch("requests.post", return_value=mock_response):
        progress_steps = list(model.pull_model_progress())
        assert len(progress_steps) == 2
        assert progress_steps[0]["status"] == "downloading"
        assert progress_steps[0]["completed"] == 50
        assert progress_steps[0]["total"] == 100
        assert progress_steps[1]["status"] == "success"


def test_ollama_pull_status_background(mock_chat_ollama):
    """백그라운드 스레드를 이용한 비동기 모델 다운로드 상태 관리 검증"""
    import time

    with patch.object(OllamaModel, "check_health", return_value=True):
        model = OllamaModel(model_name="gemma2:2b")

    def mock_pull_progress():
        yield {"status": "downloading", "completed": 20, "total": 100}
        time.sleep(0.05)
        yield {"status": "success", "completed": 100, "total": 100}

    with patch.object(model, "pull_model_progress", side_effect=mock_pull_progress):
        model.start_pull_background()

        time.sleep(0.01)
        status = model.get_pull_status()
        assert status is not None
        assert status["status"] in ["downloading", "success", "pulling"]

        for _ in range(15):
            status = model.get_pull_status()
            if status and status["status"] == "success":
                break
            time.sleep(0.01)

        assert status["status"] == "success"
