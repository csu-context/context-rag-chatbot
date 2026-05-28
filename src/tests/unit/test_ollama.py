from unittest.mock import MagicMock, patch

import pytest

from src.models.base import LLMResponse
from src.models.factory import LLMFactory
from src.models.llm_ollama import OllamaModel


@pytest.fixture
def mock_chat_ollama():
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
    model = OllamaModel(model_name="llama3", base_url="http://test:11434")
    assert model.model_name == "llama3"
    assert model.base_url == "http://test:11434"
    mock_chat_ollama.assert_called_once_with(
        model="llama3",
        base_url="http://test:11434",
        temperature=0.1,
        num_predict=8192,
        repeat_penalty=1.0,
        num_ctx=8192,
        keep_alive=-1,
        think=False,
    )


def test_ollama_model_invoke(mock_chat_ollama):
    model = OllamaModel(model_name="llama3")
    response = model.invoke("안녕")

    assert isinstance(response, LLMResponse)
    assert response.content == "로컬 모델 응답입니다."
    assert response.usage["input_tokens"] == 10
    assert response.cost == 0.0  # 로컬 모델은 항상 0원
    assert response.model_name == "llama3"


def test_factory_ollama_creation(mock_chat_ollama):
    # LLMFactory를 통한 생성 테스트
    model = LLMFactory.create_llm(model_type="ollama", model_name="solar")
    assert isinstance(model, OllamaModel)
    assert model.model_name == "solar"


def test_ollama_is_model_available_success(mock_chat_ollama):
    model = OllamaModel(model_name="gemma2:2b")

    mock_response = MagicMock()
    mock_response.__enter__.return_value = mock_response
    mock_response.status = 200
    mock_response.read.return_value = b'{"models": [{"name": "gemma2:2b"}]}'

    with patch("urllib.request.urlopen", return_value=mock_response):
        assert model.is_model_available() is True


def test_ollama_is_model_available_failure(mock_chat_ollama):
    model = OllamaModel(model_name="gemma2:2b")

    mock_response = MagicMock()
    mock_response.__enter__.return_value = mock_response
    mock_response.status = 200
    mock_response.read.return_value = b'{"models": [{"name": "llama3:latest"}]}'

    with patch("urllib.request.urlopen", return_value=mock_response):
        assert model.is_model_available() is False


def test_ollama_pull_model_progress(mock_chat_ollama):
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
    import time

    from src.models.llm_ollama import OllamaModel

    model = OllamaModel(model_name="gemma2:2b")

    def mock_pull_progress():
        yield {"status": "downloading", "completed": 20, "total": 100}
        time.sleep(0.01)
        yield {"status": "success", "completed": 100, "total": 100}

    with (
        patch.object(model, "pull_model_progress", side_effect=mock_pull_progress),
        patch.object(model, "is_model_available", return_value=False),
    ):
        model.start_pull_background()

        time.sleep(0.01)
        status = model.get_pull_status()
        assert status is not None
        assert status["status"] in ["downloading", "success", "pulling"]

        for _ in range(50):
            status = model.get_pull_status()
            if status and status["status"] == "success":
                break
            time.sleep(0.01)

        assert status["status"] == "success"
