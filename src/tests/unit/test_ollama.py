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
    mock_chat_ollama.assert_called_once_with(model="llama3", base_url="http://test:11434", temperature=0.1)


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
    model = LLMFactory().get_model(model_type="ollama", model_name="solar")
    assert isinstance(model, OllamaModel)
    assert model.model_name == "solar"
