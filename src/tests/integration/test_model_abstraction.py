from unittest.mock import MagicMock, patch

import pytest

from src.models.factory import LLMFactory
from src.models.llm_anthropic import AnthropicModel
from src.models.llm_ollama import OllamaModel


def test_factory_anthropic_model():
    """팩토리가 AnthropicModel 인스턴스를 올바르게 생성하는지 확인"""
    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        patch("src.models.llm_anthropic.ChatAnthropic") as mock_chat,
    ):
        mock_instance = MagicMock()
        mock_chat.return_value = mock_instance

        factory = LLMFactory()
        model = factory.get_model("anthropic", model_name="claude-3-5-sonnet")

        assert isinstance(model, AnthropicModel)
        assert model.model_name == "claude-3-5-sonnet"


def test_factory_ollama_model():
    """팩토리가 OllamaModel 인스턴스를 올바르게 생성하는지 확인"""
    with (
        patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://localhost:11434"}),
        patch("src.models.llm_ollama.ChatOllama") as mock_chat,
    ):
        mock_instance = MagicMock()
        mock_chat.return_value = mock_instance

        factory = LLMFactory()
        model = factory.get_model("ollama", model_name="llama3")

        assert isinstance(model, OllamaModel)
        assert model.model_name == "llama3"


def test_factory_invalid_model_type():
    """올바르지 않은 모델 타입을 입력했을 때 ValueError가 발생하는지 확인"""
    factory = LLMFactory()
    with pytest.raises(ValueError, match="지원하지 않는 모델 타입입니다"):
        factory.get_model("invalid-type")
