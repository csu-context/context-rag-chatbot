from unittest.mock import MagicMock, patch

from src.models.base import LLMResponse
from src.models.factory import LLMFactory


def test_llm_response_structure():
    """공통 응답 객체(LLMResponse)가 올바른 구조를 가지는지 확인"""
    response = LLMResponse(content="테스트 답변", usage={"total_tokens": 10}, latency=1.5, model_name="test-model")
    assert response.content == "테스트 답변"
    assert response.usage["total_tokens"] == 10


def test_factory_anthropic_model():
    """팩토리가 AnthropicModel 인스턴스를 올바르게 생성하는지 확인"""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        # ChatAnthropic의 인스턴스 생성을 모킹합니다.
        with patch("src.models.llm_anthropic.ChatAnthropic") as mock_chat:
            mock_instance = MagicMock()
            mock_chat.return_value = mock_instance

            factory = LLMFactory()
            model = factory.get_model("anthropic", model_name="claude-3-opus")

            assert model.model_type == "anthropic"
            assert model.model_name == "claude-3-opus"


def test_factory_ollama_model():
    """팩토리가 OllamaModel 인스턴스를 올바르게 생성하는지 확인"""
    with patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://localhost:11434"}):
        # ChatOllama의 인스턴스 생성을 모킹합니다.
        with patch("src.models.llm_ollama.ChatOllama") as mock_chat:
            mock_instance = MagicMock()
            mock_chat.return_value = mock_instance

            factory = LLMFactory()
            model = factory.get_model("ollama", model_name="gemma2")

            assert model.model_type == "ollama"
            assert model.model_name == "gemma2"
