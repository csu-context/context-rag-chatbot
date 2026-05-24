from unittest.mock import MagicMock, patch

from src.models.factory import LLMFactory
from src.models.llm_anthropic import AnthropicModel


def test_factory_anthropic_model():
    """팩토리가 AnthropicModel 인스턴스를 올바르게 생성하는지 확인"""
    with (
        patch("src.models.factory.settings") as mock_settings,
        patch("src.models.llm_anthropic.ChatAnthropic") as mock_chat,
    ):
        mock_settings.ALLOW_EXTERNAL_API = True
        mock_settings.ANTHROPIC_API_KEY = "test-key"

        mock_instance = MagicMock()
        mock_chat.return_value = mock_instance

        factory = LLMFactory()
        model = factory.get_model("anthropic", model_name="claude-3-5-sonnet")

        assert isinstance(model, AnthropicModel)
        assert model.model_name == "claude-3-5-sonnet"


def test_factory_invalid_model_type():
    """올바르지 않은 모델 타입을 입력했을 때 예외 대신 Fallback이 작동하는지 확인"""
    with patch("src.models.factory.settings") as mock_settings, patch("src.models.llm_anthropic.ChatAnthropic"):
        mock_settings.ALLOW_EXTERNAL_API = True
        mock_settings.ANTHROPIC_API_KEY = "test-key"

        factory = LLMFactory()
        model = factory.get_model("invalid-type")

        assert isinstance(model, AnthropicModel)
