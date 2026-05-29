from unittest.mock import MagicMock, patch

import pytest

from src.models.base import LLMResponse
from src.models.llm_claude import ClaudeModel
from src.models.llm_gemini import GeminiModel


def test_gemini_model_init_and_invoke():
    # Value error when no api key
    with pytest.raises(ValueError) as exc_info:
        GeminiModel(model_name="gemini-1.5-pro", api_key=None)
    assert "API 키가 제공되지 않았습니다" in str(exc_info.value)

    mock_chat_model = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "mocked gemini response"
    mock_response.usage_metadata = {"input_token_count": 10, "output_token_count": 20, "total_token_count": 30}
    mock_response.response_metadata = {"finish_reason": "stop"}
    mock_chat_model.invoke.return_value = mock_response

    with patch("src.models.llm_gemini.ChatGoogleGenerativeAI", return_value=mock_chat_model):
        model = GeminiModel(model_name="gemini-1.5-pro", api_key="fake-key", temperature=0.2)
        assert model.get_model() == mock_chat_model

        # Invoke
        res = model.invoke("Hello")
        assert isinstance(res, LLMResponse)
        assert res.content == "mocked gemini response"
        assert res.usage["input_tokens"] == 10
        assert res.usage["output_tokens"] == 20
        assert res.usage["total_tokens"] == 30
        assert res.metadata["finish_reason"] == "stop"

        # Update parameters
        model.update_parameters(temperature=0.7, model_name="gemini-2.0-flash")
        assert model.temperature == 0.7
        assert model.model_name == "gemini-2.0-flash"
        assert mock_chat_model.temperature == 0.7
        assert mock_chat_model.model == "gemini-2.0-flash"


def test_claude_model_init_and_invoke():
    # Value error when no api key
    with pytest.raises(ValueError) as exc_info:
        ClaudeModel(model_name="claude-3-opus", api_key=None)
    assert "API 키가 제공되지 않았습니다" in str(exc_info.value)

    mock_chat_model = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "mocked claude response"
    mock_response.usage_metadata = {"input_token_count": 15, "output_token_count": 25, "total_token_count": 40}
    mock_response.response_metadata = {"stop_reason": "end_turn"}
    mock_chat_model.invoke.return_value = mock_response

    with patch("src.models.llm_claude.ChatAnthropic", return_value=mock_chat_model):
        model = ClaudeModel(model_name="claude-3-opus", api_key="fake-key", temperature=0.3)
        assert model.get_model() == mock_chat_model

        # Invoke
        res = model.invoke("Hello Claude")
        assert isinstance(res, LLMResponse)
        assert res.content == "mocked claude response"
        assert res.usage["input_tokens"] == 15
        assert res.usage["output_tokens"] == 25
        assert res.usage["total_tokens"] == 40
        assert res.metadata["stop_reason"] == "end_turn"
