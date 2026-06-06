from unittest.mock import MagicMock, patch

import pytest

from src.common.constants import LLMDefaults
from src.models.factory import LLMFactory
from src.models.llm_claude import ClaudeModel
from src.models.llm_gemini import GeminiModel
from src.models.llm_ollama import OllamaModel


@pytest.fixture
def mock_settings():
    with patch("src.models.factory.settings") as mock_settings:
        mock_settings.MODEL_TYPE = "ollama"
        mock_settings.MODEL_NAME = "test-model"
        mock_settings.TEMPERATURE = 0.5
        mock_settings.ALLOW_EXTERNAL_API = True
        mock_settings.LLM_FALLBACK_ENABLED = True
        mock_settings.GEMINI_API_KEY = "test-gemini-key"
        mock_settings.GOOGLE_API_KEY = "test-google-key"
        mock_settings.ANTHROPIC_API_KEY = "test-anthropic-key"
        mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
        yield mock_settings


class TestLLMFactory:
    def test_create_llm_gemini(self, mock_settings):
        llm = LLMFactory.create_llm(model_type="gemini", model_name="gemini-pro")
        assert isinstance(llm, GeminiModel)
        assert llm.model_name == "gemini-pro"
        assert llm.api_key == "test-gemini-key"

    def test_create_llm_claude(self, mock_settings):
        llm = LLMFactory.create_llm(model_type="claude", model_name="claude-3")
        assert isinstance(llm, ClaudeModel)
        assert llm.model_name == "claude-3"
        assert llm.api_key == "test-anthropic-key"

    def test_create_llm_ollama(self, mock_settings):
        llm = LLMFactory.create_llm(model_type="ollama", model_name="llama3")
        assert isinstance(llm, OllamaModel)
        assert llm.model_name == "llama3"
        assert llm.base_url == "http://localhost:11434"

    def test_create_llm_external_api_not_allowed(self, mock_settings):
        mock_settings.ALLOW_EXTERNAL_API = False
        llm = LLMFactory.create_llm(model_type="gemini", model_name="gemini-pro")
        # Should fallback to ollama
        assert isinstance(llm, OllamaModel)
        assert llm.model_name == LLMDefaults.OLLAMA_DEFAULT

    def test_create_llm_external_api_not_allowed_keep_name(self, mock_settings):
        mock_settings.ALLOW_EXTERNAL_API = False
        llm = LLMFactory.create_llm(model_type="gemini", model_name="local-custom-model")
        # Should fallback to ollama and keep the name if it doesn't contain gemini/claude/gpt
        assert isinstance(llm, OllamaModel)
        assert llm.model_name == "local-custom-model"

    def test_create_llm_unknown_type_allowed(self, mock_settings):
        llm = LLMFactory.create_llm(model_type="unknown_type")
        # Should fallback to Claude
        assert isinstance(llm, ClaudeModel)
        assert llm.model_name == LLMDefaults.CLAUDE_DEFAULT

    def test_create_llm_unknown_type_not_allowed(self, mock_settings):
        mock_settings.ALLOW_EXTERNAL_API = False
        llm = LLMFactory.create_llm(model_type="unknown_type")
        # Should fallback to Ollama
        assert isinstance(llm, OllamaModel)
        assert llm.model_name == LLMDefaults.OLLAMA_DEFAULT

    @patch("src.models.llm_ollama.OllamaModel.get_model")
    @patch("src.models.llm_gemini.GeminiModel.get_model")
    @patch("src.models.llm_claude.ClaudeModel.get_model")
    def test_create_llm_with_fallback(self, mock_claude_get, mock_gemini_get, mock_ollama_get, mock_settings):
        mock_primary = MagicMock()
        mock_ollama_get.return_value = mock_primary

        mock_gemini_model = MagicMock()
        mock_gemini_get.return_value = mock_gemini_model

        mock_claude_model = MagicMock()
        mock_claude_get.return_value = mock_claude_model

        LLMFactory.create_llm_with_fallback(model_type="ollama", model_name="llama3")

        # Primary is ollama, fallbacks should be gemini and claude
        mock_primary.with_fallbacks.assert_called_once()
        args, _ = mock_primary.with_fallbacks.call_args
        fallbacks = args[0]
        assert len(fallbacks) == 2
        assert mock_gemini_model in fallbacks
        assert mock_claude_model in fallbacks

    @patch("src.models.llm_ollama.OllamaModel.get_model")
    def test_create_llm_with_fallback_disabled(self, mock_ollama_get, mock_settings):
        mock_settings.LLM_FALLBACK_ENABLED = False
        mock_primary = MagicMock()
        mock_ollama_get.return_value = mock_primary

        result = LLMFactory.create_llm_with_fallback(model_type="ollama", model_name="llama3")

        assert result == mock_primary
        mock_primary.with_fallbacks.assert_not_called()

    @patch("src.models.llm_ollama.OllamaModel.get_model")
    @patch("src.models.llm_gemini.GeminiModel.get_model")
    @patch("src.models.llm_claude.ClaudeModel.get_model")
    def test_create_llm_with_fallback_exception_in_fallback(
        self, mock_claude_get, mock_gemini_get, mock_ollama_get, mock_settings
    ):
        mock_primary = MagicMock()
        mock_ollama_get.return_value = mock_primary

        # Gemini raises an error
        mock_gemini_get.side_effect = Exception("API Key Invalid")

        mock_claude_model = MagicMock()
        mock_claude_get.return_value = mock_claude_model

        LLMFactory.create_llm_with_fallback(model_type="ollama", model_name="llama3")

        # Primary is ollama, fallbacks should only be claude because gemini failed
        mock_primary.with_fallbacks.assert_called_once()
        args, _ = mock_primary.with_fallbacks.call_args
        fallbacks = args[0]
        assert len(fallbacks) == 1
        assert mock_claude_model in fallbacks
