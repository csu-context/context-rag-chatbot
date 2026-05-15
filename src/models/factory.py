import logging

from src.common.config import settings
from src.common.constants import LLMDefaults
from src.models.base import BaseLLM
from src.models.llm_claude import ClaudeModel
from src.models.llm_gemini import GeminiModel
from src.models.llm_ollama import OllamaModel

logger = logging.getLogger(__name__)


class LLMFactory:
    """환경 변수 및 설정에 따라 모델 인스턴스를 생성하는 팩토리 클래스"""

    @staticmethod
    def create_llm(model_type: str | None = None, model_name: str | None = None, **kwargs) -> BaseLLM:
        """
        설정된 타입에 따라 모델 인스턴스를 반환합니다.

        Args:
            model_type: 'gemini', 'claude', 'ollama' 등 (기본값: settings.MODEL_TYPE)
            model_name: 구체적인 모델 명 (기본값: settings.MODEL_NAME)
        """
        type_ = (model_type or settings.MODEL_TYPE).lower()
        name_ = model_name or settings.MODEL_NAME
        temp_ = kwargs.get("temperature", settings.TEMPERATURE)

        logger.info(f"LLM 인스턴스 생성 시도 (Type: {type_}, Name: {name_})")

        if type_ == "gemini":
            name_ = name_ or LLMDefaults.GEMINI_DEFAULT
            api_key = settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY
            return GeminiModel(model_name=name_, api_key=api_key, temperature=temp_)

        elif type_ == "claude":
            name_ = name_ or LLMDefaults.CLAUDE_DEFAULT
            api_key = settings.ANTHROPIC_API_KEY
            return ClaudeModel(model_name=name_, api_key=api_key, temperature=temp_)

        elif type_ == "ollama":
            name_ = name_ or "llama3"
            base_url = settings.OLLAMA_BASE_URL
            return OllamaModel(model_name=name_, base_url=base_url, temperature=temp_)

        else:
            logger.warning(
                f"지원하지 않는 모델 타입 '{type_}'입니다. 기본 설정({LLMDefaults.CLAUDE_DEFAULT})으로 Fallback 합니다."
            )
            api_key = settings.ANTHROPIC_API_KEY
            return ClaudeModel(model_name=LLMDefaults.CLAUDE_DEFAULT, api_key=api_key, temperature=temp_)
