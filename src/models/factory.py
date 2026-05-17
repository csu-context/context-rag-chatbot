import logging
from typing import Any

from src.common.config import settings
from src.common.constants import LLMDefaults
from src.models.base import BaseLLM
from src.models.llm_anthropic import AnthropicModel
from src.models.llm_ollama import OllamaModel

logger = logging.getLogger(__name__)


class LLMFactory:
    """환경 변수 및 설정에 따라 모델 인스턴스를 생성하는 팩토리 클래스"""

    @staticmethod
    def get_model(model_type: str | None = None, model_name: str | None = None, **kwargs: Any) -> BaseLLM:
        """설정된 타입에 따라 모델 인스턴스를 반환합니다.

        Args:
            model_type: 모델 제공자 타입 (anthropic, ollama 등)
            model_name: 구체적인 모델 식별자
            **kwargs: 모델 생성을 위한 추가 파라미터
        """
        type_ = (model_type or settings.MODEL_TYPE).lower()
        name_ = model_name or settings.MODEL_NAME
        temp_ = kwargs.get("temperature", settings.TEMPERATURE)

        logger.info(f"LLM 인스턴스 초기화 수행: Provider={type_}, Model={name_}")

        if type_ == "anthropic" or type_ == "claude":
            target_name = name_ or LLMDefaults.CLAUDE_DEFAULT
            return AnthropicModel(model_name=target_name, temperature=temp_, **kwargs)

        elif type_ == "ollama":
            target_name = name_ or "gemma2"
            base_url = settings.OLLAMA_BASE_URL
            return OllamaModel(model_name=target_name, base_url=base_url, temperature=temp_, **kwargs)

        else:
            default_model = LLMDefaults.CLAUDE_DEFAULT
            logger.warning(f"지원하지 않는 타입 '{type_}'이 감지되어 Anthropic 모델로 전환합니다.")
            return AnthropicModel(model_name=default_model, temperature=temp_, **kwargs)
