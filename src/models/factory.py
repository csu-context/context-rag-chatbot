import logging
import os
from typing import Any

from src.models import AnthropicModel, BaseLLM, GeminiModel, OllamaModel

logger = logging.getLogger(__name__)

class LLMFactory:
    """
    환경 변수 및 설정에 따라 적절한 LLM 인터페이스 객체를 생성하는 팩토리 클래스입니다.
    """

    @staticmethod
    def create_llm(model_type: str | None = None, model_name: str | None = None, **kwargs: Any) -> BaseLLM:
        """
        요청된 타입과 이름에 부합하는 LLM 인스턴스를 생성하여 반환합니다.

        Args:
            model_type: 모델 제공자 타입 (anthropic, gemini, ollama 등)
            model_name: 구체적인 모델 식별자
            **kwargs: 모델 생성을 위한 추가 파라미터
        """
        type_ = model_type or os.getenv("MODEL_TYPE", "anthropic").lower()
        name_ = model_name or os.getenv("MODEL_NAME")

        logger.info(f"LLM 인스턴스 초기화 수행: Provider={type_}, Model={name_}")

        if type_ == "anthropic":
            target_name = name_ or "claude-3-5-sonnet-20240620"
            return AnthropicModel(model_name=target_name, **kwargs)

        elif type_ == "gemini":
            target_name = name_ or "gemini-1.5-flash-latest"
            return GeminiModel(model_name=target_name, **kwargs)

        elif type_ == "ollama":
            # 로컬 Ollama 환경 기반의 gemma2 모델 인스턴스 생성
            target_name = name_ or "gemma2"
            return OllamaModel(model_name=target_name, **kwargs)

        else:
            default_model = "claude-3-5-sonnet-20240620"
            logger.warning(f"지원하지 않는 타입 '{type_}'이 감지되어 Anthropic 모델로 전환합니다.")
            return AnthropicModel(model_name=default_model, **kwargs)
