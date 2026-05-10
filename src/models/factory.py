import logging
import os

from src.models.base import BaseLLM
from src.models.llm_claude import ClaudeModel
from src.models.llm_gemini import GeminiModel

logger = logging.getLogger(__name__)


class LLMFactory:
    """환경 변수 및 설정에 따라 모델 인스턴스를 생성하는 팩토리 클래스"""

    @staticmethod
    def create_llm(model_type: str | None = None, model_name: str | None = None, **kwargs) -> BaseLLM:
        """
        설정된 타입에 따라 모델 인스턴스를 반환합니다.

        Args:
            model_type: 'gemini', 'claude', 'ollama' 등 (기본값: 환경 변수 MODEL_TYPE)
            model_name: 구체적인 모델 명 (기본값: 환경 변수 MODEL_NAME)
        """
        type_ = model_type or os.getenv("MODEL_TYPE", "gemini").lower()
        name_ = model_name or os.getenv("MODEL_NAME")

        logger.info(f"LLM 인스턴스 생성 시도 (Type: {type_}, Name: {name_})")

        if type_ == "gemini":
            # Gemini 모델 이름 기본값 처리
            name_ = name_ or "gemini-1.5-flash-latest"
            return GeminiModel(model_name=name_, **kwargs)

        elif type_ == "claude":
            # Claude 모델 이름 기본값 처리 (최신 Sonnet 4.6)
            name_ = name_ or "claude-sonnet-4-6"
            return ClaudeModel(model_name=name_, **kwargs)

        elif type_ == "ollama":
            # 향후 sLLM 지원을 위한 Placeholder
            # return OllamaModel(model_name=name_ or "llama3", **kwargs)
            raise NotImplementedError("Ollama 추상화는 아직 구현되지 않았습니다.")

        else:
            logger.warning(f"지원하지 않는 모델 타입 '{type_}'입니다. Gemini 모델로 Fallback 합니다.")
            return GeminiModel(model_name="gemini-1.5-flash-latest", **kwargs)
