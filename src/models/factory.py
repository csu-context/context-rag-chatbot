import logging

from src.models.llm_anthropic import AnthropicModel  # 유라가 새로 만든 클래스!

from src.common.config import settings
from src.common.constants import LLMDefaults
from src.models.base import BaseLLM
from src.models.llm_gemini import GeminiModel
from src.models.llm_ollama import OllamaModel

logger = logging.getLogger(__name__)


class LLMFactory:
    """환경 변수 및 설정에 따라 모델 인스턴스를 생성하는 팩토리 클래스"""

    @staticmethod
    def get_model(model_type: str | None = None, model_name: str | None = None, **kwargs) -> BaseLLM:
        """
        설정된 타입에 따라 모델 인스턴스를 반환합니다.
        """
        type_ = (model_type or getattr(settings, "MODEL_TYPE", "anthropic")).lower()
        name_ = model_name or getattr(settings, "MODEL_NAME", None)
        temp_ = kwargs.get("temperature", getattr(settings, "TEMPERATURE", 0.0))

        # 보안 정책 검사
        if not getattr(settings, "ALLOW_EXTERNAL_API", True) and type_ in ["gemini", "claude", "anthropic"]:
            logger.warning(
                f"보안 정책: 외부 API 호출이 허용되지 않습니다. '{type_}' 모델 생성이 차단되었습니다. "
                "로컬 sLLM(ollama)으로 자동 전환합니다."
            )
            type_ = "ollama"
            if not name_ or any(ext in name_.lower() for ext in ["gemini", "claude", "gpt", "anthropic"]):
                name_ = "llama3.2:1b"
        elif getattr(settings, "ALLOW_EXTERNAL_API", True) and type_ in ["gemini", "claude", "anthropic"]:
            logger.warning(
                f"보안 경고: 외부 API 모델 '{type_}' 인스턴스를 생성합니다. "
                "현재 외부 API 호출이 허용되어 있습니다 (ALLOW_EXTERNAL_API=True)."
            )

        logger.info(f"LLM 인스턴스 생성 시도 (타입: {type_}, 이름: {name_})")

        if type_ == "gemini":
            name_ = name_ or getattr(LLMDefaults, "GEMINI_DEFAULT", "gemini-1.5-flash")
            api_key = getattr(settings, "GEMINI_API_KEY", getattr(settings, "GOOGLE_API_KEY", None))
            return GeminiModel(model_name=name_, api_key=api_key, temperature=temp_)

        elif type_ in ["claude", "anthropic"]:
            name_ = name_ or getattr(LLMDefaults, "CLAUDE_DEFAULT", "claude-3-5-sonnet-20240620")
            api_key = getattr(settings, "ANTHROPIC_API_KEY", None)
            return AnthropicModel(model_name=name_, api_key=api_key, temperature=temp_)

        elif type_ == "ollama":
            name_ = name_ or "phi3"
            base_url = getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")
            return OllamaModel(model_name=name_, base_url=base_url, temperature=temp_)

        else:
            if not getattr(settings, "ALLOW_EXTERNAL_API", True):
                logger.warning(
                    f"지원하지 않는 모델 타입 '{type_}'이며 외부 API가 비허용되어 로컬 sLLM(ollama)으로 전환합니다."
                )
                base_url = getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")
                return OllamaModel(model_name="llama3.2:1b", base_url=base_url, temperature=temp_)
            else:
                logger.warning(f"지원하지 않는 모델 타입 '{type_}'입니다. 기본 설정(anthropic)으로 Fallback 합니다.")
                api_key = getattr(settings, "ANTHROPIC_API_KEY", None)
                fallback_name = getattr(LLMDefaults, "CLAUDE_DEFAULT", "claude-3-5-sonnet-20240620")
                return AnthropicModel(model_name=fallback_name, api_key=api_key, temperature=temp_)

    @staticmethod
    def create_llm(*args, **kwargs):
        """코드와의 호환성을 위해 get_model을 연결해주는 메서드"""
        return LLMFactory.get_model(*args, **kwargs)
