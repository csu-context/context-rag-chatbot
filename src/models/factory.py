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
        type_ = (model_type or settings.MODEL_TYPE).lower()
        name_ = model_name or settings.MODEL_NAME
        temp_ = kwargs.get("temperature", settings.TEMPERATURE)

        if not settings.ALLOW_EXTERNAL_API and type_ in ["gemini", "claude"]:
            logger.warning(
                f"보안 정책: 외부 API 호출이 허용되지 않습니다. '{type_}' 모델 생성이 차단되었습니다. "
                "로컬 sLLM(ollama)으로 자동 전환합니다."
            )
            type_ = "ollama"
            if not name_ or any(ext in name_.lower() for ext in ["gemini", "claude", "gpt"]):
                name_ = LLMDefaults.OLLAMA_DEFAULT
        elif settings.ALLOW_EXTERNAL_API and type_ in ["gemini", "claude"]:
            logger.warning(
                f"보안 경고: 외부 API 모델 '{type_}' 인스턴스를 생성합니다. "
                "현재 외부 API 호출이 허용되어 있습니다 (ALLOW_EXTERNAL_API=True)."
            )

        logger.info(f"LLM 인스턴스 생성 시도 (타입: {type_}, 이름: {name_})")

        match type_:
            case "gemini":
                name_ = name_ or LLMDefaults.GEMINI_DEFAULT
                api_key = settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY
                return GeminiModel(model_name=name_, api_key=api_key, temperature=temp_)
            case "claude":
                name_ = name_ or LLMDefaults.CLAUDE_DEFAULT
                api_key = settings.ANTHROPIC_API_KEY
                return ClaudeModel(model_name=name_, api_key=api_key, temperature=temp_)
            case "ollama":
                name_ = name_ or LLMDefaults.OLLAMA_DEFAULT
                base_url = settings.OLLAMA_BASE_URL
                return OllamaModel(model_name=name_, base_url=base_url, temperature=temp_)
            case _:
                if not settings.ALLOW_EXTERNAL_API:
                    logger.warning(
                        f"지원하지 않는 모델 타입 '{type_}'이며 외부 API가 비허용되어 로컬 sLLM(ollama)으로 전환합니다."
                    )
                    return OllamaModel(
                        model_name=LLMDefaults.OLLAMA_DEFAULT, base_url=settings.OLLAMA_BASE_URL, temperature=temp_
                    )
                logger.warning(
                    f"지원하지 않는 모델 타입 '{type_}'입니다. "
                    f"기본 설정({LLMDefaults.CLAUDE_DEFAULT})으로 Fallback 합니다. "
                    "보안 경고: 외부 API 호출이 허용되어 있습니다 (ALLOW_EXTERNAL_API=True)."
                )
                return ClaudeModel(
                    model_name=LLMDefaults.CLAUDE_DEFAULT, api_key=settings.ANTHROPIC_API_KEY, temperature=temp_
                )

    @staticmethod
    def create_llm_with_fallback(model_type: str | None = None, model_name: str | None = None, **kwargs):
        """Fallback 체인(Ollama → Gemini → Claude). FALLBACK_ENABLED=False 또는 외부API 비허용 시 primary만 반환."""
        primary = LLMFactory.create_llm(model_type, model_name, **kwargs)
        primary_model = primary.get_model()

        if not settings.LLM_FALLBACK_ENABLED or not settings.ALLOW_EXTERNAL_API:
            return primary_model

        fallbacks = []
        fallback_order = [
            ("gemini", LLMDefaults.GEMINI_DEFAULT, settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY),
            ("claude", LLMDefaults.CLAUDE_DEFAULT, settings.ANTHROPIC_API_KEY),
        ]
        current_type = (model_type or settings.MODEL_TYPE).lower()

        for fb_type, fb_name, fb_key in fallback_order:
            if fb_type == current_type:
                continue
            if not fb_key:
                continue
            try:
                fb_llm = LLMFactory.create_llm(fb_type, fb_name, **kwargs)
                fallbacks.append(fb_llm.get_model())
                logger.info(f"Fallback 모델 등록: {fb_type}/{fb_name}")
            except Exception as e:
                logger.warning(f"Fallback 모델 초기화 실패 ({fb_type}): {e}")

        if not fallbacks:
            return primary_model

        return primary_model.with_fallbacks(fallbacks)
