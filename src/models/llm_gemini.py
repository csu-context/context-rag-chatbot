import logging
import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from src.models.base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)


class GeminiModel(BaseLLM):
    """Google Gemini 모델 래퍼 클래스"""

    def __init__(self, model_name: str, api_key: str | None = None, temperature: float = 0.1):
        super().__init__(model_name=model_name)
        self.temperature = temperature
        self.api_key = api_key
        if not self.api_key:
            raise ValueError(f"Gemini 모델({model_name})을 위한 API 키가 제공되지 않았습니다.")

        self._model = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=self.temperature,
            convert_system_message_to_human=True,  # 시스템 프롬프트 호환성 설정
        )
        logger.info(f"Gemini 모델 초기화 완료: {self.model_name}")

    def invoke(self, prompt: Any, **kwargs: Any) -> LLMResponse:
        """단일 질문에 대한 응답을 생성합니다."""
        start_time = time.time()

        try:
            response = self._model.invoke(prompt, **kwargs)
            latency = time.time() - start_time

            # 토큰 사용량 정보 추출 (LangChain 특성상 모델별로 다를 수 있음)
            usage = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = {
                    "input_tokens": response.usage_metadata.get("input_token_count", 0),
                    "output_tokens": response.usage_metadata.get("output_token_count", 0),
                    "total_tokens": response.usage_metadata.get("total_token_count", 0),
                }

            return LLMResponse(
                content=str(response.content) if hasattr(response, "content") else str(response),
                usage=usage,
                latency=latency,
                model_name=self.model_name,
                metadata=getattr(response, "response_metadata", {}),
            )

        except Exception as e:
            logger.error(f"Gemini 호출 중 오류 발생: {e}")
            raise e

    def get_model(self) -> BaseChatModel:
        """LangChain의 ChatGoogleGenerativeAI 인스턴스를 반환합니다."""
        return self._model

    def update_parameters(self, **kwargs) -> None:
        """모델 파라미터를 동적으로 업데이트합니다."""
        if "temperature" in kwargs:
            self.temperature = kwargs["temperature"]
            self._model.temperature = self.temperature
        if "model_name" in kwargs:
            self.model_name = kwargs["model_name"]
            self._model.model = self.model_name
        logger.info(f"Gemini 모델 파라미터 업데이트: {kwargs}")
