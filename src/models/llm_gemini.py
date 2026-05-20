import logging
import time
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from src.models.base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)


class GeminiModel(BaseLLM):
    """Google Gemini API를 제어하는 sLLM 구현체 클래스입니다."""

    def __init__(self, model_name: str, api_key: str | None = None, temperature: float = 0.1, **kwargs: Any):
        """Gemini 모델 초기화

        :param model_name: Gemini 모델 명 (예: gemini-pro, gemini-1.5-flash)
        :param api_key: Google API Key
        :param temperature: 생성 온도
        """
        super().__init__(model_name=model_name)
        self._model_name = model_name

        if not api_key:
            raise ValueError("Gemini API Key가 누락되었습니다.")

        self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=temperature, **kwargs)

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """동기 방식으로 Gemini 모델에 입력을 전달하고 LLMResponse를 반환합니다."""
        start_time = time.time()
        try:
            response = self.llm.invoke(prompt, **kwargs)
            latency = time.time() - start_time

            usage = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                meta = response.usage_metadata
                usage = {
                    "input_tokens": meta.get("input_token_count") or meta.get("input_tokens") or 0,
                    "output_tokens": meta.get("output_token_count") or meta.get("output_tokens") or 0,
                    "total_tokens": meta.get("total_token_count") or meta.get("total_tokens") or 0,
                }

            return LLMResponse(
                content=response.content,
                usage=usage,
                latency=latency,
                model_name=self.model_name,
                metadata=getattr(response, "response_metadata", {}),
                cost=0.0,
            )
        except Exception as e:
            logger.error(f"Gemini({self.model_name}) 호출 중 오류 발생: {e}")
            raise e

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """비동기 방식으로 Gemini 모델에 입력을 전달하고 LLMResponse를 반환합니다."""
        start_time = time.time()
        try:
            response = await self.llm.ainvoke(prompt, **kwargs)
            latency = time.time() - start_time

            usage = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                meta = response.usage_metadata
                usage = {
                    "input_tokens": meta.get("input_token_count") or meta.get("input_tokens") or 0,
                    "output_tokens": meta.get("output_token_count") or meta.get("output_tokens") or 0,
                    "total_tokens": meta.get("total_token_count") or meta.get("total_tokens") or 0,
                }

            return LLMResponse(
                content=response.content,
                usage=usage,
                latency=latency,
                model_name=self.model_name,
                metadata=getattr(response, "response_metadata", {}),
                cost=0.0,
            )
        except Exception as e:
            logger.error(f"Gemini({self.model_name}) 비동기 호출 중 오류 발생: {e}")
            raise e

    def get_model(self) -> ChatGoogleGenerativeAI:
        """내부 랭체인 모델 인스턴스 반환"""
        return self.llm

    @property
    def model_name(self) -> str:
        """추상 클래스의 model_name 속성 구현"""
        return self._model_name

    @property
    def model_type(self) -> str:
        """현재 인스턴스의 모델 제공자 타입 반환"""
        return "gemini"
