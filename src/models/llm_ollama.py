import logging
import os
import time
from typing import Any

from langchain_ollama import ChatOllama

from src.models.base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)


class OllamaModel(BaseLLM):
    """Ollama를 통해 로컬 환경에서 구동되는 sLLM 모델을 제어하는 클래스입니다."""

    def __init__(self, model_name: str, **kwargs: Any):
        """Ollama 모델 초기화

        :param model_name: Ollama 모델 명 (예: llama3, gemma2)
        :param kwargs: base_url, temperature 등 추가 설정
        """
        super().__init__(model_name=model_name)
        self._model_name = model_name

        # 환경 변수 또는 전달된 값에서 base_url과 temperature 추출
        self.base_url = kwargs.pop("base_url", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        if not self.base_url:
            raise ValueError("환경 변수 'OLLAMA_BASE_URL' 설정이 올바르지 않습니다.")

        temperature = kwargs.pop("temperature", 0.1)

        # 랭체인 Ollama 인스턴스명을 하나로 통일 (self.llm)
        self.llm = ChatOllama(model=model_name, base_url=self.base_url, temperature=temperature, **kwargs)

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """동기 방식으로 Ollama 모델에 입력을 전달하고 LLMResponse를 반환합니다."""
        start_time = time.time()
        try:
            response = self.llm.invoke(prompt, **kwargs)
            latency = time.time() - start_time

            # 토큰 사용량 메타데이터 추출
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
                cost=0.0,  # 로컬 sLLM이므로 비용은 0원
            )
        except Exception as e:
            logger.error(f"Ollama({self.model_name}) 호출 중 오류 발생: {e}")
            raise e

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """비동기 방식으로 Ollama 모델에 입력을 전달하고 LLMResponse를 반환합니다."""
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
            logger.error(f"Ollama({self.model_name}) 비동기 호출 중 오류 발생: {e}")
            raise e

    def get_model(self) -> ChatOllama:
        """내부 랭체인 모델 인스턴스 반환"""
        return self.llm

    @property
    def model_name(self) -> str:
        """추상 클래스의 model_name 속성 구현"""
        return self._model_name

    @property
    def model_type(self) -> str:
        """현재 인스턴스의 모델 제공자 타입 반환"""
        return "ollama"
