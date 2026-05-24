import logging
import os
import time
from typing import Any, Optional

from langchain_ollama import ChatOllama

from src.models.base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)


class OllamaModel(BaseLLM):
    """Ollama를 통해 로컬 환경에서 구동되는 sLLM 모델을 제어하는 클래스입니다."""

    def __init__(self, model_name: Optional[str] = None, **kwargs: Any):
        """Ollama 모델 초기화

        :param model_name: Ollama 모델 명 (예: llama3, gemma2:2b)
        :param kwargs: base_url, temperature 등 추가 설정
        """
        # 메모리 부족 방지를 위해 phi3를 기본 모델로 설정
        if not model_name:
            model_name = os.getenv("MODEL_NAME", "phi3")

        super().__init__(model_name=model_name)
        self._model_name = model_name

        # 환경 변수 또는 전달된 값에서 base_url과 temperature 추출
        self.base_url = kwargs.pop("base_url", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        if not self.base_url:
            raise ValueError("환경 변수 'OLLAMA_BASE_URL' 설정이 올바르지 않습니다.")

        temperature = kwargs.pop("temperature", 0.1)

        # 랭체인 Ollama 인스턴스 초기화
        self.llm = ChatOllama(
            model=self._model_name,
            base_url=self.base_url,
            temperature=temperature,
            num_ctx=2048,
            **kwargs
        )

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """동기 방식으로 Ollama 모델에 입력을 전달합니다."""
        start_time = time.time()
        try:
            response = self.llm.invoke(prompt, **kwargs)
            latency = time.time() - start_time

            usage = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                meta = response.usage_metadata
                usage = {
                    "input_tokens": meta.get("input_token_count") or 0,
                    "output_tokens": meta.get("output_token_count") or 0,
                    "total_tokens": meta.get("total_token_count") or 0,
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
            logger.error(f"Ollama({self.model_name}) 호출 중 오류 발생: {e}")
            raise e

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """비동기 방식으로 Ollama 모델에 입력을 전달합니다."""
        start_time = time.time()
        try:
            response = await self.llm.ainvoke(prompt, **kwargs)
            latency = time.time() - start_time

            usage = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                meta = response.usage_metadata
                usage = {
                    "input_tokens": meta.get("input_token_count") or 0,
                    "output_tokens": meta.get("output_token_count") or 0,
                    "total_tokens": meta.get("total_token_count") or 0,
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
        return self.llm

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_type(self) -> str:
        return "ollama"
