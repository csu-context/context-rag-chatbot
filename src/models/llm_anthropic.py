import os
import time
from typing import Any

from langchain_anthropic import ChatAnthropic

from src.models.base import BaseLLM, LLMResponse


class AnthropicModel(BaseLLM):
    """Anthropic Claude 모델을 프로젝트 표준 인터페이스에 맞춰 구현한 클래스입니다."""

    def __init__(self, model_name: str, **kwargs: Any):
        super().__init__(model_name=model_name)
        self._model_name = model_name

        api_key = kwargs.pop("api_key", os.getenv("ANTHROPIC_API_KEY"))
        if not api_key:
            raise ValueError("환경 변수 'ANTHROPIC_API_KEY'가 설정되지 않았습니다.")

        # noinspection PyArgumentList
        self.llm = ChatAnthropic(model=model_name, anthropic_api_key=api_key, **kwargs)

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        start_time = time.time()

        response = self.llm.invoke(prompt, **kwargs)
        latency = time.time() - start_time

        usage = {}
        if hasattr(response, "response_metadata"):
            meta_usage = response.response_metadata.get("usage", {})

            if isinstance(meta_usage, dict):
                input_tokens = meta_usage.get("input_tokens", 0)
                output_tokens = meta_usage.get("output_tokens", 0)
            else:
                input_tokens = getattr(meta_usage, "input_tokens", 0)
                output_tokens = getattr(meta_usage, "output_tokens", 0)

            usage = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }

        return LLMResponse(
            content=str(response.content),
            usage=usage,
            latency=latency,
            model_name=self._model_name,
            metadata=getattr(response, "response_metadata", {}),
            cost=0.0,
        )

    def get_model(self) -> Any:
        """BaseLLM의 추상 메서드 구현: 내부 LangChain 모델 인스턴스 반환."""
        return self.llm

    @property
    def model_name(self) -> str:
        return self._model_name

    @model_name.setter
    def model_name(self, value: str):
        self._model_name = value
