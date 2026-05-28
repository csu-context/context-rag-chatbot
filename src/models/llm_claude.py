import logging
import time
from typing import Any

from langchain_anthropic import ChatAnthropic

from src.models.base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)


class ClaudeModel(BaseLLM):
    """Anthropic Claude 모델 구현체"""

    def __init__(self, model_name: str, api_key: str | None = None, temperature: float = 0.1):
        super().__init__(model_name=model_name)
        self.api_key = api_key
        if not self.api_key:
            raise ValueError(f"Claude 모델({model_name})을 위한 API 키가 제공되지 않았습니다.")

        self.model = ChatAnthropic(
            model=model_name,
            temperature=temperature,
            anthropic_api_key=self.api_key,
        )

    def invoke(self, prompt: Any, **kwargs: Any) -> LLMResponse:
        start_time = time.time()

        try:
            response = self.model.invoke(prompt, **kwargs)
            latency = time.time() - start_time

            usage = self.extract_usage(response)

            return LLMResponse(
                content=response.content,
                usage=usage,
                latency=latency,
                model_name=self.model_name,
                metadata=getattr(response, "response_metadata", {}),
            )

        except Exception as e:
            logger.error(f"Claude 호출 중 오류 발생: {e}")
            raise e

    def get_model(self) -> ChatAnthropic:
        return self.model
