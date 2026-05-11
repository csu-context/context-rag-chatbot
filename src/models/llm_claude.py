import logging
import os
import time
from typing import Any

from langchain_anthropic import ChatAnthropic

from src.models.base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)


class ClaudeModel(BaseLLM):
    """Anthropic Claude 모델 구현체"""

    def __init__(self, model_name: str = "claude-sonnet-4-6", temperature: float = 0.1):
        super().__init__(model_name=model_name)
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
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

            # 토큰 사용량 정보 추출 (키 값 변동 대응)
            usage = {}
            if hasattr(response, "usage_metadata"):
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
            )

        except Exception as e:
            logger.error(f"Claude 호출 중 오류 발생: {e}")
            raise e

    def get_model(self) -> ChatAnthropic:
        return self.model
