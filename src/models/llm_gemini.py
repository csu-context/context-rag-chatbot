import logging
import os
import time
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from src.models.base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)


class GeminiModel(BaseLLM):
    """Google Gemini 모델 구현체"""

    def __init__(self, model_name: str = "gemini-1.5-flash-latest", temperature: float = 0.1):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            logger.warning("GOOGLE_API_KEY가 설정되지 않았습니다.")

        self.model_name = model_name
        self.model = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=self.api_key)

    def invoke(self, prompt: Any, **kwargs: Any) -> LLMResponse:
        start_time = time.time()

        try:
            response = self.model.invoke(prompt, **kwargs)
            latency = time.time() - start_time

            # 토큰 사용량 정보 추출 (LangChain 특성상 모델별로 다를 수 있음)
            usage = {}
            if hasattr(response, "usage_metadata"):
                usage = {
                    "input_tokens": response.usage_metadata.get("input_token_count", 0),
                    "output_tokens": response.usage_metadata.get("output_token_count", 0),
                    "total_tokens": response.usage_metadata.get("total_token_count", 0),
                }

            return LLMResponse(
                content=response.content,
                usage=usage,
                latency=latency,
                model_name=self.model_name,
                metadata=getattr(response, "response_metadata", {}),
            )

        except Exception as e:
            logger.error(f"Gemini 호출 중 오류 발생: {e}")
            raise e

    def get_model(self) -> ChatGoogleGenerativeAI:
        return self.model
