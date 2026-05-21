import logging
import os
import time
from typing import Any

from langchain_ollama import ChatOllama

from src.models.base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)


class OllamaModel(BaseLLM):
    """Ollama를 통한 로컬 sLLM 구현체"""

    def __init__(self, model_name: str, base_url: str | None = None, temperature: float = 0.1):
        """
        Args:
            model_name: Ollama 모델 명 (예: llama3, solar)
            base_url: Ollama 서버 주소 (기본값: http://localhost:11434)
            temperature: 생성 온도
        """
        super().__init__(model_name=model_name)
        # 환경 변수 또는 직접 전달된 URL 사용
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        self.model = ChatOllama(
            model=model_name,
            base_url=self.base_url,
            temperature=temperature,
        )
        # 헬스 체크 연동 경고 로그
        if not self.check_health():
            logger.warning(
                f"Ollama 서비스 헬스체크 실패 ({self.base_url}). "
                "로컬 sLLM(Ollama)이 구동 중인지 확인해 주세요. "
                "Ollama가 실행되지 않으면 RAG 응답을 생성할 수 없습니다."
            )

    def check_health(self) -> bool:
        """Ollama 서비스 구동 상태를 확인합니다."""
        import urllib.request

        try:
            url = self.base_url.rstrip("/") + "/api/tags"
            with urllib.request.urlopen(url, timeout=2.0) as response:
                return response.status == 200
        except Exception:
            return False

    def invoke(self, prompt: Any, **kwargs: Any) -> LLMResponse:
        start_time = time.time()

        try:
            response = self.model.invoke(prompt, **kwargs)
            latency = time.time() - start_time

            # Ollama는 로컬 구동이므로 명시적으로 비용을 0.0으로 설정
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
                cost=0.0,  # 로컬 모델 비용 0원
            )

        except Exception as e:
            if not self.check_health():
                msg = (
                    f"Ollama 서비스({self.base_url})에 연결할 수 없습니다. "
                    "Ollama 서버가 활성화되어 있고 로컬 모델이 다운로드되어 있는지 확인하십시오."
                )
                logger.error(msg)
                raise ConnectionError(msg) from e
            logger.error(f"Ollama({self.model_name}) 호출 중 오류 발생: {e}")
            raise e

    def get_model(self) -> ChatOllama:
        return self.model
