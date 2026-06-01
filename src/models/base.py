from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """LLM 응답을 위한 공통 규격 객체"""

    content: str = Field(..., description="생성된 답변 텍스트")
    usage: dict[str, int] = Field(default_factory=dict, description="토큰 사용량 정보")
    latency: float = Field(0.0, description="응답 생성 시간 (초)")
    model_name: str = Field("", description="사용된 모델 명")
    metadata: dict[str, Any] = Field(default_factory=dict, description="추가 메타데이터")
    cost: float = Field(0.0, description="API 호출 비용 (USD)")

    def __init__(self, **data: Any):
        super().__init__(**data)
        if not self.cost and self.usage and self.model_name:
            self.cost = self._calculate_cost()

    def _calculate_cost(self) -> float:
        """config/pricing.yaml 동적 로드로 하드코딩 없이 신규 모델 과금 지원."""
        pricing_table = _load_pricing()
        matched_model = ""
        for known_model in pricing_table:
            if self.model_name.startswith(known_model):
                matched_model = known_model
                break
        if not matched_model:
            return 0.0
        pricing = pricing_table[matched_model]
        input_tokens = self.usage.get("input_tokens", 0)
        output_tokens = self.usage.get("output_tokens", 0)
        return (input_tokens * pricing["input"] / 1_000_000) + (output_tokens * pricing["output"] / 1_000_000)


def _load_pricing() -> dict[str, dict[str, float]]:
    """config/pricing.yaml 동적 로드. 실패 시 constants 폴백."""
    from pathlib import Path

    from src.common.constants import LLMPricing

    pricing_file = Path(__file__).resolve().parent.parent.parent / "config" / "pricing.yaml"
    if pricing_file.exists():
        try:
            import yaml  # type: ignore[import-untyped]

            with open(pricing_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data.get("llm_pricing", LLMPricing.PRICING)
        except Exception:
            pass
    return LLMPricing.PRICING


class BaseLLM(ABC):
    """모든 LLM 모델 구현체가 상속받아야 하는 추상 베이스 클래스"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @staticmethod
    def extract_usage(response: Any) -> dict[str, int]:
        """LangChain 응답 객체에서 토큰 사용량을 추출합니다."""
        if not hasattr(response, "usage_metadata") or not response.usage_metadata:
            return {}
        meta = response.usage_metadata
        return {
            "input_tokens": meta.get("input_token_count") or meta.get("input_tokens") or 0,
            "output_tokens": meta.get("output_token_count") or meta.get("output_tokens") or 0,
            "total_tokens": meta.get("total_token_count") or meta.get("total_tokens") or 0,
        }

    @abstractmethod
    def invoke(self, prompt: Any, **kwargs: Any) -> LLMResponse:
        """단일 질문에 대한 응답을 생성합니다."""
        pass

    @abstractmethod
    def get_model(self) -> Any:
        """내부 LangChain 모델 인스턴스를 반환합니다 (Chain 연동용)."""
        pass
