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


class BaseLLM(ABC):
    """모든 LLM 모델 구현체가 상속받아야 하는 추상 베이스 클래스"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def invoke(self, prompt: Any, **kwargs: Any) -> LLMResponse:
        """단일 질문에 대한 응답을 생성합니다."""
        pass

    @abstractmethod
    def get_model(self) -> Any:
        """내부 LangChain 모델 인스턴스를 반환합니다 (Chain 연동용)."""
        pass
