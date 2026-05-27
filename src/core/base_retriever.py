from abc import ABC, abstractmethod
from typing import Any


class BaseRetriever(ABC):
    """모든 검색 엔진(Retriever)이 구현해야 하는 인터페이스를 정의합니다."""

    @abstractmethod
    def retrieve(self, query: str, n: int = 5) -> list[dict[str, Any]]:
        pass
