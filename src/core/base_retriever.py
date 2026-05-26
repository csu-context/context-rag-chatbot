from abc import ABC, abstractmethod
from typing import Any


class BaseRetriever(ABC):
    """모든 검색 엔진(Retriever)이 구현해야 하는 인터페이스를 정의합니다."""

    @abstractmethod
    def retrieve(self, query: str, n: int = 5) -> list[dict[str, Any]]:
        """주어진 사용자 질의에 대해 가장 관련성이 높은 문서 목록을 반환합니다.

        Args:
            query (str): 사용자 검색 질의어
            n (int): 반환할 상위 문서의 개수

        Returns:
            list[dict[str, Any]]: 표준화된 형식의 문서 정보 딕셔너리 리스트
        """
        pass
