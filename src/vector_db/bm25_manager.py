import logging
from typing import Any

import numpy as np

from src.core.base_retriever import BaseRetriever
from src.utils.paths import BM25_CACHE_DIR, PROCESSED_DATA_DIR
from src.vector_db.bm25_builder import BM25Builder
from src.vector_db.bm25_tokenizer import BM25Tokenizer

logger = logging.getLogger(__name__)


class BM25Manager(BaseRetriever):
    """BM25 검색을 수행하는 검색기(Retriever). 색인 관리는 BM25Builder에 위임합니다."""

    def __init__(self, data_dir=PROCESSED_DATA_DIR, cache_dir=BM25_CACHE_DIR):
        self.data_dir = data_dir
        self.cache_dir = cache_dir

        # Tokenizer 초기화
        self.tokenizer = BM25Tokenizer()

        # Builder를 통해 인덱스 빌드 및 로드
        self.builder = BM25Builder(data_dir=self.data_dir, cache_dir=self.cache_dir, tokenizer=self.tokenizer)
        self.corpus_data, self.bm25 = self.builder.build_or_load()

    def _apply_metadata_filter(self, metadata_filter: dict) -> list[int]:
        """주어진 메타데이터 필터에 부합하는 문서의 인덱스 목록을 반환합니다."""
        return [
            idx
            for idx, doc in enumerate(self.corpus_data)
            if all(doc.get("metadata", {}).get(k) == v for k, v in metadata_filter.items())
        ]

    def get_top_n(
        self,
        query: str,
        n: int = 5,
        return_scores: bool = False,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        if not self.bm25 or not self.corpus_data:
            return []

        tokenized_query = self.tokenizer.tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        # 메타데이터 필터링 적용 및 매칭되는 문서 인덱스 분류
        if metadata_filter:
            matching_indices = self._apply_metadata_filter(metadata_filter)
        else:
            matching_indices = list(range(len(self.corpus_data)))

        if not matching_indices:
            return []

        s_max = float(np.max(scores[matching_indices]))
        denom = s_max if s_max > 0.0 else 1.0

        # 매칭된 인덱스들에 대해서만 스코어 기반 정렬 수행
        matching_scores = scores[matching_indices]
        if not matching_scores.any():
            return []

        n_matching = len(matching_indices)
        if n_matching <= n:
            sorted_sub_indices = np.argsort(matching_scores)[::-1].tolist()
        else:
            top_k_sub = np.argpartition(matching_scores, -n)[-n:]
            sorted_sub_indices = top_k_sub[np.argsort(matching_scores[top_k_sub])[::-1]].tolist()

        top_indices = [matching_indices[i] for i in sorted_sub_indices]
        normalized = [float(scores[i]) / denom for i in top_indices]

        results = []
        for rank, (idx, norm_score) in enumerate(zip(top_indices, normalized, strict=False)):
            doc = self.corpus_data[idx]
            if return_scores:
                results.append({**doc, "_bm25_score": round(norm_score, 4), "_rank": rank + 1})
            else:
                results.append(doc)
        return results

    def retrieve(self, query: str, n: int = 5, metadata_filter: dict | None = None) -> list[dict[str, Any]]:
        """BaseRetriever 인터페이스 구현. BM25 키워드 검색을 실행합니다."""
        return self.get_top_n(query=query, n=n, return_scores=True, metadata_filter=metadata_filter)


if __name__ == "__main__":
    manager = BM25Manager()
    if manager.bm25:
        logger.info("BM25 통합 검색 엔진이 준비되었습니다.")
    else:
        logger.warning("검색 가능한 데이터가 없습니다.")
