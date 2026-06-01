import atexit
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.common.config import settings
from src.common.constants import MetadataFields
from src.core.base_retriever import BaseRetriever

logger = logging.getLogger(__name__)

# 하이브리드 검색의 BM25/Vector leg를 병렬 실행하기 위한 공유 스레드풀.
# 매 쿼리마다 ThreadPoolExecutor를 생성·해제하면 그 오버헤드(~수 ms)가 짧은 leg의
# 병렬 이득을 잡아먹어 직렬보다 느려질 수 있다. 모듈 수명 동안 풀을 재사용해 이를 제거한다.
# (워커는 submit 시점에 지연 생성되므로 import 비용은 사실상 없다.)
_RETRIEVAL_EXECUTOR = ThreadPoolExecutor(
    max_workers=settings.RETRIEVER_EXECUTOR_MAX_WORKERS,
    thread_name_prefix="hybrid-retrieval",
)
atexit.register(_RETRIEVAL_EXECUTOR.shutdown, wait=False)


class EnsembleRetriever(BaseRetriever):
    def __init__(self, chroma_manager=None, bm25_manager=None):
        self.rrf_k = settings.RRF_K
        self.weight_bm25 = settings.HYBRID_WEIGHT_BM25
        self.weight_vector = settings.HYBRID_WEIGHT_VECTOR

        from src.vector_db.bm25_manager import BM25Manager

        self.chroma = chroma_manager
        self.bm25 = bm25_manager if bm25_manager else BM25Manager()

    def retrieve(self, query: str, n: int = 5, metadata_filter: dict | None = None) -> list[dict[str, Any]]:
        """BaseRetriever 인터페이스 구현. 하이브리드 RRF 검색을 수행합니다."""
        return self.get_relevant_documents(query, n, metadata_filter=metadata_filter)

    def get_relevant_documents(self, query: str, n: int = 5, metadata_filter: dict | None = None) -> list:
        """BM25 + Vector 병렬 실행 후 RRF 병합."""
        n_candidates = max(settings.RETRIEVER_CANDIDATE_POOL_MIN, n)

        bm25_future = _RETRIEVAL_EXECUTOR.submit(self._get_bm25_results, query, n_candidates, metadata_filter)
        vector_future = _RETRIEVAL_EXECUTOR.submit(self._get_vector_results, query, n_candidates, metadata_filter)
        bm25_results = bm25_future.result()
        vector_results = vector_future.result()

        if not bm25_results and not vector_results:
            logger.warning(f"두 엔진 모두 결과 없음: '{query}'")
            return []
        if not bm25_results:
            logger.info("BM25 결과 없음 -> Vector 결과만 반환")
            return vector_results[:n]
        if not vector_results:
            logger.info("Vector 결과 없음 -> BM25 결과만 반환")
            return bm25_results[:n]

        main_results = self._rrf_fusion(bm25_results, vector_results, n)

        # 표 전용 보조 검색 — 일반 청크에 묻히는 표 데이터 보장
        if settings.TABLE_RETRIEVAL_ENABLED and metadata_filter is None:
            table_filter = {MetadataFields.IS_TABLE: True}
            table_n = max(2, n // 3)
            table_vec = self._get_vector_results(query, table_n, table_filter)
            if table_vec:
                seen_ids = {self._get_doc_id(d) for d in main_results}
                for doc in table_vec:
                    if self._get_doc_id(doc) not in seen_ids:
                        main_results.append(doc)
                        seen_ids.add(self._get_doc_id(doc))

        return main_results

    def _safe_retrieve(self, manager, name: str, query: str, n: int, metadata_filter: dict | None = None) -> list:
        if not manager:
            logger.info(f"{name} 미연결 -> {name} 검색 생략")
            return []
        try:
            return manager.retrieve(query, n, metadata_filter=metadata_filter)
        except Exception as e:
            logger.error(f"{name} 검색 오류: {e}")
            return []

    def _get_bm25_results(self, query: str, n: int, metadata_filter: dict | None = None) -> list:
        return self._safe_retrieve(self.bm25, "BM25Manager", query, n, metadata_filter)

    def _get_vector_results(self, query: str, n: int, metadata_filter: dict | None = None) -> list:
        return self._safe_retrieve(self.chroma, "ChromaManager", query, n, metadata_filter)

    def _rrf_fusion(self, bm25_results: list, vector_results: list, n: int) -> list:
        """Reciprocal Rank Fusion: score = weight * (1 / (k + rank))"""
        scores: dict[str, float] = {}
        docs: dict[str, dict] = {}

        for rank, doc in enumerate(bm25_results, start=1):
            doc_id = self._get_doc_id(doc)
            scores[doc_id] = scores.get(doc_id, 0) + self.weight_bm25 * (1 / (self.rrf_k + rank))
            docs[doc_id] = doc

        for rank, doc in enumerate(vector_results, start=1):
            doc_id = self._get_doc_id(doc)
            scores[doc_id] = scores.get(doc_id, 0) + self.weight_vector * (1 / (self.rrf_k + rank))
            docs[doc_id] = doc

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:n]
        return [
            {**docs[did], "_rrf_score": round(scores[did], 6), "_rank": i} for i, did in enumerate(sorted_ids, start=1)
        ]

    def _get_doc_id(self, doc: dict) -> str:
        if MetadataFields.CHUNK_ID in doc:
            return str(doc[MetadataFields.CHUNK_ID])

        metadata = doc.get("metadata") or {}

        if MetadataFields.CHUNK_ID in metadata:
            return str(metadata[MetadataFields.CHUNK_ID])

        src = metadata.get(MetadataFields.SRC_NAME, "unknown")
        pg = metadata.get(MetadataFields.PG_NUM, "0")
        return f"{src}::p{pg}"

    def compare_retrievers(self, query: str, n: int = 3) -> None:
        """BM25 단독 / Vector 단독 / 하이브리드 결과를 비교 출력."""
        print(f"\n{'=' * 60}")
        print(f"질의: '{query}'")
        print(f"{'=' * 60}")

        bm25_results = self._get_bm25_results(query, n)
        print(f"\n[BM25 단독] {len(bm25_results)}개")
        for r in bm25_results:
            score = r.get("_bm25_score", "-")
            print(f"  - {r.get('content', '')[:50]}  (score: {score})")

        vector_results = self._get_vector_results(query, n)
        print(f"\n[Vector 단독] {len(vector_results)}개")
        for r in vector_results:
            print(f"  - {r.get('content', '')[:50]}")

        hybrid_results = self.get_relevant_documents(query, n)
        print(f"\n[하이브리드 RRF] {len(hybrid_results)}개")
        for r in hybrid_results:
            print(f"  - {r.get('content', '')[:50]}  (rrf: {r.get('_rrf_score', '-')})")

        print(f"{'=' * 60}\n")


class RetrieverFactory:
    """설정에 따른 Retriever 인스턴스를 동적으로 생성하는 팩토리 클래스"""

    @staticmethod
    def create_retriever(
        retriever_type: str | None = None,
        chroma_manager=None,
        bm25_manager=None,
    ) -> BaseRetriever:
        if retriever_type is None:
            retriever_type = settings.RETRIEVER_TYPE

        retriever_type = retriever_type.lower()

        from src.vector_db.bm25_manager import BM25Manager
        from src.vector_db.chroma_manager import ChromaDBManager

        # chroma_manager와 bm25_manager 준비
        c_manager = chroma_manager if chroma_manager else ChromaDBManager()
        b_manager = bm25_manager if bm25_manager else BM25Manager()

        if retriever_type == "vector":
            return c_manager
        elif retriever_type == "bm25":
            return b_manager
        elif retriever_type in ["hybrid", "ensemble"]:
            return EnsembleRetriever(chroma_manager=c_manager, bm25_manager=b_manager)
        else:
            logger.warning(f"알 수 없는 검색 타입 '{retriever_type}'. vector 검색으로 폴백합니다.")
            return c_manager
