import logging
import os

from dotenv import load_dotenv

try:
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

from src.vector_db.bm25_manager import BM25Manager


class EnsembleRetriever:
    def __init__(self, chroma_manager=None, bm25_manager=None):
        load_dotenv()
        self.rrf_k = int(os.getenv("RRF_K", 60))
        self.weight_bm25 = float(os.getenv("HYBRID_WEIGHT_BM25", 0.5))
        self.weight_vector = float(os.getenv("HYBRID_WEIGHT_VECTOR", 0.5))
        self.chroma = chroma_manager
        self.bm25 = bm25_manager if bm25_manager else BM25Manager()

    def get_relevant_documents(self, query: str, n: int = 5) -> list:
        """BM25 + Vector 하이브리드 검색 결과를 RRF로 병합하여 반환."""
        bm25_results = self._get_bm25_results(query, n)
        vector_results = self._get_vector_results(query, n)

        if not bm25_results and not vector_results:
            logger.warning(f"두 엔진 모두 결과 없음: '{query}'")
            return []
        if not bm25_results:
            logger.info("BM25 결과 없음 → Vector 결과만 반환")
            return vector_results
        if not vector_results:
            logger.info("Vector 결과 없음 → BM25 결과만 반환")
            return bm25_results

        return self._rrf_fusion(bm25_results, vector_results, n)

    def _get_bm25_results(self, query: str, n: int) -> list:
        try:
            return self.bm25.get_top_n(query, n=n, return_scores=True)
        except Exception as e:
            logger.error(f"BM25 검색 오류: {e}")
            return []

    def _get_vector_results(self, query: str, n: int) -> list:
        if not self.chroma:
            logger.info("ChromaManager 미연결 → Vector 검색 생략")
            return []
        try:
            return self.chroma.search(query, k=n)
        except Exception as e:
            logger.error(f"Vector 검색 오류: {e}")
            return []

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
        """문서 고유 ID 생성 (chunk_id 우선, 없으면 src_name + pg_num 조합)."""
        if "chunk_id" in doc:
            return str(doc["chunk_id"])
        metadata = doc.get("metadata", {})
        src = metadata.get("src_name", "unknown")
        pg = metadata.get("pg_num", "0")
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
