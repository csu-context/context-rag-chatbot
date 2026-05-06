import os
import logging
from dotenv import load_dotenv

# 피드백 2: 프로젝트 표준 로거 사용 (src.utils.logger가 있다고 가정)
# 만약 경로가 다르다면 팀의 표준 로거를 import해주기
try:
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

from src.vector_db.bm25_manager import BM25Manager

load_dotenv()


class EnsembleRetriever:
    def __init__(self, chroma_manager=None, bm25_manager=None):
        self.rrf_k = int(os.getenv("RRF_K", 60))
        self.weight_bm25 = float(os.getenv("HYBRID_WEIGHT_BM25", 0.5))
        self.weight_vector = float(os.getenv("HYBRID_WEIGHT_VECTOR", 0.5))
        self.chroma = chroma_manager
        self.bm25 = bm25_manager if bm25_manager else BM25Manager()

    def get_relevant_documents(self, query: str, n: int = 5) -> list:
        bm25_results = self._get_bm25_results(query, n)
        vector_results = self._get_vector_results(query, n)

        if not bm25_results and not vector_results:
            logger.warning(f"두 엔진 모두 결과 없음: '{query}'")
            return []

        return self._rrf_fusion(bm25_results, vector_results, n)

    def _get_bm25_results(self, query: str, n: int) -> list:
        try:
            return self.bm25.get_top_n(query, n=n, return_scores=True)
        except Exception as e:
            logger.error(f"BM25 검색 오류: {e}")
            return []

    def _get_vector_results(self, query: str, n: int) -> list:
        if not self.chroma:
            return []
        try:
            # 피드백: ChromaDBManager의 search() 메서드 호출
            return self.chroma.search(query, k=n)
        except Exception as e:
            logger.error(f"Vector 검색 오류: {e}")
            return []

    def _rrf_fusion(self, bm25_results: list, vector_results: list, n: int) -> list:
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
        return [{**docs[did], "_rrf_score": round(scores[did], 6), "_rank": i}
                for i, did in enumerate(sorted_ids, start=1)]

    def _get_doc_id(self, doc: dict) -> str:
        # 피드백 1: chunk_id는 최상위(Top-level)에 있음
        if "chunk_id" in doc:
            return str(doc["chunk_id"])

        # 피드백 1: 실제 데이터는 src_name, pg_num 사용
        metadata = doc.get("metadata", {})
        src = metadata.get("src_name", "unknown")
        pg = metadata.get("pg_num", "0")
        return f"{src}::p{pg}"