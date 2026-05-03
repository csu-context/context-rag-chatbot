import logging
import os

from dotenv import load_dotenv

from src.vector_db.bm25_manager import BM25Manager

load_dotenv()
logger = logging.getLogger(__name__)

# 환경 변수에서 RRF 파라미터 및 가중치 로드
RRF_K = int(os.getenv("RRF_K", 60))
HYBRID_WEIGHT_BM25 = float(os.getenv("HYBRID_WEIGHT_BM25", 0.5))
HYBRID_WEIGHT_VECTOR = float(os.getenv("HYBRID_WEIGHT_VECTOR", 0.5))


class EnsembleRetriever:
    def __init__(self, chroma_manager=None, bm25_manager=None):
        self.chroma = chroma_manager
        self.bm25 = bm25_manager if bm25_manager else BM25Manager()

    # ──────────────────────────────────────────
    # 핵심 인터페이스
    # ──────────────────────────────────────────

    def get_relevant_documents(self, query: str, n: int = 5) -> list:
        """
        BM25 + Vector 하이브리드 검색 결과를 RRF로 병합하여 반환.
        한쪽 엔진 결과가 0개여도 에러 없이 다른 엔진 결과를 반환.
        """
        bm25_results = self._get_bm25_results(query, n)
        vector_results = self._get_vector_results(query, n)

        # 한쪽이 비어있으면 다른 쪽 결과 그대로 반환
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

    # ──────────────────────────────────────────
    # 각 엔진 결과 수집
    # ──────────────────────────────────────────

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
            return self.chroma.query(query, n=n)
        except Exception as e:
            logger.error(f"Vector 검색 오류: {e}")
            return []

    # ──────────────────────────────────────────
    # RRF 알고리즘
    # ──────────────────────────────────────────

    def _rrf_fusion(self, bm25_results: list, vector_results: list, n: int) -> list:
        """
        Reciprocal Rank Fusion: score = weight * (1 / (k + rank))
        동일 문서가 양쪽에서 나올 경우 점수 합산 후 중복 제거.
        문서 식별: src_name + pg_num 조합을 고유 ID로 사용.
        """
        scores: dict[str, float] = {}
        docs: dict[str, dict] = {}

        # BM25 결과 점수화
        for rank, doc in enumerate(bm25_results, start=1):
            doc_id = self._get_doc_id(doc)
            rrf_score = HYBRID_WEIGHT_BM25 * (1 / (RRF_K + rank))
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            docs[doc_id] = doc

        # Vector 결과 점수화
        for rank, doc in enumerate(vector_results, start=1):
            doc_id = self._get_doc_id(doc)
            rrf_score = HYBRID_WEIGHT_VECTOR * (1 / (RRF_K + rank))
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            docs[doc_id] = doc

        # 점수 내림차순 정렬 후 상위 n개 반환
        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:n]

        results = []
        for rank, doc_id in enumerate(sorted_ids, start=1):
            doc = docs[doc_id]
            results.append(
                {
                    **doc,
                    "_rrf_score": round(scores[doc_id], 6),
                    "_rank": rank,
                }
            )

        return results

    # ──────────────────────────────────────────
    # 유틸
    # ──────────────────────────────────────────

    def _get_doc_id(self, doc: dict) -> str:
        """문서 고유 ID 생성 (src_name + pg_num 조합)."""
        metadata = doc.get("metadata", {})
        src = metadata.get("src_name", "unknown")
        pg = metadata.get("pg_num", "0")
        return f"{src}::p{pg}"

    # ──────────────────────────────────────────
    # 비교 분석
    # ──────────────────────────────────────────

    def compare_retrievers(self, query: str, n: int = 3) -> None:
        """
        BM25 단독 / Vector 단독 / 하이브리드 결과를 비교 출력.
        가중치 튜닝 시 참고용.
        """
        print(f"\n{'=' * 60}")
        print(f"질의: '{query}'")
        print(f"{'=' * 60}")

        bm25_results = self._get_bm25_results(query, n)
        print(f"\n[BM25 단독] {len(bm25_results)}개")
        for r in bm25_results:
            print(f"  - {r.get('content', '')[:50]}  (score: {r.get('_bm25_score', '-')})")

        vector_results = self._get_vector_results(query, n)
        print(f"\n[Vector 단독] {len(vector_results)}개")
        for r in vector_results:
            print(f"  - {r.get('content', '')[:50]}")

        hybrid_results = self.get_relevant_documents(query, n)
        print(f"\n[하이브리드 RRF] {len(hybrid_results)}개")
        for r in hybrid_results:
            score = r.get("_rrf_score") or r.get("_bm25_score", "-")
            print(f"  - {r.get('content', '')[:50]}  (rrf: {score})")

        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    retriever = EnsembleRetriever()

    test_queries = ["휴학 신청 기간", "복학 신청 방법", "성적 장학금"]

    for q in test_queries:
        retriever.compare_retrievers(q, n=3)
