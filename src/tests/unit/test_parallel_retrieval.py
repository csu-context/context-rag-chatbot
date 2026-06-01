import threading
import time
from unittest.mock import MagicMock

from src.core.retriever import EnsembleRetriever
from src.vector_db.bm25_manager import BM25Manager
from src.vector_db.chroma_manager import ChromaDBManager


def test_hybrid_search_runs_parallel():
    """BM25와 Vector 검색이 병렬로 실행되는지 검증."""
    call_times: list[float] = []
    lock = threading.Lock()

    def slow_retrieve(query, n, metadata_filter=None):
        with lock:
            call_times.append(time.time())
        time.sleep(0.15)
        return [{"content": "doc", "metadata": {"chunk_id": "c1"}, "_bm25_score": 1.0}]

    chroma_mock = MagicMock(spec=ChromaDBManager)
    bm25_mock = MagicMock(spec=BM25Manager)
    chroma_mock.retrieve.side_effect = slow_retrieve
    bm25_mock.retrieve.side_effect = slow_retrieve

    retriever = EnsembleRetriever(chroma_manager=chroma_mock, bm25_manager=bm25_mock)

    from unittest.mock import patch

    # 표 보조 검색 비활성화하여 병렬 타이밍만 검증
    with patch("src.core.retriever.settings") as mock_settings:
        mock_settings.RETRIEVER_CANDIDATE_POOL_MIN = 3
        mock_settings.TABLE_RETRIEVAL_ENABLED = False
        mock_settings.HYBRID_WEIGHT_BM25 = 0.5
        mock_settings.HYBRID_WEIGHT_VECTOR = 0.5
        mock_settings.RRF_K = 60

        start = time.time()
        retriever.get_relevant_documents("질문", n=3)
        elapsed = time.time() - start

    # 두 검색이 병렬이면 0.3초 미만 (직렬이면 0.3초+)
    assert elapsed < 0.28, f"병렬 실행 기대하나 {elapsed:.2f}초 소요 (직렬로 실행된 것으로 추정)"
    assert len(call_times) == 2


def test_hybrid_search_partial_result_on_bm25_empty():
    """BM25 결과 없을 때 Vector 결과만 반환."""
    chroma_mock = MagicMock(spec=ChromaDBManager)
    bm25_mock = MagicMock(spec=BM25Manager)
    chroma_mock.retrieve.return_value = [{"content": "vector_doc", "metadata": {}}]
    bm25_mock.retrieve.return_value = []

    retriever = EnsembleRetriever(chroma_manager=chroma_mock, bm25_manager=bm25_mock)
    results = retriever.get_relevant_documents("질문", n=3)

    assert len(results) > 0
    assert results[0]["content"] == "vector_doc"
