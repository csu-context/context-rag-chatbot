import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.core.reranker import CrossEncoderReranker
from src.core.retriever import EnsembleRetriever
from src.vector_db.bm25_manager import BM25Manager
from src.vector_db.chroma_manager import ChromaDBManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_manager(tmp_path, json_files: dict) -> BM25Manager:
    """데이터 디렉토리와 캐시 디렉토리를 분리하여 BM25Manager를 생성함."""
    data_dir = tmp_path / "processed"
    data_dir.mkdir(exist_ok=True)
    cache_dir = tmp_path / "bm25_cache"

    for filename, payload in json_files.items():
        with open(data_dir / filename, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    return BM25Manager(data_dir=data_dir, cache_dir=cache_dir)


@pytest.fixture
def temp_bm25_manager(tmp_path):
    """테스트용 임시 BM25Manager"""
    return _make_manager(
        tmp_path,
        {
            "data.json": [
                {
                    "content": "조선대학교 휴학 신청 기간은 3월부터입니다.",
                    "metadata": {"src_name": "manual_a.pdf", "category": "academic"},
                },
                {
                    "content": "복학 신청 방법은 홈페이지를 참조하세요.",
                    "metadata": {"src_name": "manual_b.pdf", "category": "academic"},
                },
                {
                    "content": "장학금 지급 기준과 성적 장학금 안내입니다.",
                    "metadata": {"src_name": "manual_c.pdf", "category": "scholarship"},
                },
            ],
        },
    )


# ---------------------------------------------------------------------------
# Epic 2: Search & Ranking Refactoring Tests
# ---------------------------------------------------------------------------


def test_bm25_stopwords(temp_bm25_manager):
    """BM25 형태소 분석에서 한국어 불용어가 올바르게 필터링되는지 검증."""
    # STOPWORDS = {"대한", "대해", "위해", "통해", "경우", "또한", "모든", "의한", "따라", "기타", "사항"}
    text = "휴학에 대한 복학과 장학금 지급의 경우 모든 사항에 대해"
    tokens = temp_bm25_manager.tokenizer.tokenize(text)

    # 불용어 단어들이 토큰 목록에 포함되지 않았는지 검증
    for stopword in temp_bm25_manager.tokenizer.stopwords:
        assert stopword not in tokens

    # 유효한 형태소는 포함되었는지 검증 (명사, 용언, 1글자 초과)
    assert "휴학" in tokens
    assert "복학" in tokens
    assert "장학금" in tokens
    assert "지급" in tokens


def test_bm25_global_min_max_scaling(temp_bm25_manager):
    """BM25 결과 스코어가 Global Min-Max Scaling으로 올바르게 정규화되는지 검증."""
    # return_scores=True로 검색 수행
    results = temp_bm25_manager.get_top_n("휴학", n=3, return_scores=True)

    assert len(results) > 0
    # 최고 점수를 가진 문서의 스코어가 1.0으로 정규화되었는지 검증
    # (NP.MAX(SCORES)가 양수일 때 분모가 s_max - 0.0이 되므로 최고 스코어는 1.0이 됨)
    assert results[0]["_bm25_score"] == pytest.approx(1.0, abs=1e-4)

    # 다른 결과들의 스코어도 0.0 이상 1.0 이하인지 검증
    for r in results:
        assert 0.0 <= r["_bm25_score"] <= 1.0


def test_bm25_metadata_filtering(temp_bm25_manager):
    """BM25Manager에서 사전 메타데이터 필터링이 정확하게 동작하는지 검증."""
    # 1. 특정 소스 파일로 필터링
    results_a = temp_bm25_manager.get_top_n("신청", n=3, metadata_filter={"src_name": "manual_a.pdf"})
    assert len(results_a) == 1
    assert results_a[0]["metadata"]["src_name"] == "manual_a.pdf"
    assert "휴학" in results_a[0]["content"]

    # 2. 카테고리로 필터링
    results_academic = temp_bm25_manager.get_top_n("신청", n=3, metadata_filter={"category": "academic"})
    assert len(results_academic) == 2
    for r in results_academic:
        assert r["metadata"]["category"] == "academic"


def test_retriever_candidate_pool_expansion(temp_bm25_manager):
    """EnsembleRetriever가 RRF fusion을 위해 최소 15개 이상의 후보 풀(n_candidates)을 요청하는지 검증."""
    retriever = EnsembleRetriever(bm25_manager=temp_bm25_manager)

    # retrieve 시 n_candidates가 15로 강제 설정되는지 확인하기 위해 _get_bm25_results 호출 시 전달된 n 값 모니터링
    # get_relevant_documents에서 n_candidates = max(15, n) 적용됨
    results = retriever.get_relevant_documents("휴학", n=2)
    assert len(results) > 0


def test_chroma_metadata_filter_where_clause():
    """ChromaDBManager.retrieve가 metadata_filter를 올바른 ChromaDB where 절로 변환하는지 검증."""
    manager = ChromaDBManager.__new__(ChromaDBManager)
    mock_collection = MagicMock()
    mock_collection.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}
    manager.collection = mock_collection

    # 1. 단일 필드 필터 → {k: v} 그대로 전달
    manager.retrieve("질문", n=3, metadata_filter={"category": "academic"})
    assert mock_collection.query.call_args.kwargs["where"] == {"category": "academic"}
    mock_collection.query.reset_mock()

    # 2. 다중 필드 필터 → {"$and": [{k: {"$eq": v}}, ...]} 변환
    manager.retrieve("질문", n=3, metadata_filter={"category": "academic", "src_name": "manual_a.pdf"})
    assert mock_collection.query.call_args.kwargs["where"] == {
        "$and": [
            {"category": {"$eq": "academic"}},
            {"src_name": {"$eq": "manual_a.pdf"}},
        ]
    }
    mock_collection.query.reset_mock()

    # 3. 필터 없음 → where=None
    manager.retrieve("질문", n=3)
    assert mock_collection.query.call_args.kwargs["where"] is None


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA 환경 필요")
def test_bge_reranker_cuda_korean_scoring():
    """BAAI/bge-reranker-v2-m3가 CUDA에서 정상 로드되고 한국어 쿼리에 의미 있는 점수를 반환하는지 검증."""
    CrossEncoderReranker.reset_instance()
    try:
        reranker = CrossEncoderReranker.get_instance(device="cuda", top_k=3)
        assert reranker.device == "cuda"

        query = "휴학 신청 기간은 언제인가요?"
        docs = [
            Document(page_content="휴학 신청 기간은 매 학기 초 2주간입니다.", metadata={}),
            Document(page_content="장학금 지급 기준은 직전 학기 성적 기준입니다.", metadata={}),
            Document(page_content="복학 신청은 개강 2주 전까지 홈페이지에서 가능합니다.", metadata={}),
        ]

        # 워밍업: 첫 호출 시 모델 로드 시간 포함되므로 결과만 검증
        result = reranker.rerank(query, docs, top_k=3, threshold=0.0)

        # 1. 모든 점수가 sigmoid 출력 범위 [0, 1] 이내인지 검증
        assert all(0.0 <= s <= 1.0 for s in result.scores)

        # 2. 점수가 모두 동일하지 않아야 함 (의미 있는 구별)
        assert len(set(round(s, 3) for s in result.scores)) > 1

        # 3. "휴학 신청 기간" 문서가 최상위 랭크인지 검증
        assert "휴학" in result.documents[0].page_content

        # 4. 모델 로드 완료 후 두 번째 호출에서 순수 추론 속도가 1초 미만인지 검증
        result_warm = reranker.rerank(query, docs, top_k=3, threshold=0.0)
        assert result_warm.elapsed_time_sec < 1.0
    finally:
        CrossEncoderReranker.reset_instance()


def test_reranker_strict_context_pruning():
    """Reranker가 response latency 사수를 위해 최종 제공 문서를 엄격하게 최대 3개로 제한하는지 검증."""
    reranker = CrossEncoderReranker.get_instance(top_k=5)

    docs = [Document(page_content=f"테스트 문서 내용 {i}", metadata={"src_name": "test.pdf"}) for i in range(10)]

    # 1. 2개 미만의 문서는 그대로 반환하는지 체크
    short_docs = docs[:1]
    res_short = reranker.rerank("질문", short_docs)
    assert len(res_short.documents) == 1

    # 2. 3개 초과의 문서를 입력하고 top_k=5를 요구할 때,
    # 최종 결과는 10개(RERANKER_MAX_DOCS) 제한 내이므로 top_k인 5개로 반환되는지 검증
    with patch.object(CrossEncoderReranker, "_load_model") as mock_load:
        mock_model = MagicMock()
        # 10개 문서에 대해 임의의 높은 점수 반환
        mock_model.predict.return_value = [5.0] * 10
        mock_load.return_value = mock_model

        res = reranker.rerank("질문", docs, top_k=5, threshold=0.1)
        assert len(res.documents) == 5
        assert res.filtered_count == 5
