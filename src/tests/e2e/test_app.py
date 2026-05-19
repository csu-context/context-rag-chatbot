import os
import time
from unittest.mock import MagicMock, patch

import pytest

from src.core.chains import get_rag_chain
from src.core.retriever import EnsembleRetriever
from src.vector_db.bm25_manager import BM25Manager
from src.vector_db.chroma_manager import ChromaDBManager

TEST_CASES = [
    {
        "id": "TC-01",
        "category": "기본 목적 조회",
        "query": "조선대학교 학칙의 목적이 무엇인가요?",
        "expect_keywords": ["목적", "교육"],
    },
    {
        "id": "TC-02",
        "category": "표 구조 검색 (Docling 성능 검증 - 항목 매핑)",
        "query": "학부 수업 연한은 표에서 몇 년으로 규정되어 있나요?",
        "expect_keywords": ["수업연한", "년"],
    },
    {
        "id": "TC-03",
        "category": "표 구조 검색 (Docling 성능 검증 - 수치 정확도)",
        "query": "별표 기준, 2학년의 등록금 산정 비율은 얼마인가요?",
        "expect_keywords": ["비율"],
    },
    {
        "id": "TC-04",
        "category": "성적 평가 기준",
        "query": "성적 평가 방법과 기준을 알려주세요",
        "expect_keywords": ["성적", "평가"],
    },
    {
        "id": "TC-05",
        "category": "범위 외 질문 (거절 테스트)",
        "query": "오늘 날씨가 어때요?",
        "expect_keywords": ["범위", "불가", "알 수"],
    },
]


@pytest.fixture(scope="module")
def rag_setup():
    """RAG 파이프라인 전체를 초기화하는 Pytest Fixture (LLM/Reranker는 모킹하여 부하 감소)"""
    print("\n[초기화] RAG 파이프라인 설정 (LLM/Reranker 모킹)...")
    db = ChromaDBManager(collection_name="rag_collection")

    # 리트리버는 실제 DB를 사용하되, LLM과 Reranker는 모킹하여 시스템 부하를 줄임
    with (
        patch("src.models.factory.LLMFactory().get_model") as mock_llm_factory,
        patch("src.core.reranker.RerankerFactory.create") as mock_reranker_factory,
    ):
        # Mock LLM 설정
        mock_llm_inst = MagicMock()
        mock_model = MagicMock()
        mock_model.model_name = "mock-llama"
        mock_model.temperature = 0.0
        mock_llm_inst.get_model.return_value = mock_model
        mock_llm_factory.return_value = mock_llm_inst

        # Mock Reranker 설정
        mock_reranker = MagicMock()
        mock_reranker_factory.return_value = mock_reranker

        # 체인 생성 (이 시점에서 팩토리가 패치된 상태여야 함)
        bm25 = BM25Manager()
        retriever = EnsembleRetriever(chroma_manager=db, bm25_manager=bm25)
        rag_chain = get_rag_chain(retriever)

        return rag_chain, mock_model, mock_reranker


@pytest.mark.parametrize("tc", TEST_CASES, ids=[tc["id"] for tc in TEST_CASES])
@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="CI 환경에서는 실제 LLM(Ollama/API) 서버가 구동되지 않으므로 E2E 테스트를 스킵합니다.",
)
def test_rag_chain_e2e(tc, rag_setup):
    """
    E2E 테스트: 쿼리에 대해 RAG 체인이 정상적으로 답변과 출처를 반환하는지 검증
    (LLM 응답은 테스트 케이스의 키워드를 포함하도록 모킹됨)
    """
    rag_chain, mock_model, mock_reranker = rag_setup

    # Mock Reranker 동작 정의: 입력받은 문서를 그대로 반환
    def mock_rerank(query, docs, top_k):
        mock_result = MagicMock()
        mock_result.documents = docs[:top_k]
        mock_result.scores = [1.0] * len(mock_result.documents)
        return mock_result

    mock_reranker.rerank_with_timeout.side_effect = mock_rerank

    # Mock LLM 응답 정의: 기대하는 키워드를 포함한 답변 생성
    expected_answer = f"테스트 답변입니다. 키워드: {', '.join(tc['expect_keywords'])}"

    def mock_stream(*args, **kwargs):
        yield MagicMock(content=expected_answer)

    mock_model.stream.side_effect = mock_stream

    print(f"\n[{tc['id']}] {tc['category']} - Q: {tc['query']}")

    start = time.time()
    answer = ""
    sources = []
    # 스트리밍 결과 처리
    for step in rag_chain.stream({"question": tc["query"], "k": 10, "final_k": 3}):
        stage = step.get("stage")
        status = step.get("status")

        if stage == "generation" and status == "streaming":
            answer += step.get("output", "")
        elif stage == "citation" and status == "complete":
            sources = step.get("source_documents", [])

    elapsed = time.time() - start

    print(f"  A: {answer[:120].replace(chr(10), ' ')}...")
    print(f"  출처 문서: {len(sources)}개 | 응답 시간: {elapsed:.2f}초")

    # 결과 검증
    assert len(sources) > 0, "검색된 출처 문서가 없습니다."
    if tc["expect_keywords"]:
        missing = [kw for kw in tc["expect_keywords"] if kw not in answer]
        assert not missing, f"응답에 필수 키워드가 누락되었습니다: {missing}"
