import asyncio
import os
import time

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
    """RAG 파이프라인 전체를 초기화하는 Pytest Fixture"""
    print("\n[초기화] RAG 파이프라인 설정...")
    db = ChromaDBManager(collection_name="rag_collection")
    bm25 = BM25Manager()
    retriever = EnsembleRetriever(chroma_manager=db, bm25_manager=bm25)
    rag_chain = get_rag_chain(retriever)

    print(f"  - ChromaDB: {db.get_count()} 청크")
    print(f"  - BM25: {len(bm25.corpus_data)} docs")
    return rag_chain


@pytest.mark.parametrize("tc", TEST_CASES, ids=[tc["id"] for tc in TEST_CASES])
@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="CI 환경에서는 실제 LLM(Ollama/API) 서버가 구동되지 않으므로 E2E 테스트를 스킵합니다.",
)
def test_rag_chain_e2e(tc, rag_setup):
    """
    E2E 테스트: 쿼리에 대해 RAG 체인이 정상적으로 답변과 출처를 반환하는지,
    특히 표(Table)에 존재하는 데이터가 손실 없이 답변에 포함되는지 검증
    """
    rag_chain = rag_setup
    print(f"\n[{tc['id']}] {tc['category']} - Q: {tc['query']}")

    start = time.time()
    resp = asyncio.run(rag_chain.ainvoke({"question": tc["query"], "k": 10, "final_k": 3}))
    elapsed = time.time() - start

    answer = resp["answer"]
    sources = resp["source_documents"]

    print(f"  A: {answer[:120].replace(chr(10), ' ')}...")
    print(f"  출처 문서: {len(sources)}개 | 응답 시간: {elapsed:.2f}초")

    # 결과 검증
    assert len(sources) > 0, "검색된 출처 문서가 없습니다."
    if tc["expect_keywords"]:
        missing = [kw for kw in tc["expect_keywords"] if kw not in answer]
        assert not missing, f"응답에 필수 키워드가 누락되었습니다: {missing}"
