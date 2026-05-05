import pytest
from langchain_core.documents import Document
from src.core.chains import get_rag_chain

class MockRetriever:
    def __init__(self, docs):
        self.docs = docs

    def search(self, query_text, k=5):
        return [
            {"content": doc.page_content, "metadata": doc.metadata, "score": 0.9}
            for doc in self.docs
        ]

def test_rag_normal_response():
    """문서 내 정보가 있는 경우 정상 답변 및 출처 인용 검증"""
    docs = [
        Document(
            page_content="2026년 신입 사원의 연봉은 5,000만 원입니다.",
            metadata={"src_name": "연봉규정_2026.pdf", "pg_num": 5},
        )
    ]
    retriever = MockRetriever(docs)
    chain = get_rag_chain(retriever)

    response = chain.invoke({"question": "올해 신입 사원 연봉이 얼마야?", "k": 1})
    
    assert "5,000" in response
    assert "연봉규정_2026.pdf" in response

def test_rag_hallucination_prevention():
    """문서 내 정보가 없는 경우 환각 방지 메시지 검증"""
    docs = [
        Document(
            page_content="회사의 점심 시간은 12시부터 1시까지입니다.",
            metadata={"src_name": "복지안내.pdf", "pg_num": 2},
        )
    ]
    retriever = MockRetriever(docs)
    chain = get_rag_chain(retriever)

    response = chain.invoke({"question": "회사에서 법인 차량을 빌릴 수 있어?", "k": 1})
    
    # 환각 방지 멘트 포함 여부 (prompts.py의 RAG_SYSTEM_PROMPT 내용 기준)
    assert "제공된 문서 내에서" in response or "답변이 불가능" in response

