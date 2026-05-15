import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.core.chains import get_rag_chain


class MockRetriever:
    def __init__(self, docs):
        self.docs = docs

    def search(self, query_text, k=5):
        return [{"content": doc.page_content, "metadata": doc.metadata, "score": 0.9} for doc in self.docs]


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("CI") == "true"
    or (os.getenv("GOOGLE_API_KEY", "") in ["", "None"] and os.getenv("ANTHROPIC_API_KEY", "") in ["", "None"]),
    reason="CI 환경 스킵 또는 API 키 미설정",
)
async def test_rag_normal_response():
    """문서 내 정보가 있는 경우 정상 답변 및 출처 인용 검증"""
    docs = [
        Document(
            page_content="2026년 신입 사원의 연봉은 5,000만 원입니다.",
            metadata={"src_name": "연봉규정_2026.pdf", "pg_num": 5},
        )
    ]
    retriever = MockRetriever(docs)

    with patch("src.models.factory.LLMFactory.create_llm") as mock_factory:
        mock_llm_inst = MagicMock()
        mock_model = MagicMock()

        # TracingLogger 직렬화를 위해 속성에 실제 값 할당
        mock_model.model_name = "test-model"
        mock_model.temperature = 0.1

        def mock_stream(*args, **kwargs):
            yield MagicMock(content="2026년 신입 사원 연봉은 5,000만 원입니다.")

        mock_model.stream = mock_stream
        mock_model.ainvoke = AsyncMock(return_value=MagicMock(content="2026년 신입 사원 연봉은 5,000만 원입니다."))
        mock_llm_inst.get_model.return_value = mock_model
        mock_factory.return_value = mock_llm_inst

        chain = get_rag_chain(retriever)

        full_response = ""
        async for step in chain.astream({"question": "올해 신입 사원 연봉이 얼마야?", "k": 1}):
            if step.get("stage") == "generation" and step.get("status") == "streaming":
                full_response += step.get("output", "")
            elif step.get("stage") == "citation" and step.get("status") == "complete":
                full_response += f"\n\n{step.get('output', '')}"

    assert "5,000" in full_response
    assert "연봉규정_2026.pdf" in full_response


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("CI") == "true"
    or (os.getenv("GOOGLE_API_KEY", "") in ["", "None"] and os.getenv("ANTHROPIC_API_KEY", "") in ["", "None"]),
    reason="CI 환경 스킵 또는 API 키 미설정",
)
async def test_rag_hallucination_prevention():
    """문서 내 정보가 없는 경우 환각 방지 메시지 검증"""
    docs = [
        Document(
            page_content="회사의 점심 시간은 12시부터 1시까지입니다.",
            metadata={"src_name": "복지안내.pdf", "pg_num": 2},
        )
    ]
    retriever = MockRetriever(docs)

    with patch("src.models.factory.LLMFactory.create_llm") as mock_factory:
        mock_llm_inst = MagicMock()
        mock_model = MagicMock()

        # TracingLogger 직렬화를 위해 속성에 실제 값 할당
        mock_model.model_name = "test-model"
        mock_model.temperature = 0.1

        def mock_stream(*args, **kwargs):
            yield MagicMock(content="제공된 문서에서 관련 내용을 찾을 수 없습니다.")

        mock_model.stream = mock_stream
        mock_model.ainvoke = AsyncMock(return_value=MagicMock(content="제공된 문서에서 관련 내용을 찾을 수 없습니다."))
        mock_llm_inst.get_model.return_value = mock_model
        mock_factory.return_value = mock_llm_inst

        chain = get_rag_chain(retriever)

        full_response = ""
        async for step in chain.astream({"question": "회사에서 법인 차량을 빌릴 수 있어?", "k": 1}):
            if step.get("stage") == "generation" and step.get("status") == "streaming":
                full_response += step.get("output", "")

    hallucination_keywords = ["제공된 문서", "찾을 수 없습니다", "답변이 불가능", "관련된 내용을"]
    assert any(keyword in full_response for keyword in hallucination_keywords)
