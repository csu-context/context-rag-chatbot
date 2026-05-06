import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from src.common.constants import MetadataFields
from src.core.prompts import RAG_SYSTEM_PROMPT
from src.core.reranker import CrossEncoderReranker
from src.models.factory import LLMFactory
from src.utils.citation import format_citations
from src.utils.logger import TracingLogger

logger = logging.getLogger(__name__)


def _perform_retrieval(retriever_or_db: Any, query: str, k: int) -> list[Document]:
    """리트리버 유연화에 따른 검색 수행 로직 분리"""
    if hasattr(retriever_or_db, "get_relevant_documents"):
        results = retriever_or_db.get_relevant_documents(query, n=k)
        docs = []
        for res in results:
            if isinstance(res, Document):
                docs.append(res)
            else:
                docs.append(
                    Document(
                        page_content=res.get("content", ""),
                        metadata={**res.get("metadata", {}), "score": res.get("score") or res.get("_rrf_score", 0)},
                    )
                )
        return docs

    # 기존 ChromaDBManager 호환성 유지
    search_results = retriever_or_db.search(query_text=query, k=k)
    return [
        Document(page_content=res["content"], metadata={**res["metadata"], "score": res["score"]})
        for res in search_results
    ]


def _format_docs(docs: list[Document]) -> str:
    """프롬프트 주입을 위한 컨텍스트 포맷팅."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get(MetadataFields.SRC_NAME) or "알 수 없는 파일"
        page = doc.metadata.get(MetadataFields.PG_NUM) or "-"
        content = f"내용: {doc.page_content}\n출처: [{source}, p.{page}]"
        formatted.append(content)
    return "\n\n".join(formatted)


def _extract_answer(answer_obj: Any) -> str:
    """LLM 응답 객체에서 텍스트 답변을 안전하게 추출합니다."""
    if hasattr(answer_obj, "content"):
        content = answer_obj.content
        if isinstance(content, list):
            text_parts = [part.get("text", "") if isinstance(part, dict) else str(part) for part in content]
            return "".join(text_parts)
        return str(content)
    return str(answer_obj)


def get_rag_chain(retriever_or_db):
    """
    RAG 파이프라인 체인을 생성합니다.
    retriever_or_db: ChromaDBManager 인스턴스 또는 get_relevant_documents를 지원하는 리트리버
    """
    llm_instance = LLMFactory.create_llm()
    llm = llm_instance.get_model()
    tracing_logger = TracingLogger()

    def run_full_pipeline(input_dict: dict[str, Any]) -> str:
        query = input_dict.get("question", "")
        k = input_dict.get("k", 5)

        with tracing_logger.start_session(query=query) as session:
            # 1. Retrieval
            with session.trace_step("retrieval") as step:
                docs = _perform_retrieval(retriever_or_db, query, k)
                step.update(
                    {
                        "output_count": len(docs),
                        "data": [{"content": d.page_content[:100] + "...", "metadata": d.metadata} for d in docs],
                    }
                )

            # 2. Reranking
            with session.trace_step("reranking") as step:
                if docs:
                    reranker = CrossEncoderReranker.get_instance()
                    rerank_result = reranker.rerank_with_timeout(query, docs)
                    final_docs = rerank_result.documents
                    scores = rerank_result.scores
                else:
                    final_docs, scores = [], []

                step.update({"output_count": len(final_docs), "scores": [f"{s:.4f}" for s in scores]})

            # 3. Generation
            with session.trace_step("generation") as step:
                context = _format_docs(final_docs)
                prompt_val = ChatPromptTemplate.from_messages(
                    [("system", RAG_SYSTEM_PROMPT), ("human", "{question}")]
                ).invoke({"question": query, "context": context})

                answer_obj = llm.invoke(prompt_val)
                answer = _extract_answer(answer_obj)

                step.update(
                    {
                        "prompt_preview": str(prompt_val.to_messages()[0].content)[:200] + "...",
                        "answer_length": len(answer),
                    }
                )

            # 4. Final Formatting (Citation)
            if final_docs:
                citations = format_citations(final_docs)
                return f"{answer}\n\n{citations}"
            return answer

    return RunnableLambda(run_full_pipeline)
