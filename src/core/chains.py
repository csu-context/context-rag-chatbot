import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from src.core.prompts import RAG_SYSTEM_PROMPT
from src.core.reranker import CrossEncoderReranker
from src.models.factory import LLMFactory
from src.utils.citation import format_citations

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
        source = doc.metadata.get("src_name") or "알 수 없는 파일"
        page = doc.metadata.get("pg_num") or "-"
        content = f"내용: {doc.page_content}\n출처: [{source}, p.{page}]"
        formatted.append(content)
    return "\n\n".join(formatted)


def _combine_answer_and_citations(input_dict: dict[str, Any]) -> str:
    """답변과 인용 정보를 결합하여 최종 응답 생성."""
    answer_obj = input_dict["answer"]

    if hasattr(answer_obj, "content"):
        content = answer_obj.content
        if isinstance(content, list):
            text_parts = [part.get("text", "") if isinstance(part, dict) else str(part) for part in content]
            answer = "".join(text_parts)
        else:
            answer = str(content)
    else:
        answer = str(answer_obj)

    docs = input_dict["docs"]
    if not docs:
        return answer

    citations = format_citations(docs)
    return f"{answer}\n\n{citations}"


def get_rag_chain(retriever_or_db):
    """
    RAG 파이프라인 체인을 생성합니다.
    retriever_or_db: ChromaDBManager 인스턴스 또는 get_relevant_documents를 지원하는 리트리버
    """
    llm_instance = LLMFactory.create_llm()
    llm = llm_instance.get_model()

    def retrieve_and_rerank(input_dict: dict[str, Any]) -> list[Document]:
        query = input_dict.get("question", "")
        k = input_dict.get("k", 5)

        try:
            docs = _perform_retrieval(retriever_or_db, query, k)
            if not docs:
                return []

            reranker = CrossEncoderReranker.get_instance()
            result = reranker.rerank_with_timeout(query, docs)
            return result.documents
        except Exception as e:
            logger.warning(f"검색/리랭킹 실패: {e}")
            return []

    prompt = ChatPromptTemplate.from_messages([("system", RAG_SYSTEM_PROMPT), ("human", "{question}")])

    return (
        RunnablePassthrough.assign(docs=RunnableLambda(retrieve_and_rerank))
        .assign(context=lambda x: _format_docs(x["docs"]))
        .assign(answer=prompt | llm)
    ) | _combine_answer_and_citations
