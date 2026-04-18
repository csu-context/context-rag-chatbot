import logging
import os
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.prompts import RAG_SYSTEM_PROMPT
from src.core.reranker import CrossEncoderReranker
from src.utils.citation import format_citations

logger = logging.getLogger(__name__)

def get_rag_chain(retriever):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        safety_settings=None,
    )

    def retrieve_and_rerank(input_dict: dict[str, Any]) -> list[Document]:
        """검색 → 리랭킹 → 상위 문서 반환 파이프라인."""
        query = input_dict.get("question", "")

        try:
            docs = retriever.invoke(input_dict)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

        if not docs:
            logger.warning("No documents retrieved from vector DB.")
            return []

        reranker = CrossEncoderReranker.get_instance()
        result = reranker.rerank_with_timeout(query, docs)

        if result.filtered_count > 0:
            logger.debug(f"Reranked {len(docs)} → {len(result.documents)} docs (filtered {result.filtered_count})")

        return result.documents

    def format_docs(docs: list[Document]) -> str:
        formatted = []
        for doc in docs:
            source = doc.metadata.get("src_name") or doc.metadata.get("source", "unknown")
            page = doc.metadata.get("pg_num") or doc.metadata.get("page", "-")
            content = f"내용: {doc.page_content}\n출처: [{source}, p.{page}]"
            formatted.append(content)
        return "\n\n".join(formatted)

    prompt = ChatPromptTemplate.from_messages([("system", RAG_SYSTEM_PROMPT), ("human", "{question}")])

    def combine_answer_and_citations(input_dict: dict[str, Any]) -> str:
        answer = input_dict["answer"].content
        citations = format_citations(input_dict["docs"])
        return f"{answer}{citations}"

    rag_chain = (
        RunnablePassthrough.assign(docs=RunnableLambda(retrieve_and_rerank))
        .assign(context=lambda x: format_docs(x["docs"]))
        .assign(answer=prompt | llm)
    ) | combine_answer_and_citations

    return rag_chain
