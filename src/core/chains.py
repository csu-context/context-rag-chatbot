import logging
from collections.abc import Iterator
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from src.common.constants import MetadataFields
from src.core.prompts import RAG_SYSTEM_PROMPT
from src.core.reranker import RerankerFactory
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
    """LLM의 응답 객체에서 텍스트를 추출합니다."""
    if hasattr(answer_obj, "content"):
        return str(answer_obj.content)
    return str(answer_obj)


def _do_retrieval(retriever_or_db, query: str, k: int, session: Any) -> list[Document]:
    with session.trace_step("retrieval") as step:
        docs = _perform_retrieval(retriever_or_db, query, k)
        step.update(
            {
                "output_count": len(docs),
                "data": [{"content": d.page_content[:100] + "...", "metadata": d.metadata} for d in docs],
            }
        )
        return docs


def _do_reranking(query: str, docs: list[Document], final_k: int, session: Any) -> tuple[list[Document], list[float]]:
    with session.trace_step("reranking") as step:
        if docs:
            # 리랭킹 전 ID 순서 기록 (순위 변화 추적용)
            pre_rerank_ids = [d.metadata.get(MetadataFields.CHUNK_ID) or "unknown" for d in docs]
            pre_rerank_scores = [d.metadata.get("score", 0.0) for d in docs]

            reranker = RerankerFactory.create()
            rerank_result = reranker.rerank_with_timeout(query, docs, top_k=final_k)
            final_docs = rerank_result.documents
            scores = rerank_result.scores

            # 리랭킹 후 ID 순서 기록
            post_rerank_ids = [d.metadata.get(MetadataFields.CHUNK_ID) or "unknown" for d in final_docs]
        else:
            final_docs, scores, pre_rerank_ids, post_rerank_ids, pre_rerank_scores = [], [], [], [], []

        step.update(
            {
                "output_count": len(final_docs),
                "scores": [f"{s:.4f}" for s in scores],
                "rank_change": {"before": pre_rerank_ids, "after": post_rerank_ids},
                "pre_rerank_scores": pre_rerank_scores,
            }
        )
        return final_docs, scores


def _stream_generation(query: str, final_docs: list[Document], llm: Any, session: Any) -> Iterator[str]:
    """LLM 스트리밍을 통해 답변을 생성하고 토큰을 순차적으로 반환합니다."""
    with session.trace_step("generation") as step:
        context = _format_docs(final_docs)
        prompt_template = ChatPromptTemplate.from_messages([("system", RAG_SYSTEM_PROMPT), ("human", "{question}")])
        prompt_val = prompt_template.invoke({"question": query, "context": context})

        # LLM 정보 및 파라미터 추출
        llm_params = {}
        if hasattr(llm, "model_name"):
            llm_params["model"] = llm.model_name
        if hasattr(llm, "temperature"):
            llm_params["temperature"] = llm.temperature

        step.update(
            {
                "prompt_preview": str(prompt_val.to_messages()[0].content)[:200] + "...",
                "llm_params": llm_params,
            }
        )

        full_answer = ""
        for chunk in llm.stream(prompt_val):
            content = _extract_answer(chunk)
            full_answer += content
            yield content

        step.update({"answer_length": len(full_answer)})


def get_rag_chain(retriever_or_db):
    """
    RAG 파이프라인 체인을 생성합니다. 스트리밍 및 상태 추적을 지원합니다.
    retriever_or_db: ChromaDBManager 인스턴스 또는 get_relevant_documents를 지원하는 리트리버
    """
    llm_instance = LLMFactory.create_llm()
    llm = llm_instance.get_model()
    tracing_logger = TracingLogger()

    def run_streaming_pipeline(input_dict: dict[str, Any]) -> Iterator[dict[str, Any]]:
        query = input_dict.get("question", "")
        retrieval_k = input_dict.get("k", 20)
        final_k = input_dict.get("final_k", 5)

        with tracing_logger.start_session(query=query) as session:
            # 1. Retrieval
            yield {"stage": "retrieval", "status": "running"}
            docs = _do_retrieval(retriever_or_db, query, retrieval_k, session)
            yield {"stage": "retrieval", "status": "complete", "output": docs}

            # 2. Reranking
            yield {"stage": "reranking", "status": "running"}
            final_docs, scores = _do_reranking(query, docs, final_k, session)
            # 스코어를 메타데이터에 추가
            for doc, score in zip(final_docs, scores, strict=False):
                doc.metadata["rerank_score"] = score
            yield {"stage": "reranking", "status": "complete", "output": final_docs}

            # 3. Generation
            yield {"stage": "generation", "status": "running"}
            answer_stream = _stream_generation(query, final_docs, llm, session)
            for token in answer_stream:
                yield {"stage": "generation", "status": "streaming", "output": token}
            yield {"stage": "generation", "status": "complete"}

            # 4. Final Formatting (Citation)
            yield {"stage": "citation", "status": "running"}
            citations = format_citations(final_docs)
            yield {"stage": "citation", "status": "complete", "output": citations, "source_documents": final_docs}

    return RunnableLambda(run_streaming_pipeline)
