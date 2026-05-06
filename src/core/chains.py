import logging
import time
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

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

        trace = {
            "query": query,
            "steps": [],
            "total_latency_ms": 0,
        }
        start_total = time.time()

        try:
            # 1. Retrieval
            step_start = time.time()
            docs = _perform_retrieval(retriever_or_db, query, k)
            retrieval_latency = (time.time() - step_start) * 1000

            trace["steps"].append(
                {
                    "step": "retrieval",
                    "latency_ms": f"{retrieval_latency:.2f}",
                    "output_count": len(docs),
                    "data": [{"content": d.page_content[:100] + "...", "metadata": d.metadata} for d in docs],
                }
            )

            # 2. Reranking
            step_start = time.time()
            if docs:
                reranker = CrossEncoderReranker.get_instance()
                rerank_result = reranker.rerank_with_timeout(query, docs)
                final_docs = rerank_result.documents
                scores = rerank_result.scores
            else:
                final_docs = []
                scores = []

            rerank_latency = (time.time() - step_start) * 1000
            trace["steps"].append(
                {
                    "step": "reranking",
                    "latency_ms": f"{rerank_latency:.2f}",
                    "output_count": len(final_docs),
                    "scores": [f"{s:.4f}" for s in scores],
                }
            )

            # 3. Generation
            step_start = time.time()
            context = ""
            if final_docs:
                formatted_docs = []
                for doc in final_docs:
                    source = doc.metadata.get("src_name") or "알 수 없는 파일"
                    page = doc.metadata.get("pg_num") or "-"
                    formatted_docs.append(f"내용: {doc.page_content}\n출처: [{source}, p.{page}]")
                context = "\n\n".join(formatted_docs)

            prompt_val = ChatPromptTemplate.from_messages(
                [("system", RAG_SYSTEM_PROMPT), ("human", "{question}")]
            ).invoke({"question": query, "context": context})

            answer_obj = llm.invoke(prompt_val)
            answer = answer_obj.content if hasattr(answer_obj, "content") else str(answer_obj)

            gen_latency = (time.time() - step_start) * 1000
            trace["steps"].append(
                {
                    "step": "generation",
                    "latency_ms": f"{gen_latency:.2f}",
                    "prompt_preview": str(prompt_val.to_messages()[0].content)[:200] + "...",
                    "answer_length": len(answer),
                }
            )

            # 4. Final Formatting (Citation)
            if final_docs:
                citations = format_citations(final_docs)
                final_answer = f"{answer}\n\n{citations}"
            else:
                final_answer = answer

            trace["total_latency_ms"] = f"{(time.time() - start_total) * 1000:.2f}"
            trace["status"] = "success"

        except Exception as e:
            logger.error(f"RAG 파이프라인 실행 실패: {e}", exc_info=True)
            trace["status"] = "error"
            trace["error_message"] = str(e)
            trace["total_latency_ms"] = f"{(time.time() - start_total) * 1000:.2f}"
            final_answer = "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다."

        # 트레이싱 로그 기록
        tracing_logger.log_trace(trace)
        return final_answer

    return RunnableLambda(run_full_pipeline)
