import functools
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from src.common.constants import MetadataFields
from src.core.cache import SemanticCache
from src.core.prompts import RAG_SYSTEM_PROMPT
from src.core.reranker import RerankerFactory
from src.models.factory import LLMFactory
from src.utils.citation import format_citations
from src.utils.logger import TracingLogger
from src.utils.paths import PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=128)
def _load_source_json(json_path: Path) -> list | None:
    """source_id별 JSON 파일을 LRU 캐시로 로드합니다. 동일 경로의 반복 디스크 I/O를 방지합니다."""
    try:
        if json_path.exists():
            with open(json_path, encoding="utf-8") as f:
                return json.load(f)
        return None
    except Exception as e:
        logger.error(f"JSON 파일 로드 실패: {json_path} - {e}")
        return None


def invalidate_source_json_cache() -> None:
    """인덱싱으로 JSON 파일이 갱신된 경우 LRU 캐시를 무효화합니다."""
    _load_source_json.cache_clear()


class RAGPipeline:
    """RAG 파이프라인의 핵심 로직을 관리하는 클래스"""

    def __init__(self, retriever_or_db: Any, llm: Any = None, reranker: Any = None):
        self.retriever_or_db = retriever_or_db
        self.llm = llm or LLMFactory.create_llm().get_model()
        self.reranker = reranker or RerankerFactory.create()
        self.tracing_logger = TracingLogger()
        self.cache = SemanticCache()

    def _resolve_parent_documents(self, docs: list[Document]) -> list[Document]:
        """자식 청크로 검색된 문서들을 부모 청크의 원문으로 전환합니다."""
        resolved_docs = []

        for doc in docs:
            parent_id = doc.metadata.get(MetadataFields.PARENT_ID)
            source_id = doc.metadata.get(MetadataFields.SOURCE_ID)

            if parent_id and source_id:
                try:
                    json_path = PROCESSED_DATA_DIR / f"{source_id}.json"
                    parents_list = _load_source_json(json_path)
                    if parents_list:
                        for p in parents_list:
                            if p.get("parent_id") == parent_id:
                                doc.page_content = p.get("parent_text", doc.page_content)
                                break
                except Exception as e:
                    logger.error(f"부모 청크 로드 실패: {e}")

            resolved_docs.append(doc)

        return resolved_docs

    def _perform_retrieval(self, query: str, k: int) -> list[Document]:
        """리트리버 타입에 따른 검색 수행 로직"""
        if hasattr(self.retriever_or_db, "get_relevant_documents"):
            results = self.retriever_or_db.get_relevant_documents(query, n=k)
            docs = []
            for res in results:
                if isinstance(res, Document):
                    docs.append(res)
                else:
                    docs.append(
                        Document(
                            page_content=res.get("content", ""),
                            metadata={
                                **res.get("metadata", {}),
                                "score": res.get("score") or res.get("_rrf_score", 0),
                            },
                        )
                    )
            return self._resolve_parent_documents(docs)

        # 기존 ChromaDBManager 호환성 유지
        search_results = self.retriever_or_db.search(query_text=query, k=k)
        docs = [
            Document(page_content=res["content"], metadata={**res["metadata"], "score": res["score"]})
            for res in search_results
        ]
        return self._resolve_parent_documents(docs)

    def _format_docs(self, docs: list[Document]) -> str:
        """프롬프트 주입을 위한 컨텍스트 포맷팅"""
        formatted = []
        for doc in docs:
            source = doc.metadata.get(MetadataFields.SRC_NAME) or "알 수 없는 파일"
            page = doc.metadata.get(MetadataFields.PG_NUM) or "-"
            content = f"내용: {doc.page_content}\n출처: [{source}, p.{page}]"
            formatted.append(content)
        return "\n\n".join(formatted)

    def _do_retrieval(self, query: str, k: int, session: Any) -> list[Document]:
        with session.trace_step("retrieval") as step:
            docs = self._perform_retrieval(query, k)
            step.update(
                {
                    "output_count": len(docs),
                    "data": [{"content": d.page_content[:100] + "...", "metadata": d.metadata} for d in docs],
                }
            )
            return docs

    def _do_reranking(
        self, query: str, docs: list[Document], final_k: int, session: Any
    ) -> tuple[list[Document], list[float]]:
        with session.trace_step("reranking") as step:
            if docs:
                pre_rerank_ids = [d.metadata.get(MetadataFields.CHUNK_ID) or "unknown" for d in docs]
                pre_rerank_scores = [d.metadata.get("score", 0.0) for d in docs]

                rerank_result = self.reranker.rerank_with_timeout(query, docs, top_k=final_k)
                final_docs = rerank_result.documents
                scores = rerank_result.scores
                post_rerank_ids = [d.metadata.get(MetadataFields.CHUNK_ID) or "unknown" for d in final_docs]
            else:
                final_docs, scores, pre_rerank_ids, post_rerank_ids, pre_rerank_scores = [], [], [], [], []
                rerank_result = None

            step.update(
                {
                    "output_count": len(final_docs),
                    "model": str(rerank_result.model_name) if rerank_result else "none",
                    "scores": [f"{s:.4f}" for s in scores],
                    "rank_change": {"before": pre_rerank_ids, "after": post_rerank_ids},
                    "pre_rerank_scores": pre_rerank_scores,
                }
            )
            return final_docs, scores

    def _build_cache_query(self, query: str, history: list[dict[str, Any]]) -> str:
        """대화 맥락에 따른 캐시 오염을 방지하기 위해 최근 대화 이력을 쿼리에 결합합니다."""
        if not history:
            return query
        recent_history = history[-6:]
        history_parts = []
        for msg in recent_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            history_parts.append(f"{role}: {content}")
        history_str = "\n".join(history_parts)
        return f"[History]\n{history_str}\n\n[Current Query]\n{query}"

    def _stream_generation(
        self, query: str, final_docs: list[Document], history: list[dict[str, Any]], session: Any
    ) -> Iterator[str]:
        with session.trace_step("generation") as step:
            context = self._format_docs(final_docs)

            # system 메시지는 RAG_SYSTEM_PROMPT 템플릿 유지
            messages = [("system", RAG_SYSTEM_PROMPT)]

            # history 슬라이딩 윈도우 K=3 적용 (마지막 6개 메시지)
            recent_history = history[-6:]
            for msg in recent_history:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(("human", content))
                elif role == "assistant":
                    messages.append(("ai", content))

            # 현재 질문
            messages.append(("human", "{question}"))

            prompt_template = ChatPromptTemplate.from_messages(messages)
            prompt_val = prompt_template.invoke({"question": query, "context": context})

            llm_params = {
                "model": str(getattr(self.llm, "model_name", "unknown")),
                "temperature": str(getattr(self.llm, "temperature", "unknown")),
            }

            prompt_messages = prompt_val.to_messages()
            preview_content = prompt_messages[0].content if prompt_messages else ""
            step.update(
                {
                    "prompt_preview": str(preview_content)[:200] + "...",
                    "llm_params": llm_params,
                }
            )

            full_answer = ""
            for chunk in self.llm.stream(prompt_val):
                content = self._extract_answer(chunk)
                full_answer += content
                yield content

            step.update({"answer_length": len(full_answer)})
            session.data["final_answer"] = full_answer

    @staticmethod
    def _extract_answer(answer_obj: Any) -> str:
        if hasattr(answer_obj, "content"):
            return str(answer_obj.content)
        return str(answer_obj)

    def stream(self, input_dict: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """전체 RAG 파이프라인을 스트리밍 모드로 실행합니다."""
        query = input_dict.get("question", "")
        retrieval_k = input_dict.get("k", 20)
        final_k = input_dict.get("final_k", 5)
        history = input_dict.get("history", [])

        # 대화 이력이 병합된 고유 캐시 쿼리 생성
        cache_query = self._build_cache_query(query, history)
        use_cache = True

        with self.tracing_logger.start_session(query=query) as session:
            # 1. Semantic Cache Check
            yield {"stage": "cache", "status": "running"}
            cached_result = self.cache.get(cache_query) if use_cache else None
            if cached_result:
                session.data["cache_hit"] = True
                yield {"stage": "cache", "status": "hit"}

                # Stream cached answer
                answer = cached_result["answer"]
                for char in answer:
                    yield {"stage": "generation", "status": "streaming", "output": char}
                yield {"stage": "generation", "status": "complete"}

                # Yield cached sources for citation
                sources = cached_result["sources"]
                yield {"stage": "citation", "status": "complete", "output": "from cache", "source_documents": sources}
                return

            yield {"stage": "cache", "status": "miss"}
            session.data["cache_hit"] = False

            # 2. Retrieval
            yield {"stage": "retrieval", "status": "running"}
            docs = self._do_retrieval(query, retrieval_k, session)
            yield {"stage": "retrieval", "status": "complete", "output": docs}

            # 3. Reranking
            yield {"stage": "reranking", "status": "running"}
            final_docs, scores = self._do_reranking(query, docs, final_k, session)
            for doc, score in zip(final_docs, scores, strict=False):
                doc.metadata["rerank_score"] = score
            yield {"stage": "reranking", "status": "complete", "output": final_docs}

            # 4. Generation
            yield {"stage": "generation", "status": "running"}
            full_answer = ""
            for token in self._stream_generation(query, final_docs, history, session):
                full_answer += token
                yield {"stage": "generation", "status": "streaming", "output": token}
            yield {"stage": "generation", "status": "complete"}

            # 5. Final Formatting (Citation) & Caching
            yield {"stage": "citation", "status": "running"}
            citations_str = format_citations(final_docs)

            # Cache the actual document data, not the formatted string
            docs_for_cache = []
            for doc in final_docs:
                docs_for_cache.append(
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": doc.metadata.get("rerank_score", doc.metadata.get("score", 0.0)),
                    }
                )

            if use_cache:
                self.cache.add(cache_query, full_answer, docs_for_cache)

            yield {"stage": "citation", "status": "complete", "output": citations_str, "source_documents": final_docs}


def get_rag_chain(retriever_or_db):
    """
    RAG 파이프라인 체인을 생성합니다. LangChain Runnable 인터페이스를 준수합니다.
    """
    pipeline = RAGPipeline(retriever_or_db)
    return RunnableLambda(pipeline.stream)
