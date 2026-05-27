import logging
import re
from collections.abc import Iterator
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from src.common.config import settings
from src.common.constants import MetadataFields
from src.core.cache import SemanticCache
from src.core.prompts import get_system_prompt
from src.core.reranker import RerankerFactory
from src.models.factory import LLMFactory
from src.utils.citation import format_citations
from src.utils.logger import TracingLogger

_INJECTION_PATTERN = re.compile(
    r"(###|---+|\bSystem:|\bAssistant:|\bHuman:|\[INST\]|\[/INST\]|<\|system\|>|<\|user\|>)",
    re.IGNORECASE,
)

# 모델 계열별 한국어 토크나이저 실측 평균 (chars / token).
# tiktoken/transformers 미연동 상태에서 모델 타입별 정적 근사값 사용.
_TOKEN_RATIO: dict[str, float] = {
    "ollama": 1.5,  # Llama/Qwen 계열
    "gemini": 1.3,
    "claude": 1.2,
}

_HISTORY_WINDOW = 6  # 슬라이딩 윈도우: 최근 N개 메시지(= K턴 × 2)

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG 파이프라인의 핵심 로직을 관리하는 클래스"""

    def __init__(self, retriever_or_db: Any, llm: Any = None, reranker: Any = None):
        self.retriever_or_db = retriever_or_db
        self.llm = llm or LLMFactory.create_llm_with_fallback()
        self.reranker = reranker or RerankerFactory.create()
        self.tracing_logger = TracingLogger()
        self.cache = SemanticCache()

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
            return docs

        # 기존 ChromaDBManager 호환성 유지
        search_results = self.retriever_or_db.search(query_text=query, k=k)
        return [
            Document(page_content=res["content"], metadata={**res["metadata"], "score": res["score"]})
            for res in search_results
        ]

    @staticmethod
    def _escape_injection(text: str) -> str:
        """문서 내 프롬프트 인젝션 유발 패턴을 이스케이프합니다."""
        return _INJECTION_PATTERN.sub(lambda m: f"[{m.group(0)}]", text)

    def _format_docs(self, docs: list[Document]) -> str:
        """프롬프트 주입 방어 XML 샌드박싱 적용 컨텍스트 포맷팅"""
        formatted = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get(MetadataFields.SRC_NAME) or "알 수 없는 파일"
            page = doc.metadata.get(MetadataFields.PG_NUM) or "-"
            safe_content = self._escape_injection(doc.page_content)
            entry = f'<document index="{i}">\n내용: {safe_content}\n출처: [{source}, p.{page}]\n</document>'
            formatted.append(entry)
        return "\n\n".join(formatted)

    def _trim_docs_to_token_limit(
        self, docs: list[Document], system_prompt: str, history: list[dict]
    ) -> list[Document]:
        """토큰 한도 초과 시 낮은 점수 문서부터 제거합니다."""
        limit = int(settings.OLLAMA_NUM_CTX * 0.75)
        ratio = _TOKEN_RATIO.get(settings.MODEL_TYPE, 1.5)

        def _est(text: str) -> int:
            return max(1, int(len(text) / ratio))

        fixed_tokens = _est(system_prompt) + sum(_est(m.get("content", "")) for m in history[-_HISTORY_WINDOW:])
        available = limit - fixed_tokens

        ranked = sorted(docs, key=lambda d: d.metadata.get("rerank_score", d.metadata.get("score", 0.0)), reverse=True)
        kept, total = [], 0
        for doc in ranked:
            doc_tokens = _est(doc.page_content)
            if total + doc_tokens > available:
                break
            kept.append(doc)
            total += doc_tokens

        if len(kept) < len(docs):
            logger.warning(f"컨텍스트 토큰 한도 초과: {len(docs)}개 → {len(kept)}개 문서로 트리밍")

        return kept

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
        history_str = "\n".join(
            f"{m.get('role', '')}: {m.get('content', '')}" for m in history[-_HISTORY_WINDOW:]
        )
        return f"[History]\n{history_str}\n\n[Current Query]\n{query}"

    def _stream_generation(
        self, query: str, final_docs: list[Document], history: list[dict[str, Any]], session: Any
    ) -> Iterator[str]:
        with session.trace_step("generation") as step:
            system_prompt = get_system_prompt()
            trimmed_docs = self._trim_docs_to_token_limit(final_docs, system_prompt, history)
            context = self._format_docs(trimmed_docs)

            messages = [("system", system_prompt)]

            for msg in history[-_HISTORY_WINDOW:]:
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

            model_id = str(getattr(self.llm, "model_name", "unknown"))
            llm_params = {
                "model": model_id,
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
            try:
                for chunk in self.llm.stream(prompt_val):
                    content = self._extract_answer(chunk)
                    full_answer += content
                    yield content
            except Exception as exc:
                logger.error(
                    "스트리밍 중 LLM 오류 — 모델: %s, 출력된 토큰: %d자, 예외: %s",
                    model_id,
                    len(full_answer),
                    exc,
                    exc_info=True,
                )
                raise

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
        use_cache = len(history) == 0

        with self.tracing_logger.start_session(query=query) as session:
            # 1. Semantic Cache Check
            yield {"stage": "cache", "status": "running"}
            cached_result = self.cache.get(cache_query) if use_cache else None
            if cached_result:
                session.data["cache_hit"] = True
                yield {"stage": "cache", "status": "hit"}

                answer = cached_result["answer"]
                for char in answer:
                    yield {"stage": "generation", "status": "streaming", "output": char}
                yield {"stage": "generation", "status": "complete"}

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
            docs_for_cache = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": doc.metadata.get("rerank_score", doc.metadata.get("score", 0.0)),
                }
                for doc in final_docs
            ]

            if use_cache:
                self.cache.add(cache_query, full_answer, docs_for_cache)

            yield {"stage": "citation", "status": "complete", "output": citations_str, "source_documents": final_docs}


def get_rag_chain(retriever_or_db):
    """
    RAG 파이프라인 체인을 생성합니다. LangChain Runnable 인터페이스를 준수합니다.
    """
    pipeline = RAGPipeline(retriever_or_db)
    return RunnableLambda(pipeline.stream)
