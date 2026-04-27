import logging
import os
import time
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.prompts import RAG_SYSTEM_PROMPT
from src.core.reranker import CrossEncoderReranker
from src.utils.citation import format_citations

logger = logging.getLogger(__name__)


def get_rag_chain(vector_db):  # noqa: C901
    """
    RAG 파이프라인 체인을 생성합니다.
    vector_db: src.vector_db.chroma_manager.ChromaDBManager 인스턴스
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        # 보안을 위해 앞 4자리만 출력
        logger.info(f"GOOGLE_API_KEY 로드됨: {api_key[:4]}****")
    else:
        logger.error("GOOGLE_API_KEY를 찾을 수 없습니다! .env 파일을 확인하세요.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=0.1,
        google_api_key=api_key,
    )

    def retrieve_and_rerank(input_dict: dict[str, Any]) -> list[Document]:
        """ChromaDB 검색 -> Document 변환 -> 리랭킹 파이프라인."""
        query = input_dict.get("question", "")
        k = input_dict.get("k", 5)

        # 1. DB 검색 시간 측정
        search_start = time.time()
        try:
            search_results = vector_db.search(query_text=query, k=k)
            docs = [
                Document(page_content=res["content"], metadata={**res["metadata"], "score": res["score"]})
                for res in search_results
            ]
        except Exception as e:
            logger.error(f"DB 검색 중 오류 발생: {e}")
            return []
        search_end = time.time()
        search_duration = search_end - search_start

        if not docs:
            logger.info(f"검색 결과 없음 (소요시간: {search_duration:.2f}s)")
            return []

        # 2. 리랭킹 시간 측정
        rerank_start = time.time()
        try:
            reranker = CrossEncoderReranker.get_instance()
            result = reranker.rerank_with_timeout(query, docs)
            rerank_duration = time.time() - rerank_start

            # 성능 데이터 기록 (1. 일반 로그)
            logger.info(f"단계별 성능 측정: 검색={search_duration:.2f}s, 리랭킹={rerank_duration:.2f}s")

            # 성능 데이터 기록 (2. 전용 파일 로그)
            try:
                from src.utils.logger import PerformanceLogger

                perf_logger = PerformanceLogger()
                perf_logger.log("Search", search_duration, f"k={k} docs={len(docs)}")
                perf_logger.log("Rerank", rerank_duration, f"filtered={len(docs)}->{len(result.documents)}")
            except Exception as log_e:
                logger.error(f"성능 로그 기록 실패: {log_e}")

            return result.documents
        except Exception as e:
            logger.error(f"리랭킹 실패: {e}")
            return docs

    def format_docs(docs: list[Document]) -> str:
        """프롬프트 주입을 위한 컨텍스트 포맷팅."""
        formatted = []
        for doc in docs:
            source = doc.metadata.get("src_name") or "알 수 없는 파일"
            page = doc.metadata.get("pg_num") or "-"
            content = f"내용: {doc.page_content}\n출처: [{source}, p.{page}]"
            formatted.append(content)
        return "\n\n".join(formatted)

    prompt = ChatPromptTemplate.from_messages([("system", RAG_SYSTEM_PROMPT), ("human", "{question}")])

    def combine_answer_and_citations(input_dict: dict[str, Any]) -> str:
        """답변과 인용 정보를 결합하여 최종 응답 생성."""
        answer_obj = input_dict["answer"]

        # 1. 텍스트 추출
        if hasattr(answer_obj, "content"):
            content = answer_obj.content
            # content가 리스트인 경우 (Gemini 멀티모달 응답 등) 첫 번째 텍스트 추출
            if isinstance(content, list):
                text_parts = [part.get("text", "") if isinstance(part, dict) else str(part) for part in content]
                answer = "".join(text_parts)
            else:
                answer = str(content)
        else:
            answer = str(answer_obj)

        # 2. 인용 정보 결합
        docs = input_dict["docs"]
        if not docs:
            return answer

        citations = format_citations(docs)
        return f"{answer}\n\n{citations}"

    # 통합 RAG 체인 구성 (LCEL)
    rag_chain = (
        RunnablePassthrough.assign(docs=RunnableLambda(retrieve_and_rerank))
        .assign(context=lambda x: format_docs(x["docs"]))
        .assign(answer=prompt | llm)
    ) | combine_answer_and_citations

    return rag_chain
