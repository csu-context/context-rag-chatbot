import logging
import os
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.common.constants import MetadataFields
from src.core.chains import get_rag_chain
from src.utils.logger import PerformanceLogger, setup_global_logging
from src.utils.paths import ensure_directories
from src.vector_db.chroma_manager import ChromaDBManager

# 환경 변수 및 로깅 설정
load_dotenv()
setup_global_logging()
perf_logger = PerformanceLogger()  # 전용 로거 인스턴스 생성
logger = logging.getLogger(__name__)

# --- 1. 페이지 설정 (가장 상단에 위치) ---
st.set_page_config(
    page_title="기업 매뉴얼 챗봇 (1차 통합)",
    layout="wide",
)

# 필수 디렉토리 확인 및 생성
ensure_directories()


# --- 팝업 다이얼로그 정의 (문서 원문 보기) ---
@st.dialog("문서 원문 보기")
def show_document_dialog(source: str, page: str, score: float, content: str):
    st.markdown(f"**출처:** {source}")
    st.markdown(f"**페이지:** {page}")
    st.markdown(f"**관련도 점수:** {score:.4f}")
    st.text_area("원문 내용", content, height=300)


# --- 2. RAG 시스템 초기화 (캐싱) ---
@st.cache_resource
def initialize_rag_system():
    """리트리버 및 RAG 체인을 초기화하고 캐싱합니다."""
    retriever_type = os.getenv("RETRIEVER_TYPE", "vector").lower()
    db_manager = ChromaDBManager(collection_name="rag_collection")

    retriever = db_manager  # 기본값: Vector 전용

    if retriever_type in ["hybrid", "ensemble"]:
        try:
            from src.core.retriever import EnsembleRetriever

            retriever = EnsembleRetriever(chroma_manager=db_manager)
            logger.info("EnsembleRetriever (Hybrid) 활성화")
        except (ImportError, ModuleNotFoundError):
            logger.warning("EnsembleRetriever를 찾을 수 없습니다. Vector 전용 모드로 실행합니다.")

    rag_chain = get_rag_chain(retriever)
    return db_manager, rag_chain


try:
    db_manager, rag_chain = initialize_rag_system()
except Exception as e:
    st.error(f"시스템 초기화 중 오류 발생: {e}")
    st.stop()

# --- 3. 초기 세션 상태 설정 (대화 기록 유지용) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# CSS 추가: 인용구 뱃지 스타일
st.markdown(
    """
<style>
.citation-badge {
    background-color: #f0f2f6;
    color: #31333F;
    padding: 0.2rem 0.5rem;
    border-radius: 0.5rem;
    font-size: 0.8em;
    margin: 0 0.1rem;
    text-decoration: none;
    border: 1px solid #dcdcdc;
    cursor: pointer;
}
.citation-badge:hover {
    background-color: #e0e2e6;
    border-color: #c0c2c6;
}
</style>
""",
    unsafe_allow_html=True,
)

# --- 4. 사이드바 (Sidebar) 구성 ---
with st.sidebar:
    st.title("설정 및 관리")

    st.subheader("검색 설정")
    k_value = st.slider(
        "초벌 검색 문서 개수 (K)",
        min_value=1,
        max_value=20,
        value=10,
        step=1,
    )
    final_k_value = st.slider(
        "리랭킹 후 최종 문서 개수",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
    )

    st.divider()
    st.subheader("데이터베이스 상태")
    count = db_manager.get_count()
    st.write(f"현재 저장된 청크 수: **{count}**")

    if st.button("상태 새로고침"):
        st.rerun()

    st.divider()
    show_expert_mode = st.toggle(
        "상세 추론 과정 보기", value=False, help="리랭킹 점수 등 전문가용 추론 과정을 표시합니다."
    )


# --- UI 스트리밍 핸들러 (UI-비즈니스 로직 분리) ---
class StreamUIHandler:
    """RAG 체인에서 발생하는 상태 이벤트를 수신하여 Streamlit UI를 업데이트하는 전용 클래스"""

    def __init__(self):
        self.retrieval_msg = st.empty()
        self.reranking_msg = st.empty()
        self.generation_msg = st.empty()
        self.response_container = st.empty()

        self.full_response = ""
        self.docs = []
        self.final_docs = []

        self.stage_handlers = {
            "retrieval": self._handle_retrieval,
            "reranking": self._handle_reranking,
            "generation": self._handle_generation,
        }

    def process_step(self, step: dict):
        stage = step.get("stage")
        state = step.get("status")

        handler = self.stage_handlers.get(stage)
        if handler:
            handler(state, step)

    def _handle_retrieval(self, state: str, step: dict):
        if state == "running":
            self.retrieval_msg.info("[검색 중] 관련 문서를 찾고 있습니다...")
        elif state == "complete":
            self.docs = step.get("output", [])
            self.retrieval_msg.success(f"[검색 완료] {len(self.docs)}개의 문서를 찾았습니다.")

    def _handle_reranking(self, state: str, step: dict):
        if state == "running":
            self.reranking_msg.info("[리랭킹 중] 가장 관련성 높은 문서를 선별하고 있습니다...")
        elif state == "complete":
            self.final_docs = step.get("output", [])
            self.reranking_msg.success(f"[리랭킹 완료] 상위 {len(self.final_docs)}개의 핵심 문서를 선별했습니다.")

    def _handle_generation(self, state: str, step: dict):
        if state == "running":
            self.generation_msg.info("[답변 생성 중] 답변을 조립하고 있습니다...")
        elif state == "streaming":
            token = step.get("output", "")
            self.full_response += token
            self.response_container.markdown(self.full_response + "▌")
        elif state == "complete":
            self.generation_msg.success("[답변 생성 완료]")
            self.response_container.markdown(self.full_response)

    def handle_error(self, error: Exception):
        if self.full_response:
            error_msg = "\n\n[시스템 안내] 통신 오류로 인해 답변이 불완전할 수 있습니다."
            self.response_container.markdown(self.full_response + error_msg)
            return self.full_response + error_msg
        else:
            st.error(f"답변 생성 중 오류가 발생했습니다: {error}")
            return "내부 시스템 오류로 답변을 생성할 수 없습니다."


# --- 5. 메인 채팅창 구성 ---
st.title("지능형 사내 규정 어시스턴트")
st.markdown("사내 매뉴얼 및 규정 문서를 기반으로 답변을 생성합니다.")
st.markdown("---")

# 대화 히스토리 및 인용 뱃지 영구 출력
for msg_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # 어시스턴트 메시지인 경우에만 저장된 인용 데이터를 버튼으로 렌더링
        if message["role"] == "assistant" and message.get("citations"):
            citations = message["citations"]

            if show_expert_mode:
                with st.expander("추론 과정 및 문서 순위 변화 분석"):
                    st.markdown("### 리랭킹 전/후 점수 비교")
                    df_data = []
                    for i, d in enumerate(citations):
                        source = d.metadata.get(MetadataFields.SRC_NAME) or "알 수 없는 파일"
                        page = d.metadata.get(MetadataFields.PG_NUM) or "-"
                        orig_score = d.metadata.get("score", 0.0)
                        rerank_score = d.metadata.get("rerank_score", 0.0)

                        df_data.append(
                            {
                                "최종 순위": i + 1,
                                "출처": source,
                                "페이지": page,
                                "초기 점수 (Vector)": orig_score,
                                "최종 점수 (Reranker)": rerank_score,
                            }
                        )
                    st.dataframe(pd.DataFrame(df_data), use_container_width=True)

            st.markdown("**참고 문서:**")
            cols = st.columns(len(citations))
            for i, doc in enumerate(citations):
                source = doc.metadata.get(MetadataFields.SRC_NAME) or "알 수 없는 파일"
                page = doc.metadata.get(MetadataFields.PG_NUM) or "-"

                with cols[i]:
                    # 재실행 시에도 버튼 키가 고정되도록 msg_idx 사용
                    if st.button(f"[{i + 1}] {source[:10]}... (p.{page})", key=f"cite_btn_{msg_idx}_{i}"):
                        show_document_dialog(source, str(page), doc.metadata.get("rerank_score", 0.0), doc.page_content)

# --- 6. 사용자 입력 및 RAG 실행 ---
if prompt := st.chat_input("규정에 대해 궁금한 점을 물어보세요."):
    # 사용자 메시지 표시 및 저장
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 어시스턴트 답변 생성
    with st.chat_message("assistant"):
        start_time = time.time()

        status = st.status("답변 생성 파이프라인 진행 중...", expanded=True)
        ui_handler = StreamUIHandler()

        try:
            for step in rag_chain.stream({"question": prompt, "k": k_value, "final_k": final_k_value}):
                ui_handler.process_step(step)

            status.update(label="파이프라인 실행 완료", state="complete", expanded=False)

            st.session_state.messages.append(
                {"role": "assistant", "content": ui_handler.full_response, "citations": ui_handler.final_docs}
            )

            end_time = time.time()
            elapsed_time = end_time - start_time
            perf_logger.log("Total", elapsed_time, prompt[:30])

            st.rerun()

        except Exception as e:
            logger.error(f"답변 생성 오류: {e}", exc_info=True)
            status.update(label="파이프라인 실행 실패", state="error", expanded=True)

            error_content = ui_handler.handle_error(e)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_content, "citations": ui_handler.final_docs}
            )

# --- 7. 푸터 ---
st.markdown("---")
st.caption("© 2026 Context-RAG-Chatbot Team. 모든 답변은 등록된 문서에 근거합니다.")
