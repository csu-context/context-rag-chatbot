import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
import math
import os
import time

import streamlit as st
from dotenv import load_dotenv

from src.common.constants import MetadataFields
from src.core.chains import get_rag_chain
from src.pipeline import PipelineOrchestrator
from src.utils.logger import PerformanceLogger, setup_global_logging
from src.utils.paths import RAW_DATA_DIR, ensure_directories
from src.vector_db.chroma_manager import ChromaDBManager

# 환경 변수 및 로깅 설정
load_dotenv()

# [메모리 부족 방지] 가벼운 모델 강제 지정
if not os.getenv("MODEL_NAME"):
    os.environ["MODEL_NAME"] = "phi3"

setup_global_logging()
perf_logger = PerformanceLogger()
logger = logging.getLogger(__name__)

# --- 1. 페이지 설정 ---
st.set_page_config(
    page_title="기업 매뉴얼 챗봇 (관리 시스템 통합)",
    layout="wide",
)

ensure_directories()

# --- 2. 세션 상태 초기화 ---
if "admin_active" not in st.session_state:
    st.session_state.admin_active = False


# --- 3. 팝업 다이얼로그 정의 ---

@st.dialog("문서 원문 보기")
def show_document_dialog(doc_source: str, doc_page: str, doc_score: float, doc_content: str):
    st.markdown(f"**출처:** {doc_source}")
    st.markdown(f"**페이지:** {doc_page}")
    st.markdown(f"**관련도 점수:** {doc_score:.4f}")
    st.text_area("원문 내용", doc_content, height=300)


@st.dialog("데이터 관리 시스템", width="large")
def show_admin_dialog():
    st.markdown("지식 베이스(RAW_DATA) 관리 및 데이터베이스 동기화를 수행합니다.")

    auto_sync = st.checkbox("파일 업로드/삭제 후 자동 동기화 실행", value=True)
    st.divider()

    def trigger_sync():
        with st.status("데이터베이스 동기화 중...", expanded=True) as sync_status:
            orchestrator = PipelineOrchestrator()
            orchestrator.run_ingestion()
            sync_status.update(label="동기화 완료", state="complete", expanded=False)
        st.success("DB 동기화 완료")
        time.sleep(0.5)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("신규 문서 업로드")
        uploaded_files = st.file_uploader("파일 선택 (PDF, MD)", accept_multiple_files=True, type=["pdf", "md", "markdown"])
        if st.button("업로드 실행", key="admin_upload_btn"):
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    file_path = RAW_DATA_DIR / uploaded_file.name  # type: ignore
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())  # type: ignore
                if auto_sync: trigger_sync()
                st.rerun()

    with col2:
        st.subheader("수동 동기화")
        if st.button("데이터 파이프라인 가동 (Sync)", key="dialog_sync_btn"):
            trigger_sync()
            st.rerun()

    st.divider()
    st.subheader("등록된 문서 목록")
    current_files = list(RAW_DATA_DIR.glob("*.*"))
    for i, f in enumerate(current_files):
        cols = st.columns([3, 1, 1])
        cols[0].text(f.name)
        if cols[2].button("삭제", key=f"del_{i}"):
            f.unlink()
            if auto_sync: trigger_sync()
            st.rerun()

    if st.button("관리 시스템 종료"):
        st.session_state.admin_active = False
        st.rerun()


# --- 4. RAG 시스템 초기화 ---
@st.cache_resource
def initialize_rag_system():
    global_db_manager = ChromaDBManager(collection_name="rag_collection")
    # EnsembleRetriever가 존재하는 경우 가중치 제어를 위해 반환
    try:
        from src.core.retriever import EnsembleRetriever
        retriever = EnsembleRetriever(chroma_manager=global_db_manager)
    except Exception:
        retriever = global_db_manager

    rag_chain = get_rag_chain(retriever)
    return global_db_manager, rag_chain, retriever


try:
    db_manager, rag_chain, global_retriever = initialize_rag_system()
except Exception as init_err:
    st.error(f"시스템 초기화 오류: {init_err}")
    st.stop()

# --- 5. 사이드바 (Sidebar) 구성 ---
with st.sidebar:
    st.title("설정 및 관리")
    st.divider()
    st.subheader("데이터베이스 상태")
    try:
        # DB 매니저를 통해 청크 개수 가져오기
        chunk_count = db_manager.collection.count()
        st.text(f"현재 저장된 청크 수: {chunk_count}")
    except Exception:
        st.text("상태 정보를 불러올 수 없습니다.")
    k_value = st.slider("초벌 검색 개수 (K)", 1, 10, 3, 1)
    final_k_value = st.slider("최종 선별 개수", 1, 5, 2, 1)

    st.divider()
    st.subheader("하이브리드 검색 가중치")
    bm25_weight = st.slider("BM25 (키워드) 비중", 0.0, 1.0, 0.5, 0.1)
    vector_weight = round(1.0 - bm25_weight, 1)
    st.info(f"Vector (의미) 비중: {vector_weight}")

    try:
        if hasattr(global_retriever, 'weights'):
            global_retriever.weights = [bm25_weight, vector_weight]
    except Exception:
        pass

    if st.button("데이터 관리 시스템 실행"):
        st.session_state.admin_active = True
        st.rerun()

if st.session_state.admin_active:
    show_admin_dialog()


# --- 6. 스트리밍 핸들러 ---
class StreamUIHandler:
    def __init__(self):
        self.retrieval_msg = st.empty()
        self.reranking_msg = st.empty()
        self.generation_msg = st.empty()
        self.response_container = st.empty()
        self.full_response = ""
        self.final_docs = []

    def process_step(self, step: dict):
        stage, state = step.get("stage"), step.get("status")
        if stage == "retrieval":
            if state == "running":
                self.retrieval_msg.info("관련 문서 검색 중...")
            elif state == "complete":
                self.retrieval_msg.success("검색 완료")
        elif stage == "reranking":
            if state == "complete": self.final_docs = step.get("output", [])
        elif stage == "generation":
            if state == "running":
                self.generation_msg.info("답변 생성 중...")
            elif state == "streaming":
                self.full_response += step.get("output", "")
                self.response_container.markdown(self.full_response + "▌")
            elif state == "complete":
                self.generation_msg.success("답변 완료")
                self.response_container.markdown(self.full_response)


# --- 메인 화면 ---
st.title("기업 매뉴얼 Q&A 서비스")

if prompt := st.chat_input("규정에 대해 궁금한 점을 물어보세요."):
    st.session_state.messages = st.session_state.get("messages", []) + [{"role": "user", "content": prompt}]

    with st.chat_message("assistant"):
        ui_handler = StreamUIHandler()
        try:
            for pipe_step in rag_chain.stream({
                "question": prompt, "k": k_value, "final_k": final_k_value,
                "bm25_weight": bm25_weight, "vector_weight": vector_weight
            }):
                ui_handler.process_step(pipe_step)
        except Exception as e:
            logger.error(f"오류 발생: {e}")
            st.error("답변 생성 중 오류가 발생했습니다.")
