import logging
import os
import time

import streamlit as st
from dotenv import load_dotenv

from src.core.chains import get_rag_chain
from src.pipeline import PipelineOrchestrator
from src.utils.logger import setup_global_logging
from src.utils.paths import RAW_DATA_DIR, ensure_directories
from src.vector_db.chroma_manager import ChromaDBManager

# ==============================================================================
# 시스템 초기화 및 환경 설정
# ==============================================================================
load_dotenv()

# 경고 로그 억제 및 기본 모델 환경 변수 할당
logging.getLogger("transformers").setLevel(logging.ERROR)
if not os.getenv("MODEL_NAME"):
    os.environ["MODEL_NAME"] = "phi3"

setup_global_logging()
logger = logging.getLogger(__name__)

# ==============================================================================
# Streamlit 페이지 및 세션 상태 설정
# ==============================================================================
st.set_page_config(
    page_title="기업 매뉴얼 챗봇 (관리 시스템 통합)",
    layout="wide",
)

ensure_directories()

# 전역 세션 상태(Session State) 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "admin_active" not in st.session_state:
    st.session_state.admin_active = False
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = ""


# ==============================================================================
# UI 컴포넌트: 모달 다이얼로그
# ==============================================================================
@st.dialog("문서 원문 보기")
def show_document_dialog(doc_source: str, doc_page: str, doc_score: float, doc_content: str):
    """검색된 문서의 상세 메타데이터 및 원문을 출력하는 다이얼로그입니다."""
    st.markdown(f"**출처:** {doc_source}")
    st.markdown(f"**페이지:** {doc_page}")
    st.markdown(f"**관련도 점수:** {doc_score:.4f}")
    st.text_area("원문 내용", doc_content, height=300)


@st.dialog("데이터 관리 시스템", width="large")
def show_admin_dialog():
    """지식 베이스(Raw Data) 관리 및 데이터베이스 동기화를 수행하는 관리자 인터페이스입니다."""
    st.markdown("지식 베이스(RAW_DATA) 관리 및 데이터베이스 동기화를 수행합니다.")

    auto_sync = st.checkbox("파일 업로드/삭제 후 자동 동기화 실행", value=True)
    st.divider()

    def trigger_sync():
        """데이터 수집 파이프라인을 가동하여 벡터 DB 동기화를 수행합니다."""
        with st.status("데이터베이스 동기화 중...", expanded=True) as sync_status:
            orchestrator = PipelineOrchestrator()
            orchestrator.run_ingestion()
            sync_status.update(label="동기화 완료", state="complete", expanded=False)
        st.success("DB 동기화 완료")
        time.sleep(0.5)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("신규 문서 업로드")
        uploaded_files = st.file_uploader(
            "파일 선택 (PDF, MD)", accept_multiple_files=True, type=["pdf", "md", "markdown"]
        )
        # 중첩된 if문을 and 연산자로 결합하여 복잡도를 낮춥니다.
        if st.button("업로드 실행", key="admin_upload_btn") and uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = RAW_DATA_DIR / uploaded_file.name  # type: ignore
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())  # type: ignore
            if auto_sync:
                trigger_sync()
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
            if auto_sync:
                trigger_sync()
            st.rerun()

    if st.button("관리 시스템 종료"):
        st.session_state.admin_active = False
        st.rerun()


# ==============================================================================
# RAG 파이프라인 초기화
# ==============================================================================
@st.cache_resource
def initialize_rag_system():
    """
    ChromaDB Manager 및 검색기(Retriever), RAG 체인을 초기화하고 캐싱합니다.
    """
    import importlib

    _global_db_manager = ChromaDBManager(collection_name="rag_collection")
    _retriever = _global_db_manager

    try:
        # 동적 임포트를 사용하여 IDE의 정적 분석 경고를 원천 차단합니다.
        retriever_module = importlib.import_module("src.core.retriever")
        _retriever = retriever_module.EnsembleRetriever(chroma_manager=_global_db_manager)
    except (ImportError, AttributeError):
        pass

    _rag_chain = get_rag_chain(_retriever)
    return _global_db_manager, _rag_chain, _retriever


try:
    db_manager, rag_chain, global_retriever = initialize_rag_system()
except Exception as init_err:
    st.error(f"시스템 초기화 중 치명적 오류가 발생했습니다: {init_err}")
    st.stop()

# ==============================================================================
# UI 컴포넌트: 사이드바 (Sidebar)
# ==============================================================================
with st.sidebar:
    st.title("설정 및 관리")
    st.divider()

    # 데이터베이스 상태 모니터링
    st.subheader("데이터베이스 상태")
    # noinspection PyBroadException
    try:
        chunk_count = db_manager.collection.count()
        st.text(f"현재 저장된 청크 수: {chunk_count}")
    except Exception:
        st.text("상태 정보를 불러올 수 없습니다.")

    k_value = st.slider("초벌 검색 개수 (K)", 1, 10, 3, 1)
    final_k_value = st.slider("최종 선별 개수", 1, 5, 2, 1)

    st.divider()

    # 하이브리드 검색 가중치 제어
    st.subheader("하이브리드 검색 가중치")
    bm25_weight = st.slider("BM25 (키워드) 비중", 0.0, 1.0, 0.5, 0.1)
    vector_weight = round(1.0 - bm25_weight, 1)
    st.info(f"Vector (의미) 비중: {vector_weight}")

    # 리트리버 가중치 동적 업데이트
    # noinspection PyBroadException
    try:
        if hasattr(global_retriever, "weights"):
            global_retriever.weights = [bm25_weight, vector_weight]
    except Exception:
        pass

    if st.button("데이터 관리 시스템 실행"):
        st.session_state.admin_active = True
        st.rerun()

if st.session_state.admin_active:
    show_admin_dialog()


# ==============================================================================
# 스트리밍 응답 핸들러
# ==============================================================================
class StreamUIHandler:
    """RAG 체인 실행 과정의 상태 및 스트리밍 응답을 UI에 렌더링하는 핸들러 클래스입니다."""

    def __init__(self):
        self.retrieval_msg = st.empty()
        self.reranking_msg = st.empty()
        self.generation_msg = st.empty()
        self.response_container = st.empty()
        self.full_response = ""
        self.final_docs = []

    def process_step(self, step: dict):
        """파이프라인의 각 단계(Step) 정보를 분석하여 UI를 업데이트합니다."""
        stage, state = step.get("stage"), step.get("status")

        if stage == "retrieval":
            if state == "running":
                self.retrieval_msg.info("관련 문서 검색 진행 중...")
            elif state == "complete":
                self.retrieval_msg.success("검색 완료")

        elif stage == "reranking":
            if state == "complete":
                self.final_docs = step.get("output", [])

        elif stage == "generation":
            if state == "running":
                self.generation_msg.info("답변 생성 진행 중...")
            elif state == "streaming":
                self.full_response += step.get("output", "")
                self.response_container.markdown(self.full_response + "▌")
            elif state == "complete":
                self.generation_msg.success("답변 생성 완료")
                self.response_container.markdown(self.full_response)


# ==============================================================================
# 메인 채팅 인터페이스
# ==============================================================================
st.title("기업 매뉴얼 Q&A 서비스")

# 1. 대화 히스토리 렌더링
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. 사용자 입력 처리 및 상태 전환
if prompt := st.chat_input("규정에 대해 궁금한 점을 물어보세요."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.current_prompt = prompt
    st.session_state.is_generating = True
    st.rerun()

# 3. 비동기/스트리밍 응답 생성 로직
if st.session_state.is_generating:
    with st.chat_message("assistant"):
        ui_handler = StreamUIHandler()
        try:
            for pipe_step in rag_chain.stream(
                {
                    "question": st.session_state.current_prompt,
                    "k": k_value,
                    "final_k": final_k_value,
                    "bm25_weight": bm25_weight,
                    "vector_weight": vector_weight,
                }
            ):
                ui_handler.process_step(pipe_step)

            # 생성 완료 후 컨텍스트 저장
            st.session_state.messages.append({"role": "assistant", "content": ui_handler.full_response})

        except Exception as e:
            logger.error(f"응답 파이프라인 처리 중 예외 발생: {e}")
            st.error("답변 생성 중 서버 오류가 발생했습니다.")

        finally:
            st.session_state.is_generating = False
