import logging
import math
import os
import time

import streamlit as st
from dotenv import load_dotenv

from src.core.chains import get_rag_chain
from src.pipeline import PipelineOrchestrator
from src.utils.health_check import repair_integrity, run_full_diagnostics
from src.utils.logger import setup_global_logging
from src.utils.paths import RAW_DATA_DIR, ensure_directories
from src.vector_db.chroma_manager import ChromaDBManager


# ==============================================================================
# 팀원 추가 메타데이터 필드 호환 상수를 주석 변형 없이 안전하게 매핑
# ==============================================================================
class MetadataFields:
    RELATIVE_PATH = "relative_path"
    SRC_NAME = "src_name"
    PARSER_TYPE = "parser_type"


def reset_admin_active():
    st.session_state.admin_active = False


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


@st.cache_data(ttl=30)  # 30초 캐싱
def _get_file_parser_map(_db_manager) -> dict[str, str]:
    """ChromaDB에서 파일별 파서 타입을 조회하여 캐싱합니다."""
    try:
        all_metas = _db_manager.collection.get(include=["metadatas"])["metadatas"]
        result = {}
        for m in all_metas:
            # relative_path가 있으면 우선 사용, 없으면 파일명(src_name) 사용
            rel_path = m.get(MetadataFields.RELATIVE_PATH) or m.get(MetadataFields.SRC_NAME)
            if rel_path and rel_path not in result:
                # 구 스키마('parser') 호환성 유지
                result[rel_path] = m.get(MetadataFields.PARSER_TYPE) or m.get("parser", "manual")
        return result
    except Exception as e:
        logger.warning(f"파서 맵 로드 중 오류: {e}")
        return {}


# [데이터 관리 시스템 (Admin)]
@st.dialog("데이터 관리 시스템", width="large", on_dismiss=reset_admin_active)
def show_admin_dialog(db_manager):  # noqa: C901
    st.markdown("지식 베이스(RAW_DATA) 관리 및 데이터베이스 동기화를 수행합니다.")

    auto_sync = st.checkbox("파일 업로드/삭제 후 자동 동기화 실행", value=True)
    st.divider()

    def trigger_sync():
        """데이터 수집 파이프라인을 가동하여 벡터 DB 동기화를 수행합니다."""
        with st.status("데이터베이스 동기화 중...", expanded=True) as sync_status:
            orchestrator = PipelineOrchestrator()
            orchestrator.run_ingestion()
            sync_status.update(label="동기화 완료", state="complete", expanded=False)
            initialize_rag_system.clear()
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


    # 하단 영역: 문서 목록 및 행 단위 삭제
    st.subheader("등록된 문서 목록 및 관리")
    if not auto_sync:
        st.caption("주의: 자동 동기화가 꺼져 있습니다. 변경 후 반드시 'Sync'를 실행해야 DB에 반영됩니다.")

    def format_size(size_bytes):
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB")
        i = math.floor(math.log(size_bytes, 1024))
        p = math.pow(1024, i)
        return f"{round(size_bytes / p, 2)} {size_name[i]}"

    current_files = []
    # 하위 디렉토리까지 포함하여 재귀적으로 스캔
    for ext in ["*.pdf", "*.md", "*.markdown"]:
        current_files.extend(list(RAW_DATA_DIR.glob(f"**/{ext}")))

    if not current_files:
        st.info("현재 등록된 문서가 없습니다.")
    else:
        # 컬럼 레이아웃 조정 (파서 타입 및 개별 동기화 추가)
        h_col1, h_col2, h_col3, h_col4, h_col5, h_col6, h_col7 = st.columns([0.2, 2.5, 0.8, 0.8, 0.6, 0.6, 0.6])
        h_col1.write("**No**")
        h_col2.write("**파일명**")
        h_col3.write("**크기**")
        h_col4.write("**파서**")
        h_col5.write("**청크**")
        h_col6.write("**동기화**")
        h_col7.write("**삭제**")
        st.markdown(
            "<hr style='margin: 0px 0px 10px 0px; border: 0.5px solid rgba(151,166,195,0.2);'>",
            unsafe_allow_html=True,
        )

        # ChromaDB에서 실제 메타데이터를 가져와 파서 타입 확인 (캐싱된 헬퍼 사용)
        file_parser_map = _get_file_parser_map(db_manager)

        for i, f in enumerate(current_files):
            # 상대 경로 계산
            rel_path = str(f.relative_to(RAW_DATA_DIR))
            r_col1, r_col2, r_col3, r_col4, r_col5, r_col6, r_col7 = st.columns([0.2, 2.5, 0.8, 0.8, 0.6, 0.6, 0.6])
            r_col1.write(f"{i + 1}")
            r_col2.text(rel_path)  # 파일명 대신 상대 경로 표시
            r_col3.write(format_size(f.stat().st_size))

            # 파서 타입 표시
            parser_type = file_parser_map.get(rel_path, "-")
            parser_color = "blue" if parser_type == "docling" else "green" if parser_type == "manual" else "gray"
            r_col4.markdown(f":{parser_color}[{parser_type}]")

            # 청크 수 조회 시에도 상대 경로 또는 파일명 사용 (Legacy 대응)
            chunk_count = db_manager.get_source_count(f.name)
            if chunk_count == 0 and rel_path != f.name:
                # relative_path로 다시 시도
                # (db_manager.get_source_count가 MetadataFields.SRC_NAME만 볼 경우 대응 필요)
                pass
            r_col5.write(f"{chunk_count}")

            # 개별 동기화 버튼
            if r_col6.button("🔄", key=f"sync_btn_{i}", help=f"'{f.name}' 개별 동기화"):
                orchestrator = PipelineOrchestrator()
                with st.spinner(f"{f.name} 동기화 중..."):
                    if orchestrator.process_single_file(f):
                        st.toast(f"동기화 완료: {f.name}")
                        initialize_rag_system.clear()
                        time.sleep(0.5)
                        st.rerun()

            # 삭제 버튼
            if r_col7.button("🗑️", key=f"del_btn_{i}", help=f"'{f.name}' 삭제") and f.exists():
                f.unlink()
                st.toast(f"파일 삭제됨: {f.name}")
                if auto_sync:
                    trigger_sync(force=False)
                else:
                    time.sleep(0.5)
                    st.session_state.should_rerun_app = True  # Set flag instead of direct rerun

    st.divider()

    # --- 자가 진단 및 정합성 리포트 영역 ---
    st.subheader("데이터 정합성 자가 진단")
    if st.button("진단 리포트 생성", key="health_check_btn", use_container_width=True):
        with st.spinner("시스템 진단 중..."):
            _is_healthy, report = run_full_diagnostics(silent=True, check_model=False)
            st.session_state.health_report = report

    if "health_report" in st.session_state:
        report = st.session_state.health_report
        anomalies = report["db"].get("anomalies", {})

        ghosts = anomalies.get("ghost_chunks", [])
        mismatches = anomalies.get("mismatched_hash", [])
        duplicates = anomalies.get("duplicate_parsers", [])

        has_issue = ghosts or mismatches or duplicates

        if not has_issue:
            st.success("✅ 모든 데이터가 정합성을 유지하고 있습니다.")
        else:
            if ghosts:
                st.error(f"⚠️ 유령 청크 감지: {len(ghosts)}개 파일의 데이터가 DB에 남아있습니다.")
            if mismatches:
                st.warning(f"⚠️ 업데이트 필요: {len(mismatches)}개 파일의 내용이 DB와 다릅니다.")
            if duplicates:
                st.error(f"⚠️ 중복 적재 감지: {len(duplicates)}개 파일에 여러 파서 데이터가 공존합니다.")

            if st.button("정합성 자동 복구 (Repair)", type="primary", use_container_width=True):
                with st.spinner("복구 작업 진행 중..."):
                    repair_integrity(anomalies)
                    st.success("복구가 완료되었습니다. 상태를 재확인하세요.")
                    del st.session_state.health_report
                    initialize_rag_system.clear()
                    time.sleep(0.5)
                    st.rerun()

    st.divider()
    if st.button("관리 시스템 종료 (닫기)", use_container_width=True):
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
        if hasattr(global_retriever, 'weight_bm25'):
            global_retriever.weight_bm25 = bm25_weight
            global_retriever.weight_vector = vector_weight
    except Exception:
        pass

    if st.button("데이터 관리 시스템 실행"):
        st.session_state.admin_active = True
        st.rerun()

if st.session_state.admin_active:
    show_admin_dialog(db_manager)


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
