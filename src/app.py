import logging
import math
import os
import time

import streamlit as st
from dotenv import load_dotenv

from src.common.constants import MetadataFields
from src.core.chains import get_rag_chain
from src.pipeline import PipelineOrchestrator
from src.utils.logger import PerformanceLogger, setup_global_logging
from src.utils.monitoring import get_system_stats
from src.utils.paths import RAW_DATA_DIR, ensure_directories
from src.vector_db.chroma_manager import ChromaDBManager

# 환경 변수 및 로깅 설정
load_dotenv()
setup_global_logging()
perf_logger = PerformanceLogger()  # 전용 로거 인스턴스 생성
logger = logging.getLogger(__name__)

# --- 1. 페이지 설정 ---
st.set_page_config(
    page_title="기업 매뉴얼 챗봇 (관리 시스템 통합)",
    layout="wide",
)

# 필수 디렉토리 확인 및 생성
ensure_directories()

# --- 2. 세션 상태 초기화 ---
if "admin_active" not in st.session_state:
    st.session_state.admin_active = False
if "dialog_doc_to_show" not in st.session_state:
    st.session_state.dialog_doc_to_show = None
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False


# --- 3. 팝업 다이얼로그 정의 ---


# [문서 원문 보기]
@st.dialog("문서 원문 보기")
def show_document_dialog(doc: dict):
    source = doc.get("metadata", {}).get(MetadataFields.SRC_NAME, "알 수 없음")
    page = doc.get("metadata", {}).get(MetadataFields.PG_NUM, "-")
    score = doc.get("score", 0.0)
    content = doc.get("content", "")

    st.markdown(f"**출처:** {source}")
    st.markdown(f"**페이지:** {page}")
    st.markdown(f"**관련도 점수:** {score:.4f}")
    st.text_area("원문 내용", content, height=300)
    if st.button("닫기"):
        st.rerun()


# [데이터 관리 시스템 (Admin)]
@st.dialog("데이터 관리 시스템", width="large")
def show_admin_dialog():  # noqa: C901
    st.markdown("지식 베이스(RAW_DATA) 관리 및 데이터베이스 동기화를 수행합니다.")

    # 상단 옵션 영역
    col_opt1, _ = st.columns([1, 1])
    with col_opt1:
        auto_sync = st.checkbox(
            "파일 업로드/삭제 후 자동 동기화 실행",
            value=True,
            help="체크 시 별도의 Sync 버튼 클릭 없이 즉시 DB에 반영합니다.",
        )

    st.divider()

    # 공통 동기화 로직 함수
    def trigger_sync(force=False):
        with st.status("데이터베이스 동기화 중...", expanded=True) as status:
            # 싱글톤 인스턴스 사용 (불필요한 모델 로드 방지)
            orchestrator = PipelineOrchestrator()
            orchestrator.run_ingestion(force=force)
            status.update(label="동기화 완료", state="complete", expanded=False)
        st.success("DB 동기화 완료")
        time.sleep(0.5)
        # 상태 새로고침을 위해 rerun
        st.rerun()

    # 상단 영역: 업로드 및 동기화
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("신규 문서 업로드")
        uploaded_files = st.file_uploader(
            "파일 선택 (PDF, MD)",
            accept_multiple_files=True,
            type=["pdf", "md", "markdown"],
            key="dialog_uploader",
            label_visibility="collapsed",
        )
        if st.button("업로드 실행", key="admin_upload_btn", use_container_width=True):
            if uploaded_files:
                with st.spinner("저장 중..."):
                    for uploaded_file in uploaded_files:
                        file_path = RAW_DATA_DIR / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    st.success(f"{len(uploaded_files)}개 파일 업로드 완료")

                    if auto_sync:
                        trigger_sync(force=False)
            else:
                st.warning("선택된 파일이 없습니다.")

    with col2:
        st.subheader("수동 동기화")
        st.info("자동 동기화를 껐거나, 강제 업데이트가 필요한 경우 사용하세요.")
        if st.button("데이터 파이프라인 가동 (Sync)", key="dialog_sync_btn", use_container_width=True):
            trigger_sync(force=True)

    st.divider()

    # 하단 영역: 문서 목록 및 행 단위 삭제
    st.subheader("등록된 문서 목록 및 삭제")
    if not auto_sync:
        st.caption("⚠️ 주의: 자동 동기화가 꺼져 있습니다. 삭제 후 반드시 'Sync'를 실행해야 DB에서 제거됩니다.")

    def format_size(size_bytes):
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB")
        i = math.floor(math.log(size_bytes, 1024))
        p = math.pow(1024, i)
        return f"{round(size_bytes / p, 2)} {size_name[i]}"

    current_files = []
    for ext in ["*.pdf", "*.md", "*.markdown"]:
        current_files.extend(list(RAW_DATA_DIR.glob(ext)))

    if not current_files:
        st.info("현재 등록된 문서가 없습니다.")
    else:
        h_col1, h_col2, h_col3, h_col4, h_col5 = st.columns([0.2, 3.0, 0.8, 0.6, 0.6])
        h_col1.write("**No**")
        h_col2.write("**파일명**")
        h_col3.write("**크기**")
        h_col4.write("**청크**")
        h_col5.write("**삭제**")
        st.markdown(
            "<hr style='margin: 0px 0px 10px 0px; border: 0.5px solid rgba(151,166,195,0.2);'>",
            unsafe_allow_html=True,
        )

        for i, f in enumerate(current_files):
            r_col1, r_col2, r_col3, r_col4, r_col5 = st.columns([0.2, 3.0, 0.8, 0.6, 0.6])
            r_col1.write(f"{i + 1}")
            r_col2.text(f.name)
            r_col3.write(format_size(f.stat().st_size))
            chunk_count = db_manager.get_source_count(f.name)
            r_col4.write(f"{chunk_count}")
            if r_col5.button("🗑️", key=f"del_btn_{i}", help=f"'{f.name}' 삭제") and f.exists():
                f.unlink()
                st.toast(f"파일 삭제됨: {f.name}")
                if auto_sync:
                    trigger_sync(force=False)
                else:
                    time.sleep(0.5)
                st.rerun()

    st.divider()
    if st.button("관리 시스템 종료 (닫기)", use_container_width=True):
        st.session_state.admin_active = False
        st.rerun()


# --- 4. RAG 시스템 초기화 (캐싱) ---
@st.cache_resource
def initialize_rag_system():
    retriever_type = os.getenv("RETRIEVER_TYPE", "vector").lower()
    db_manager = ChromaDBManager(collection_name="rag_collection")
    retriever = db_manager

    if retriever_type in ["hybrid", "ensemble"]:
        try:
            from src.core.retriever import EnsembleRetriever

            retriever = EnsembleRetriever(chroma_manager=db_manager)
            logger.info("EnsembleRetriever (Hybrid) 활성화")
        except (ImportError, ModuleNotFoundError):
            logger.warning("EnsembleRetriever 누락. Vector 모드로 실행합니다.")

    rag_chain = get_rag_chain(retriever)
    return db_manager, rag_chain


try:
    db_manager, rag_chain = initialize_rag_system()
except Exception as e:
    st.error(f"시스템 초기화 오류: {e}")
    st.stop()


# CSS 추가
st.markdown(
    """
<style>
.citation-badge {
    background-color: rgba(151, 166, 195, 0.15);
    padding: 0.2rem 0.5rem;
    border-radius: 0.5rem;
    font-size: 0.8em;
    margin: 0 0.1rem;
    text-decoration: none;
    border: 1px solid rgba(151, 166, 195, 0.3);
    cursor: pointer;
}
.citation-badge:hover {
    background-color: rgba(151, 166, 195, 0.25);
    border-color: rgba(151, 166, 195, 0.4);
}
div[data-testid="column"] > div > div > div > div > p {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
</style>
""",
    unsafe_allow_html=True,
)


# --- 5. 사이드바 (Sidebar) 구성 ---
with st.sidebar:
    st.title("설정 및 관리")

    st.subheader("검색 설정")
    k_value = st.slider("초벌 검색 개수 (K)", 1, 20, 10, 1, disabled=st.session_state.is_generating)
    final_k_value = st.slider("최종 선별 개수", 1, 10, 5, 1, disabled=st.session_state.is_generating)

    st.divider()
    st.subheader("데이터베이스 상태")
    count = db_manager.get_count()
    st.write(f"현재 저장된 청크 수: **{count}**")

    if st.button("상태 새로고침", use_container_width=True, disabled=st.session_state.is_generating):
        st.rerun()

    st.divider()
    show_expert_mode = st.toggle(
        "상세 추론 과정 보기",
        value=st.session_state.get("show_expert_mode", False),
        disabled=st.session_state.is_generating,
    )
    st.session_state.show_expert_mode = show_expert_mode

    st.divider()
    st.subheader("실시간 자원 모니터링")
    stats = get_system_stats()

    st.write("CPU 사용량")
    st.progress(int(stats["cpu"]), text=f"{stats['cpu']:.1f}%")

    st.write("RAM 사용량")
    st.progress(int(stats["memory"]), text=f"{stats['memory']:.1f}%")

    if stats["gpu_vram"] is not None:
        st.write("GPU VRAM 사용량")
        st.progress(int(stats["gpu_vram"]), text=f"{stats['gpu_vram']:.1f}%")
    else:
        st.write("GPU VRAM 사용량")
        st.info("현재 환경에서 GPU를 사용할 수 없습니다.")

    st.sidebar.markdown("<br>" * 2, unsafe_allow_html=True)
    if st.button("데이터 관리 시스템 실행", use_container_width=True, disabled=st.session_state.is_generating):
        st.session_state.admin_active = True
        st.rerun()

# --- 6. 다이얼로그 활성화 제어 ---
if st.session_state.get("admin_active", False):
    show_admin_dialog()


# --- UI 스트리밍 핸들러 ---
class StreamUIHandler:
    def __init__(self):
        self.response_container = st.empty()
        self.latency_placeholder = st.empty()
        self.full_response = ""
        self.final_docs = []
        self.stage_latencies = {}
        self.start_time = time.time()

    def process_step(self, step: dict):
        stage, status = step.get("stage"), step.get("status")

        if status == "running":
            self.stage_latencies[stage] = {"start": time.time()}
        elif status in ["complete", "hit", "miss"] and stage in self.stage_latencies:
            self.stage_latencies[stage]["end"] = time.time()

        if stage == "generation" and status == "streaming":
            self.full_response += step.get("output", "")
            self.response_container.markdown(self.full_response + "▌")
        elif stage == "generation" and status == "complete":
            self.response_container.markdown(self.full_response)
        elif stage == "citation" and status == "complete":
            self.final_docs = self._format_docs(step.get("source_documents", []))

        if st.session_state.show_expert_mode:
            self.display_latencies()

    def _format_docs(self, docs):
        formatted = []
        if not docs:
            return formatted
        for doc in docs:
            if hasattr(doc, "page_content"):
                formatted.append(
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": doc.metadata.get("rerank_score", doc.metadata.get("score", 0.0)),
                    }
                )
            elif isinstance(doc, dict):
                formatted.append(doc)
        return formatted

    def display_latencies(self):
        total_latency = time.time() - self.start_time
        latency_messages = [f"**총 소요시간: {total_latency:.2f}초**"]
        if total_latency > 5.0:
            latency_messages[0] = f"⚠️ {latency_messages[0]} (5초 초과)"

        for stage, times in self.stage_latencies.items():
            if "end" in times:
                duration = times["end"] - times["start"]
                latency_messages.append(f"- {stage}: {duration:.2f}초")

        self.latency_placeholder.info("\n".join(latency_messages))

    def handle_error(self, error):
        if self.full_response:
            msg = "\n\n[안내] 통신 오류로 답변이 불완전할 수 있습니다."
            self.response_container.markdown(self.full_response + msg)
            return self.full_response + msg
        return f"오류가 발생했습니다: {error}"


# --- 메인 채팅 화면 ---
st.title("📖 기업 매뉴얼 Q&A 서비스")
st.info("사내 규정 및 매뉴얼에 대해 질문하면 인용 출처와 함께 답변해 드립니다.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            for i, doc in enumerate(msg["citations"]):
                if not isinstance(doc, dict):
                    continue
                metadata = doc.get("metadata", {})
                source = metadata.get(MetadataFields.SRC_NAME, "알 수 없음")
                page = metadata.get(MetadataFields.PG_NUM, "-")
                score = doc.get("score", 0.0)

                button_label = f"📄 {source} (p.{page}) - 신뢰도: {score:.2f}"
                if score < 0.5:
                    button_label += " (낮음)"

                if st.button(button_label, key=f"cite_{msg_idx}_{i}", disabled=st.session_state.is_generating):
                    st.session_state.dialog_doc_to_show = doc
                    st.rerun()

if st.session_state.dialog_doc_to_show:
    show_document_dialog(st.session_state.dialog_doc_to_show)
    st.session_state.dialog_doc_to_show = None


def on_chat_submit():
    if st.session_state.get("admin_active"):
        st.session_state.admin_active = False
    st.session_state.is_generating = True


if prompt := st.chat_input(
    "규정에 대해 궁금한 점을 물어보세요.", on_submit=on_chat_submit, disabled=st.session_state.is_generating
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        ui_handler = StreamUIHandler()
        try:
            stream = rag_chain.stream({"question": prompt, "k": k_value, "final_k": final_k_value})
            for step in stream:
                ui_handler.process_step(step)

            duration = time.time() - ui_handler.start_time
            st.session_state.messages.append(
                {"role": "assistant", "content": ui_handler.full_response, "citations": ui_handler.final_docs}
            )

            # 로깅 데이터 준비
            log_data = {
                "query": prompt,
                "answer": ui_handler.full_response,
                "total_latency": duration,
                "latencies_per_stage": {
                    s: v["end"] - v["start"] for s, v in ui_handler.stage_latencies.items() if "end" in v
                },
                "confidence_scores": [doc.get("score", 0.0) for doc in ui_handler.final_docs],
                "system_stats": get_system_stats(),
                "status": "success",
            }
            perf_logger.log(**log_data)

            st.session_state.is_generating = False
            st.rerun()

        except Exception as e:
            duration = time.time() - ui_handler.start_time
            logger.error(f"채팅 중 오류 발생: {e}", exc_info=True)
            error_content = ui_handler.handle_error(e)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_content, "citations": ui_handler.final_docs}
            )

            log_data = {
                "query": prompt,
                "error": str(e),
                "total_latency": duration,
                "system_stats": get_system_stats(),
                "status": "error",
            }
            perf_logger.log(**log_data)

            st.session_state.is_generating = False
            st.rerun()
