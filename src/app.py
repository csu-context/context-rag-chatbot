import os
import sys

# macOS/Windows 전용 segfault 방지 — Linux 운영서버에는 적용 안 함
# Linux에서 스레드 수 1 고정 시 CPU Starvation/OOM 유발 가능
if sys.platform != "linux":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["RAYON_NUM_THREADS"] = "1"
else:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


import logging
import threading
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from st_copy_to_clipboard import st_copy_to_clipboard

from src.common.config import settings
from src.common.constants import MetadataFields
from src.core.chains import get_rag_chain
from src.core.retriever import RetrieverFactory
from src.models.factory import LLMFactory
from src.ui.dialogs.admin import show_admin_dialog
from src.ui.dialogs.chunk_viewer import show_chunks_viewer_dialog
from src.ui.session import init_session_state
from src.ui.stream_responder import StreamResponder
from src.utils.logger import PerformanceLogger, setup_global_logging
from src.utils.monitoring import get_system_stats
from src.utils.paths import ensure_directories

# 환경 변수 및 로깅 설정
load_dotenv()
setup_global_logging()
# Windows ProactorEventLoop에서 WebSocket 재연결 시 발생하는 알려진 무해한 오류 억제
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
perf_logger = PerformanceLogger()  # 전용 로거 인스턴스 생성
logger = logging.getLogger(__name__)


def get_doc_field(doc, field, default=None):
    """dict 형태와 LangChain Document 형태 모두 지원하는 메타데이터/내용 조회 헬퍼"""
    if hasattr(doc, "metadata"):  # LangChain Document 객체인 경우
        if field == "content":
            return getattr(doc, "page_content", default)
        return doc.metadata.get(field, default)
    elif isinstance(doc, dict):  # 일반 dict 객체인 경우
        if field == "content":
            return doc.get("content", doc.get("page_content", default))
        meta = doc.get("metadata", {})
        if isinstance(meta, dict):
            return meta.get(field, default)
    return default


def get_doc_score(doc, default=0.0):
    if hasattr(doc, "metadata"):
        return doc.metadata.get("score", getattr(doc, "score", default))
    elif isinstance(doc, dict):
        return doc.get("score", default)
    return default


# --- 1. 페이지 설정 ---
st.set_page_config(
    page_title="기업 매뉴얼 챗봇 (관리 시스템 통합)",
    layout="wide",
)

# 필수 디렉토리 확인 및 생성
ensure_directories()

# --- 2. 세션 상태 초기화 (중앙 이관 호출) ---
init_session_state()

# Redis 세션 복원 — REDIS_URL 설정 시 이전 대화 기록 복구
# 로드밸런서로 다른 인스턴스로 라우팅되어도 대화 연속성 유지
if settings.REDIS_URL and "redis_session_loaded" not in st.session_state:
    st.session_state.redis_session_loaded = True
    _session_id = st.query_params.get("sid", "") or id(st.session_state)
    st.session_state._redis_session_id = str(_session_id)
    from src.utils.redis_session import RedisSessionStore

    saved_msgs = RedisSessionStore.load_messages(st.session_state._redis_session_id)
    if saved_msgs:
        st.session_state.messages = saved_msgs
        logger.info(f"Redis에서 세션 복원 ({len(saved_msgs)}개 메시지)")


def reset_doc_dialog():
    st.session_state.dialog_doc_to_show = None


# [문서 원문 보기]
@st.dialog("문서 원문 보기", on_dismiss=reset_doc_dialog)
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
        reset_doc_dialog()
        st.rerun()


# --- 3. RAG 시스템 초기화 ---
# Stateless 개선 — DB/Retriever/RAG chain 모두 전역 캐시(세션 간 공유)
# SemanticCache 데이터는 ChromaDB에 저장되므로 인스턴스 공유해도 충돌 없음
@st.cache_resource
def initialize_rag_system():
    retriever_type = os.getenv("RETRIEVER_TYPE", "vector").lower()
    from src.vector_db.chroma_manager import ChromaDBManager

    db_manager = ChromaDBManager()
    retriever = RetrieverFactory.create_retriever(retriever_type=retriever_type, chroma_manager=db_manager)
    # 임베더 eager load (첫 질문 지연 제거)
    # hybrid: retriever.chroma 가 ChromaDBManager / vector: retriever 자체가 ChromaDBManager
    chroma_mgr = getattr(retriever, "chroma", None) or retriever
    if hasattr(chroma_mgr, "embedding_fn"):
        _ = chroma_mgr.embedding_fn.embedder
    rag_chain = get_rag_chain(retriever)
    return db_manager, rag_chain


try:
    db_manager, rag_chain = initialize_rag_system()
except Exception as e:
    st.error(f"시스템 초기화 오류: {e}")
    st.stop()

# 리랭커 Eager Loading (백그라운드 스레드, 최초 1회만)
if not st.session_state.get("reranker_eager_load_started"):
    st.session_state.reranker_eager_load_started = True
    if settings.RERANKER_TYPE.lower() == "local":
        from src.core.reranker import CrossEncoderReranker

        CrossEncoderReranker.eager_load_background()

# --- 모델 다운로드 및 준비 상태 사전 체크 ---
model_ready = True
is_ollama = settings.MODEL_TYPE == "ollama"

if is_ollama:
    try:
        llm_instance = LLMFactory.create_llm()
        if not llm_instance.is_model_available():
            model_ready = False
        elif "llm_warmup_done" not in st.session_state:
            st.session_state.llm_warmup_done = True
            # 동기 warmup은 첫 페이지 로드를 멈춤 → 데몬 스레드로 백그라운드 실행 (리랭커 패턴과 동일)
            threading.Thread(target=llm_instance.warmup, daemon=True, name="ollama-warmup").start()
    except Exception as e:
        model_ready = False
        logger.error(f"로컬 LLM 상태 진단 중 오류: {e}")


# CSS 추가 (style.css에서 동적 주입)
css_file_path = Path(__file__).parent / "ui" / "style.css"
if css_file_path.exists():
    with open(css_file_path, encoding="utf-8") as f:
        custom_css = f.read()
    st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)


# --- 4. 사이드바 (Sidebar) 구성 ---
with st.sidebar:
    st.title("설정 및 관리")

    # 1. 시스템 정보
    st.markdown(f"**LLM:** `{settings.MODEL_NAME}` ({settings.MODEL_TYPE.upper()})")
    st.markdown(f"**임베딩:** `{settings.EMBEDDING_MODEL_NAME.split('/')[-1]}`")
    st.divider()

    # 2. 검색 설정
    st.subheader("검색 설정")
    k_value = st.select_slider(
        "초벌 검색 개수 (K)", options=[10, 20, 30, 40, 50], value=50, disabled=st.session_state.is_generating
    )
    final_k_value = st.slider("최종 선별 개수", 1, 5, 5, 1, disabled=st.session_state.is_generating)
    st.divider()

    # 3. 데이터베이스 상태 및 관리 (가로 버튼 배치로 통합)
    st.subheader("데이터베이스 상태 및 관리")
    count = db_manager.get_count()
    st.write(f"현재 저장된 청크 수: **{count}**")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("상태 새로고침", use_container_width=True, disabled=st.session_state.is_generating):
            st.session_state.should_rerun_app = True
    with col2:
        if st.button("데이터 관리", use_container_width=True, disabled=st.session_state.is_generating):
            st.session_state.admin_active = True
            st.session_state.should_rerun_app = True
    st.divider()

    # 4. 기타 설정
    st.toggle("상세 추론 과정 보기", key="show_expert_mode", disabled=st.session_state.is_generating)

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

# --- 5. 다이얼로그 활성화 제어 (모듈화 이관 호출) ---
if st.session_state.get("admin_active", False):
    show_admin_dialog(db_manager, initialize_rag_system.clear)

if st.session_state.get("dialog_chunks_file_to_show"):
    show_chunks_viewer_dialog(st.session_state.dialog_chunks_file_to_show, db_manager)


# --- UI 스트리밍 핸들러 함수들 ---
def format_doc_status(docs_list):
    if not docs_list:
        return "0개 문서 (0개 청크)"
    unique_files = set()
    for d in docs_list:
        src_name = str(get_doc_field(d, MetadataFields.SRC_NAME, "알 수 없음"))
        unique_files.add(src_name)
    return f"{len(unique_files)}개 문서 ({len(docs_list)}개 청크)"


# --- 메인 채팅 화면 ---
st.title("기업 매뉴얼 Q&A 서비스")
st.info("사내 규정 및 매뉴얼에 대해 질문하면 인용 출처와 함께 답변해 드립니다.")


# --- 6.1. Ollama 모델 다운로드 실시간 상태 시각화 ---
@st.fragment(run_every="1s")
def render_download_progress(llm):
    # 1단계: Ollama 서비스 자체 구동 여부 먼저 확인
    if not llm.check_health():
        st.error(f"Ollama 서비스({llm.base_url})에 연결할 수 없습니다. Ollama가 실행 중인지 확인해 주세요.")
        st.info("터미널에서 `ollama serve` 명령으로 Ollama를 실행한 후 새로고침하세요.")
        return

    # 2단계: 서비스는 살아있으나 모델이 없는 경우 다운로드 시작
    llm.start_pull_background()

    # 이중 안전 체크: 백그라운드 진행 상태와 무관하게 실제 모델 다운로드가 완료되었는지 검증
    try:
        if llm.is_model_available():
            st.success("다운로드 완료. 잠시 후 서비스가 시작됩니다.")
            time.sleep(1.0)
            st.rerun()
            return
    except Exception as e:
        logger.error(f"다운로드 상태 검증 중 오류: {e}")

    # 현재 상태 조회
    progress = llm.get_pull_status()
    if not progress:
        st.info("다운로드 준비 중...")
        return

    status = progress.get("status", "")
    if status == "success":
        st.success("다운로드 완료. 잠시 후 서비스가 시작됩니다.")
        time.sleep(1.0)
        st.rerun()
    elif "downloading" in status or status.startswith("pulling"):
        completed = progress.get("completed", 0)
        total = progress.get("total", 0)
        if total > 0:
            percentage = completed / total
            msg = (
                f"로컬 LLM 모델 '{settings.MODEL_NAME}'을 다운로드하고 있습니다. "
                "다운로드가 완료될 때까지 질문을 입력할 수 없습니다."
            )
            st.warning(msg)
            st.progress(percentage)
            completed_mb = completed / (1024 * 1024)
            total_mb = total / (1024 * 1024)
            st.text(f"다운로드 중: {percentage:.1%} ({completed_mb:.1f} MB / {total_mb:.1f} MB)")
        else:
            msg = (
                f"로컬 LLM 모델 '{settings.MODEL_NAME}'을 다운로드하고 있습니다. "
                "다운로드가 완료될 때까지 질문을 입력할 수 없습니다."
            )
            st.warning(msg)
            st.text(f"상태: {status}")
    elif status == "error":
        st.error(f"오류: {progress.get('message', '알 수 없는 오류')}")
    else:
        msg = (
            f"로컬 LLM 모델 '{settings.MODEL_NAME}'을 다운로드하고 있습니다. "
            "다운로드가 완료될 때까지 질문을 입력할 수 없습니다."
        )
        st.warning(msg)
        st.text(f"상태: {status}")


if is_ollama and not model_ready:
    try:
        llm_instance = LLMFactory.create_llm()
        render_download_progress(llm_instance)
    except Exception as e:
        st.error(f"로컬 LLM 상태 진단 중 오류가 발생했습니다: {e}")


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # 답변 소요 시간 캡션 표시
        if msg["role"] == "assistant" and msg.get("latency") is not None:
            st.caption(f"답변 소요 시간: {msg['latency']:.2f}초")

        # 인용 출처(citations)를 먼저 화면에 띄웁니다
        if msg.get("citations"):
            for i, doc in enumerate(msg["citations"]):
                if not isinstance(doc, dict):
                    continue
                metadata = doc.get("metadata", {})
                source = metadata.get(MetadataFields.SRC_NAME, "알 수 없음")
                page = metadata.get(MetadataFields.PG_NUM, "-")
                score = metadata.get("rerank_score", doc.get("score", 0.0))
                display_score = max(0.0, (score - 0.5) * 2)
                is_low_confidence = score < 0.5

                button_label = f"📄 {source} (p.{page}) - 신뢰도: {display_score:.2f}"
                if is_low_confidence:
                    button_label += " ⚠️"

                if st.button(button_label, key=f"cite_{msg_idx}_{i}", disabled=st.session_state.is_generating):
                    st.session_state.dialog_doc_to_show = doc
                    st.session_state.should_rerun_app = True

                if is_low_confidence:
                    st.caption("⚠️ 신뢰도가 낮아 환각 발생 가능성이 있습니다. 원문을 직접 확인하세요.")

        # AI 답변 클립보드 복사 버튼
        if msg["role"] == "assistant":
            st_copy_to_clipboard(
                msg["content"],
                before_copy_label="❐",
                after_copy_label="✓",
                key=f"copy_btn_{msg_idx}",
            )

        if msg.get("citations"):
            for i, doc in enumerate(msg["citations"]):
                if not isinstance(doc, dict):
                    continue
                metadata = doc.get("metadata", {})
                source = metadata.get(MetadataFields.SRC_NAME, "알 수 없음")
                page = metadata.get(MetadataFields.PG_NUM, "-")
                score = metadata.get("rerank_score", doc.get("score", 0.0))
                display_score = max(0.0, (score - 0.5) * 2)
                is_low_confidence = score < 0.5

                button_label = f"📄 {source} (p.{page}) - 신뢰도: {display_score:.2f}"
                if is_low_confidence:
                    button_label += " ⚠️"

                if st.button(button_label, key=f"cite_{msg_idx}_{i}", disabled=st.session_state.is_generating):
                    st.session_state.dialog_doc_to_show = doc
                    st.session_state.should_rerun_app = True  # Set flag instead of direct rerun

                if is_low_confidence:
                    st.caption("⚠️ 신뢰도가 낮아 환각 발생 가능성이 있습니다. 원문을 직접 확인하세요.")

if st.session_state.dialog_doc_to_show:
    show_document_dialog(st.session_state.dialog_doc_to_show)


def on_chat_submit():
    if st.session_state.get("admin_active"):
        st.session_state.admin_active = False
    st.session_state.is_generating = True


if prompt := st.chat_input(
    "규정에 대해 궁금한 점을 물어보세요.",
    on_submit=on_chat_submit,
    disabled=st.session_state.is_generating or not model_ready,
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.stop_generation = False
    st.session_state.current_prompt = prompt
    st.session_state.stream_steps = []
    st.session_state.full_response = ""
    st.session_state.docs = []
    st.session_state.final_docs = []
    st.session_state.start_time = time.time()

    st.session_state.stream_iter = rag_chain.stream(
        {
            "question": prompt,
            "k": k_value,
            "final_k": final_k_value,
            "history": st.session_state.messages[:-1],
        }
    )
    st.rerun()


# active generation UI (컨트롤러로 비즈니스 논리 이관 호출)
if st.session_state.is_generating:
    with st.chat_message("assistant"):
        responder = StreamResponder(perf_logger=perf_logger, get_system_stats_fn=get_system_stats)

        # Re-render accumulated steps
        for step in st.session_state.stream_steps:
            responder.process_step(step)

        # 가로 배치를 위한 컬럼 생성 (텍스트 밀림 방지 및 세련된 레이아웃 확보)
        col_status, col_stop = st.columns([8, 2], vertical_alignment="center")

        with col_stop:
            if st.button("생성 중단", key="stop_generation_btn", use_container_width=True):
                if st.session_state.stream_iter:
                    try:
                        st.session_state.stream_iter.close()
                    except Exception as e:
                        logger.warning(f"Error closing stream iterator: {e}")
                    st.session_state.stop_generation = True
                else:
                    st.session_state.is_generating = False
                    st.session_state.stop_generation = False
                st.rerun()

        with col_status:
            responder.consume_stream()

# --- Controlled Rerun at the end of the script ---
if st.session_state.should_rerun_app:
    st.session_state.should_rerun_app = False
    st.rerun()
