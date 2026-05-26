import json
import logging
import math
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.common.config import settings
from src.common.constants import MetadataFields
from src.core.chains import get_rag_chain
from src.models.factory import LLMFactory
from src.pipeline import PipelineOrchestrator
from src.utils.health_check import repair_integrity, run_full_diagnostics
from src.utils.logger import PerformanceLogger, setup_global_logging
from src.utils.monitoring import get_system_stats
from src.utils.paths import RAW_DATA_DIR, ensure_directories
from src.vector_db.chroma_manager import ChromaDBManager

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

# --- 2. 세션 상태 초기화 ---
if "admin_active" not in st.session_state:
    st.session_state.admin_active = False
if "dialog_doc_to_show" not in st.session_state:
    st.session_state.dialog_doc_to_show = None
if "dialog_chunks_file_to_show" not in st.session_state:
    st.session_state.dialog_chunks_file_to_show = None
if "parser_change_pending" not in st.session_state:
    st.session_state.parser_change_pending = None
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "should_rerun_app" not in st.session_state:
    st.session_state.should_rerun_app = False
if "stop_generation" not in st.session_state:
    st.session_state.stop_generation = False
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = ""
if "stream_iter" not in st.session_state:
    st.session_state.stream_iter = None
if "stream_steps" not in st.session_state:
    st.session_state.stream_steps = []
if "full_response" not in st.session_state:
    st.session_state.full_response = ""
if "docs" not in st.session_state:
    st.session_state.docs = []
if "final_docs" not in st.session_state:
    st.session_state.final_docs = []
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "show_expert_mode" not in st.session_state:
    st.session_state.show_expert_mode = False


# --- 3. 팝업 다이얼로그 정의 ---


def reset_chunks_viewer():
    st.session_state.dialog_chunks_file_to_show = None
    if "chunk_viewer_page" in st.session_state:
        st.session_state.chunk_viewer_page = 0
    st.session_state.admin_active = True
    st.session_state.should_rerun_app = True


@st.dialog("문서 청크 목록 및 메타데이터 상세", width="large", on_dismiss=reset_chunks_viewer)
def show_chunks_viewer_dialog(file_name, db_manager):  # noqa: C901
    st.subheader(f"문서명: {file_name}")

    # 1. 파일명에 해당하는 해시(source_id) 구하기
    import unicodedata

    orchestrator = PipelineOrchestrator()
    manifest = orchestrator._load_manifest()
    manifest_files = manifest.get("files", {})

    target_hash = None
    normalized_file_name = unicodedata.normalize("NFC", file_name)
    for rel_path, info in manifest_files.items():
        normalized_rel_name = unicodedata.normalize("NFC", Path(rel_path).name)
        if normalized_rel_name == normalized_file_name:
            target_hash = info.get("hash")
            break

    # 2. 전처리 JSON 파일에서 자식 chunk_id와 부모 parent_text의 맵 빌드
    parent_text_map = {}
    if target_hash:
        from src.utils.paths import PROCESSED_DATA_DIR

        json_path = PROCESSED_DATA_DIR / f"{target_hash}.json"
        if json_path.exists():
            try:
                with open(json_path, encoding="utf-8") as jf:
                    processed_data = json.load(jf)
                for parent_item in processed_data:
                    p_text = parent_item.get("parent_text", "")
                    for child in parent_item.get("children", []):
                        c_id = child.get("chunk_id")
                        if c_id:
                            parent_text_map[c_id] = p_text
            except Exception as json_e:
                logger.error(f"전처리 JSON 파일 로드 실패: {json_e}")

    # 3. ChromaDB 청크 목록 조회 및 페이징 구성
    chunks = db_manager.get_source_chunks(file_name)
    total_chunks = len(chunks)
    st.write(f"총 청크 수: {total_chunks}개")

    with st.container(height=550):
        if not chunks:
            st.info("이 문서에 저장된 청크 데이터가 없습니다.")
        else:
            page_size = 5
            total_pages = max(1, math.ceil(total_chunks / page_size))

            if "chunk_viewer_page" not in st.session_state:
                st.session_state.chunk_viewer_page = 0

            current_page = st.session_state.chunk_viewer_page
            if current_page >= total_pages:
                current_page = total_pages - 1
                st.session_state.chunk_viewer_page = current_page
            elif current_page < 0:
                current_page = 0
                st.session_state.chunk_viewer_page = current_page

            start_idx = current_page * page_size
            end_idx = min(start_idx + page_size, total_chunks)

            st.write(f"표시 중: {start_idx + 1} ~ {end_idx} 번 청크")

            for idx in range(start_idx, end_idx):
                chunk = chunks[idx]
                c_id = chunk["id"]
                st.markdown(f"### Chunk {idx + 1} (ID: `{c_id}`)")
                st.write("**청크 내용 (자식 텍스트)**")
                import html

                escaped_content = html.escape(chunk["content"])
                st.markdown(
                    f"""<div style="
                        background-color: rgba(151,166,195,0.08);
                        padding: 12px;
                        border-radius: 6px;
                        border: 1px solid rgba(151,166,195,0.2);
                        max-height: 180px;
                        overflow-y: auto;
                        white-space: pre-wrap;
                        font-size: 0.95rem;
                        line-height: 1.5;
                        color: #f0f2f6;
                        margin-bottom: 10px;
                    ">{escaped_content}</div>""",
                    unsafe_allow_html=True,
                )

                # 매핑된 상위 부모 텍스트가 있을 경우 익스팬더로 표시
                parent_text = parent_text_map.get(c_id)
                if parent_text:
                    with st.expander("상위 부모 문맥 (Parent Context)"):
                        escaped_parent = html.escape(parent_text)
                        st.markdown(
                            f"""<div style="
                                background-color: rgba(151,166,195,0.04);
                                padding: 12px;
                                border-radius: 6px;
                                border: 1px solid rgba(151,166,195,0.15);
                                max-height: 250px;
                                overflow-y: auto;
                                white-space: pre-wrap;
                                font-size: 0.95rem;
                                line-height: 1.5;
                                color: #e0e2e6;
                            ">{escaped_parent}</div>""",
                            unsafe_allow_html=True,
                        )

                # 메타데이터 상세 정보는 기본으로 접힌 상태로 렌더링
                with st.expander("청크 메타데이터 상세 (Metadata)"):
                    st.json(chunk["metadata"])

                st.divider()

            # 숫자 버튼형 페이징 네비게이션 구성 (최대 10개 표시 슬라이딩 윈도우)
            max_visible = 10
            start_page = max(0, current_page - max_visible // 2)
            end_page = min(total_pages, start_page + max_visible)
            if end_page - start_page < max_visible:
                start_page = max(0, end_page - max_visible)

            cols_width = [1.2] + [1.0] * (end_page - start_page) + [1.2]
            cols = st.columns(cols_width)

            with cols[0]:
                if st.button(
                    "이전",
                    disabled=(current_page == 0),
                    use_container_width=True,
                    key="chunk_prev_btn",
                ):
                    st.session_state.chunk_viewer_page = current_page - 1
                    st.rerun()

            for idx, page_idx in enumerate(range(start_page, end_page)):
                with cols[idx + 1]:
                    btn_type = "primary" if page_idx == current_page else "secondary"
                    if st.button(
                        f"{page_idx + 1}",
                        type=btn_type,
                        use_container_width=True,
                        key=f"chunk_page_btn_{page_idx}",
                    ):
                        st.session_state.chunk_viewer_page = page_idx
                        st.rerun()

            with cols[-1]:
                if st.button(
                    "다음",
                    disabled=(current_page == total_pages - 1),
                    use_container_width=True,
                    key="chunk_next_btn",
                ):
                    st.session_state.chunk_viewer_page = current_page + 1
                    st.rerun()

    if st.button("닫기", use_container_width=True, key="close_chunks_viewer_btn"):
        reset_chunks_viewer()
        st.rerun()


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
        # Removed st.rerun() here. Dialog will close when dialog_doc_to_show is set to None
        # and the main app reruns.
        pass


def reset_admin_active():
    st.session_state.admin_active = False


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

    # 상단 옵션 영역
    col_opt1, col_opt2 = st.columns([1, 1])
    with col_opt1:
        auto_sync = st.checkbox(
            "파일 업로드/삭제 후 자동 동기화 실행",
            value=True,
            help="체크 시 별도의 Sync 버튼 클릭 없이 즉시 DB에 반영합니다.",
        )
    with col_opt2:
        default_parser_idx = 0 if settings.PARSER_TYPE.lower() == "manual" else 1
        parser_type = st.radio(
            "적용 파서 선택",
            options=["manual", "docling"],
            index=default_parser_idx,
            horizontal=True,
            help="동기화 파이프라인에서 사용할 PDF 파서 전략을 지정합니다.",
        )

    st.divider()

    # 공통 동기화 로직 함수
    def trigger_sync(force=False):
        progress_bar = st.progress(0)

        with st.status("데이터베이스 동기화 중...", expanded=True) as status:

            def sync_callback(current, total, file_name, pb=progress_bar, st_status=status):
                if total > 0:
                    percent = int((current / total) * 100)
                    percent = min(100, max(0, percent))
                    pb.progress(percent, text=f"[{current}/{total}] {file_name} 처리 중... ({percent}%)")
                    st_status.update(label=f"진행 중: {file_name}", state="running")
                    st.write(f"[{current}/{total}] {file_name} 처리 완료 ({percent}%)")
                else:
                    pb.progress(0, text="대기 중...")

            # 싱글톤 인스턴스 사용 (불필요한 모델 로드 방지)
            orchestrator = PipelineOrchestrator()
            orchestrator.run_ingestion(force=force, parser_type=parser_type, progress_callback=sync_callback)
            progress_bar.progress(100, text="모든 파일 처리 완료 (100%)")
            status.update(label="동기화 완료", state="complete", expanded=False)
        # 캐시된 RAG 시스템(db_manager, rag_chain)을 재초기화하여 리셋된 컬렉션을 반영
        initialize_rag_system.clear()
        st.success("DB 동기화 완료")
        time.sleep(0.5)
        progress_bar.empty()
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
                        st.session_state.should_rerun_app = True  # Set flag for rerun even without auto-sync
            else:
                st.warning("선택된 파일이 없습니다.")

    with col2:
        st.subheader("수동 동기화")
        st.info("자동 동기화를 껐거나, 강제 업데이트가 필요한 경우 사용하세요.")
        if st.button("데이터 파이프라인 가동 (Sync)", key="dialog_sync_btn", use_container_width=True):
            trigger_sync(force=True)

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
        orchestrator = PipelineOrchestrator()
        manifest = orchestrator._load_manifest()
        manifest_files = manifest.get("files", {})

        h_col1, h_col2, h_col3, h_col4, h_col5, h_col6, h_col7, h_col8 = st.columns(
            [0.3, 2.0, 0.6, 0.9, 0.9, 0.6, 0.6, 0.6]
        )
        h_col1.write("**No**")
        h_col2.write("**파일명**")
        h_col3.write("**크기**")
        h_col4.write("**상태**")
        h_col5.write("**적용 파서**")
        h_col6.write("**청크**")
        h_col7.write("**동기화**")
        h_col8.write("**삭제**")
        st.markdown(
            "<hr style='margin: 0px 0px 10px 0px; border: 0.5px solid rgba(151,166,195,0.2);'>",
            unsafe_allow_html=True,
        )
        for i, f in enumerate(current_files):
            r_col1, r_col2, r_col3, r_col4, r_col5, r_col6, r_col7, r_col8 = st.columns(
                [0.3, 2.0, 0.6, 0.9, 0.9, 0.6, 0.6, 0.6]
            )
            r_col1.write(f"{i + 1}")

            # 적용 파서 식별 (매니페스트 우선 조회, 차선으로 ChromaDB 조회)
            import unicodedata

            rel_path = unicodedata.normalize("NFC", str(f.relative_to(RAW_DATA_DIR)))

            r_col2.text(rel_path)  # 파일명 대신 상대 경로 표시
            r_col3.write(format_size(f.stat().st_size))

            normalized_manifest_files = {unicodedata.normalize("NFC", k): v for k, v in manifest_files.items()}
            file_manifest_info = normalized_manifest_files.get(rel_path, {})
            parser_name = file_manifest_info.get("parser_type")

            chunks = db_manager.get_source_chunks(f.name)
            chunk_count = len(chunks)
            if not parser_name:
                parser_name = parser_type
                if chunks:
                    parser_name = chunks[0].get("metadata", {}).get("parser", parser_type)

            parser_options = ["manual", "docling"]

            # 청크 개수가 0개이면 미동기화, 0보다 크면 동기화 완료 배지 표시
            pending = st.session_state.get("parser_change_pending") or {}
            if chunk_count == 0:
                r_col4.markdown(
                    "<div style='color: #ff4b4b; font-size: 0.8rem; "
                    "font-weight: bold; white-space: nowrap; margin-top: 6px;'>미동기화 (대기)</div>",
                    unsafe_allow_html=True,
                )
                display_parser = pending.get(f.name, parser_name or parser_type)
            else:
                r_col4.markdown(
                    "<div style='color: #00cc66; font-size: 0.8rem; "
                    "font-weight: bold; white-space: nowrap; margin-top: 6px;'>동기화 완료</div>",
                    unsafe_allow_html=True,
                )
                display_parser = pending.get(f.name, parser_name)

            try:
                selected_idx = parser_options.index(display_parser.lower())
            except ValueError:
                selected_idx = 0

            selected_parser = r_col5.selectbox(
                "파서 선택",
                options=parser_options,
                index=selected_idx,
                key=f"parser_select_{i}_{display_parser}",
                label_visibility="collapsed",
                disabled=False,
            )

            if selected_parser != parser_name:
                if f.name not in pending or pending[f.name] != selected_parser:
                    if st.session_state.parser_change_pending is None:
                        st.session_state.parser_change_pending = {}
                    st.session_state.parser_change_pending[f.name] = selected_parser
                    st.rerun()
            elif f.name in pending:
                st.session_state.parser_change_pending.pop(f.name, None)
                if not st.session_state.parser_change_pending:
                    st.session_state.parser_change_pending = None
                st.rerun()

            if r_col6.button(f"{chunk_count} 🔍", key=f"view_chunks_{i}", help="청크 상세 내용 보기"):
                st.session_state.dialog_chunks_file_to_show = f.name
                st.session_state.admin_active = False
                st.session_state.should_rerun_app = True
                st.rerun()

            # 개별 동기화/재색인 실행
            if r_col7.button("🔄", key=f"sync_btn_{i}", help="개별 동기화 및 재색인 실행"):
                active_parser = selected_parser
                if pending and f.name in pending:
                    st.session_state.parser_change_pending.pop(f.name, None)
                    if not st.session_state.parser_change_pending:
                        st.session_state.parser_change_pending = None

                progress_bar = st.progress(0)
                with st.status(f"{f.name} 개별 동기화 중...", expanded=True) as status:

                    def single_file_sync_callback(current, total, name, pb=progress_bar, st_status=status):
                        if total > 0:
                            percent = int((current / total) * 100)
                            percent = min(100, max(0, percent))
                            pb.progress(percent, text=f"[{current}/{total}] {name} 처리 중... ({percent}%)")
                            st_status.update(label=f"진행 중: {name}", state="running")
                            st.write(f"[{current}/{total}] {name} 처리 완료 ({percent}%)")

                    orchestrator = PipelineOrchestrator()
                    orchestrator.update_file_parser(f.name, active_parser, progress_callback=single_file_sync_callback)
                    progress_bar.progress(100, text="동기화 완료")
                    status.update(label="개별 동기화 및 재색인 완료", state="complete", expanded=False)

                initialize_rag_system.clear()
                st.toast(f"동기화 완료: {f.name}")
                time.sleep(0.5)
                progress_bar.empty()
                st.rerun()

            if r_col8.button("🗑️", key=f"del_btn_{i}", help=f"'{f.name}' 삭제") and f.exists():
                f.unlink()
                st.toast(f"파일 삭제됨: {f.name}")
                if auto_sync:
                    trigger_sync(force=False)
                else:
                    time.sleep(0.5)
                    st.rerun()

    pending = st.session_state.get("parser_change_pending")
    if pending:
        st.warning(
            "파서 변경 확인: 다음 파일들의 파서를 변경하시겠습니까? "
            "변경 시 해당 파일들은 새로운 파서 규격으로 즉시 재색인됩니다."
        )

        change_details = []
        for file_name, new_parser in pending.items():
            file_chunks = db_manager.get_source_chunks(file_name)
            old_parser = "manual"
            if file_chunks:
                old_parser = file_chunks[0].get("metadata", {}).get("parser", "manual")
            change_details.append(f"- {file_name}: {old_parser} -> {new_parser}")

        st.markdown("\n".join(change_details))

        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("예, 변경 및 재색인 실행", key="confirm_parser_change_btn", use_container_width=True):
                pending_copy = pending.copy()
                st.session_state.parser_change_pending = None
                progress_bar = st.progress(0)

                with st.status("지정된 파일들의 파서 전환 및 재색인 중...", expanded=True) as status:

                    def multi_file_sync_callback(current, total, name, pb=progress_bar, st_status=status):
                        if total > 0:
                            percent = int((current / total) * 100)
                            percent = min(100, max(0, percent))
                            pb.progress(percent, text=f"[{current}/{total}] {name} 처리 중... ({percent}%)")
                            st_status.update(label=f"진행 중: {name}", state="running")

                    orchestrator = PipelineOrchestrator()
                    orchestrator.update_multiple_file_parsers(pending_copy, progress_callback=multi_file_sync_callback)
                    progress_bar.progress(100, text="파서 전환 완료")
                    status.update(label="파서 전환 및 재색인 완료", state="complete", expanded=False)

                initialize_rag_system.clear()
                st.toast("선택한 파일들의 파서가 성공적으로 전환되었습니다.")
                time.sleep(0.5)
                progress_bar.empty()
                st.rerun()

        with col_cancel:
            if st.button("취소", key="cancel_parser_change_btn", use_container_width=True):
                st.session_state.parser_change_pending = None
                st.rerun()

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


# --- 모델 다운로드 및 준비 상태 사전 체크 ---
model_ready = True
is_ollama = settings.MODEL_TYPE == "ollama"

if is_ollama:
    try:
        llm_instance = LLMFactory.create_llm()
        if not llm_instance.is_model_available():
            model_ready = False
    except Exception as e:
        model_ready = False
        logger.error(f"로컬 LLM 상태 진단 중 오류: {e}")


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
/* 채팅 메시지 내부 스피너 아바타 높이 맞춤 */
div[data-testid="stChatMessage"] div[data-testid="stSpinner"] > div {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
/* 채팅 메시지 내부 st.columns 컨테이너의 위쪽 여백을 조정해 왼쪽 아바타와 세로축 중앙 정렬 일치 */
div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"] {
    margin-top: -0.25rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# --- 5. 사이드바 (Sidebar) 구성 ---
with st.sidebar:
    st.title("설정 및 관리")

    # 1. 시스템 정보 (콤팩트 텍스트 배치)
    st.markdown(f"**LLM:** `{settings.MODEL_NAME}` ({settings.MODEL_TYPE.upper()})")
    st.markdown(f"**임베딩:** `{settings.EMBEDDING_MODEL_NAME.split('/')[-1]}`")
    st.divider()

    # 2. 검색 설정
    st.subheader("검색 설정")
    k_value = st.slider("초벌 검색 개수 (K)", 1, 20, 10, 1, disabled=st.session_state.is_generating)
    final_k_value = st.slider("최종 선별 개수", 1, 10, 5, 1, disabled=st.session_state.is_generating)
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

# --- 6. 다이얼로그 활성화 제어 ---
if st.session_state.get("admin_active", False):
    show_admin_dialog(db_manager)

if st.session_state.get("dialog_chunks_file_to_show"):
    show_chunks_viewer_dialog(st.session_state.dialog_chunks_file_to_show, db_manager)


# --- UI 스트리밍 핸들러 ---
def check_repetition(full_response: str) -> tuple[bool, str]:
    """동일 라인이 15자 이상이고 3회 이상 반복되면 True와 해당 라인을 반환"""
    lines = full_response.split("\n")
    line_counts: dict[str, int] = {}
    for line in lines:
        line_trimmed = line.strip()
        if len(line_trimmed) >= 15:
            line_counts[line_trimmed] = line_counts.get(line_trimmed, 0) + 1
            if line_counts[line_trimmed] >= 3:
                return True, line_trimmed
    return False, ""


def format_doc_status(docs_list):
    if not docs_list:
        return "0개 문서 (0개 청크)"
    unique_files = set()
    for d in docs_list:
        src_name = get_doc_field(d, MetadataFields.SRC_NAME, "알 수 없음")
        unique_files.add(src_name)
    return f"{len(unique_files)}개 문서 ({len(docs_list)}개 청크)"


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
        over_limit = total_latency > 5.0

        if st.session_state.show_expert_mode:
            header = f"**총 소요시간: {total_latency:.2f}초**"
            if over_limit:
                header = f"⚠️ {header} (5초 초과)"

            stage_durations = {s: v["end"] - v["start"] for s, v in self.stage_latencies.items() if "end" in v}
            bottleneck = max(stage_durations, key=lambda s: stage_durations[s]) if stage_durations else None

            lines = [header]
            for stage, elapsed in stage_durations.items():
                marker = " ← 병목" if over_limit and stage == bottleneck else ""
                lines.append(f"- {stage}: {elapsed:.2f}초{marker}")

            self.latency_placeholder.info("\n".join(lines))
        else:
            self.latency_placeholder.empty()

    def handle_error(self, error):
        if self.full_response:
            msg = "\n\n[안내] 통신 오류로 답변이 불완전할 수 있습니다."
            self.response_container.markdown(self.full_response + msg)
            return self.full_response + msg
        return f"오류가 발생했습니다: {error}"


# --- 메인 채팅 화면 ---
st.title("기업 매뉴얼 Q&A 서비스")
st.info("사내 규정 및 매뉴얼에 대해 질문하면 인용 출처와 함께 답변해 드립니다.")


# --- 6.1. Ollama 모델 다운로드 실시간 상태 시각화 ---
@st.fragment(run_every="1s")
def render_download_progress(llm):
    # 백그라운드 다운로드 시작
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

        if msg.get("citations"):
            for i, doc in enumerate(msg["citations"]):
                if not isinstance(doc, dict):
                    continue
                metadata = doc.get("metadata", {})
                source = metadata.get(MetadataFields.SRC_NAME, "알 수 없음")
                page = metadata.get(MetadataFields.PG_NUM, "-")
                score = metadata.get("rerank_score", doc.get("score", 0.0))
                is_low_confidence = score < 0.5

                button_label = f"📄 {source} (p.{page}) - 신뢰도: {score:.2f}"
                if is_low_confidence:
                    button_label += " ⚠️"

                if st.button(button_label, key=f"cite_{msg_idx}_{i}", disabled=st.session_state.is_generating):
                    st.session_state.dialog_doc_to_show = doc
                    st.session_state.should_rerun_app = True  # Set flag instead of direct rerun

                if is_low_confidence:
                    st.caption("⚠️ 신뢰도가 낮아 환각 발생 가능성이 있습니다. 원문을 직접 확인하세요.")

if st.session_state.dialog_doc_to_show:
    show_document_dialog(st.session_state.dialog_doc_to_show)
    st.session_state.dialog_doc_to_show = None


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


# active generation UI
if st.session_state.is_generating:
    show_expert_mode = st.session_state.show_expert_mode
    with st.chat_message("assistant"):
        ui_handler = StreamUIHandler()

        # Re-render accumulated steps
        for step in st.session_state.stream_steps:
            ui_handler.process_step(step)

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
            # Consume the stream iterator
            if st.session_state.stream_iter:
                try:
                    import contextlib

                    spinner_ctx = (
                        st.spinner("답변을 생성하고 있습니다...") if not show_expert_mode else contextlib.nullcontext()
                    )
                    with spinner_ctx:
                        for step in st.session_state.stream_iter:
                            if st.session_state.stop_generation:
                                break

                            st.session_state.stream_steps.append(step)

                            # 중복 생성 탐지 로직 (generation streaming 상태일 때 수행)
                            if step.get("stage") == "generation" and step.get("status") == "streaming":
                                new_text = step.get("output", "")
                                # 임시 full_response 계산
                                temp_response = ui_handler.full_response + new_text
                                has_repetition, repeated_line = check_repetition(temp_response)
                                if has_repetition:
                                    logger.warning(f"동일 라인 반복 감지로 답변 생성 중단: '{repeated_line}'")
                                    ui_handler.full_response += (
                                        "\n\n[안내] 동일한 문장/라인이 반복되어 답변 생성이 안전하게 중단되었습니다."
                                    )
                                    ui_handler.response_container.markdown(ui_handler.full_response)
                                    st.session_state.is_generating = False
                                    st.session_state.stop_generation = True
                                    break

                            ui_handler.process_step(step)

                    if not st.session_state.stop_generation:
                        st.session_state.is_generating = False

                        latency = time.time() - st.session_state.start_time
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": ui_handler.full_response,
                                "citations": ui_handler.final_docs,
                                "latency": latency,
                            }
                        )
                        perf_logger.log(
                            query=st.session_state.current_prompt,
                            answer=ui_handler.full_response,
                            total_latency=latency,
                            latencies_per_stage={
                                s: v["end"] - v["start"] for s, v in ui_handler.stage_latencies.items() if "end" in v
                            },
                            confidence_scores=[
                                doc.get("metadata", {}).get("rerank_score", doc.get("score", 0.0))
                                for doc in ui_handler.final_docs
                            ],
                            system_stats=get_system_stats(),
                            status="success",
                        )
                        st.session_state.stream_iter = None
                        st.session_state.current_prompt = ""
                        st.rerun()
                    else:
                        st.session_state.is_generating = False

                        latency = time.time() - st.session_state.start_time
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": ui_handler.full_response,
                                "citations": ui_handler.final_docs,
                                "latency": latency,
                            }
                        )
                        perf_logger.log(
                            query=st.session_state.current_prompt,
                            answer=ui_handler.full_response,
                            total_latency=latency,
                            latencies_per_stage={
                                s: v["end"] - v["start"] for s, v in ui_handler.stage_latencies.items() if "end" in v
                            },
                            confidence_scores=[
                                doc.get("metadata", {}).get("rerank_score", doc.get("score", 0.0))
                                for doc in ui_handler.final_docs
                            ],
                            system_stats=get_system_stats(),
                            status="interrupted",
                        )
                        st.session_state.stream_iter = None
                        st.session_state.current_prompt = ""
                        st.session_state.stop_generation = False
                        st.rerun()

                except Exception as e:
                    st.session_state.is_generating = False
                    logger.error(f"채팅 중 오류 발생: {e}", exc_info=True)

                    error_msg = ui_handler.handle_error(e) or f"\n\n[오류] 답변 생성 중 문제가 발생했습니다: {e}"

                    latency = time.time() - st.session_state.start_time
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_msg,
                            "citations": ui_handler.final_docs,
                            "latency": latency,
                        }
                    )
                    perf_logger.log(
                        query=st.session_state.current_prompt,
                        error=str(e),
                        total_latency=latency,
                        system_stats=get_system_stats(),
                        status="error",
                    )
                    st.session_state.stream_iter = None
                    st.session_state.current_prompt = ""
                    st.rerun()

# --- Controlled Rerun at the end of the script ---
if st.session_state.should_rerun_app:
    st.session_state.should_rerun_app = False
    st.rerun()
