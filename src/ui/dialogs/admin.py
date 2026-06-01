import logging
import math
import time

import streamlit as st

from src.common.config import settings
from src.common.constants import SupportedFormats
from src.controllers.sync_controller import SyncController
from src.pipeline import PipelineOrchestrator
from src.utils.health_check import repair_integrity, run_full_diagnostics
from src.utils.paths import RAW_DATA_DIR
from src.utils.unicode import normalize_to_nfc

logger = logging.getLogger(__name__)


def reset_admin_active():
    st.session_state.admin_active = False


def _render_sync_progress(initialize_rag_system_callback):
    """R1: 백그라운드 동기화 진행률 폴링 프래그먼트."""

    @st.fragment(run_every="1s")
    def _progress_fragment():
        job = st.session_state.get("_current_sync_job")
        if not job:
            return

        snap = job.snapshot()

        if snap["running"]:
            if snap["total"] > 0:
                st.progress(
                    snap["percent"] / 100,
                    text=f"[{snap['current']}/{snap['total']}] {snap['file_name']} ({snap['percent']}%)",
                )
            else:
                st.progress(0, text="동기화 준비 중...")

            if st.button("작업 취소", key="cancel_sync_btn"):
                job.request_cancel()
            return

        # 완료 / 취소 / 오류
        st.session_state._current_sync_job = None
        if snap["completed"]:
            if initialize_rag_system_callback:
                initialize_rag_system_callback()
            st.success("동기화가 완료되었습니다.")
        elif snap["cancelled"]:
            st.warning("동기화가 취소되었습니다.")
        elif snap["error"]:
            st.error(f"동기화 오류: {snap['error']}")

        time.sleep(0.5)
        st.session_state.admin_active = False
        st.rerun()

    _progress_fragment()


@st.dialog("데이터 관리 시스템", width="large", on_dismiss=reset_admin_active)
def show_admin_dialog(db_manager, initialize_rag_system_callback):  # noqa: C901
    # R1: 백그라운드 동기화 실행 중이면 진행률 표시
    current_job = st.session_state.get("_current_sync_job")
    if current_job and current_job.running:
        st.subheader("동기화 진행 중...")
        st.info("파일을 처리하는 동안 다른 작업이 가능합니다.")
        _render_sync_progress(initialize_rag_system_callback)
        return

    st.markdown("지식 베이스(RAW_DATA) 관리 및 데이터베이스 동기화를 수행합니다.")

    # 상단 영역: 업로드 및 동기화
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("신규 문서 업로드")
        uploaded_files = st.file_uploader(
            "파일 선택 (PDF, MD, HWP, HWPX)",
            accept_multiple_files=True,
            type=[ext.lstrip(".") for ext in SupportedFormats.EXTENSIONS],
            key="dialog_uploader",
            label_visibility="collapsed",
        )
        st.write("")  # 위젯 간격 조절
        auto_sync = st.checkbox(
            "업로드 완료 후 자동 동기화(Sync) 실행",
            value=True,
            help="체크 시 업로드 직후 즉시 파이프라인을 가동하여 DB에 반영합니다.",
        )

    with col2:
        st.subheader("수동 동기화")
        st.info("자동 동기화를 껐거나, 강제 업데이트가 필요한 경우 사용하세요.")
        default_parser_idx = 0 if settings.PARSER_TYPE.lower() == "manual" else 1
        parser_type = st.radio(
            "파이프라인 적용 파서 선택",
            options=["manual", "docling"],
            index=default_parser_idx,
            format_func=lambda x: "기본(빠름)" if x == "manual" else "표 인식 강화(느림)",
            horizontal=True,
            help="동기화 시 사용할 PDF 파서 전략을 지정합니다.",
        )

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("업로드 실행", key="admin_upload_btn", use_container_width=True):
            if uploaded_files:
                # Issue 13: AxiosError 400 방지 — 빈 파일·중복·경로 문제 사전 검증
                valid_files = [f for f in uploaded_files if f.name and f.size and f.size > 0]
                invalid = [f.name for f in uploaded_files if not (f.name and f.size and f.size > 0)]
                if invalid:
                    st.warning(f"유효하지 않은 파일 제외됨: {invalid}")
                if not valid_files:
                    st.error("업로드 가능한 파일이 없습니다.")
                else:
                    with st.spinner("저장 중..."):
                        saved = []
                        for uploaded_file in valid_files:
                            try:
                                from pathlib import PurePosixPath

                                raw_name = uploaded_file.name or ""
                                # 경로 구성요소 제거 (path traversal 방지)
                                basename = PurePosixPath(raw_name).name
                                if not basename or basename in (".", ".."):
                                    st.error(f"잘못된 파일명: {raw_name}")
                                    continue
                                safe_name = normalize_to_nfc(basename)
                                file_path = RAW_DATA_DIR / safe_name
                                # 방어적 경로 검증
                                if not file_path.resolve().is_relative_to(RAW_DATA_DIR.resolve()):
                                    st.error(f"잘못된 파일 경로: {raw_name}")
                                    continue
                                with open(file_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                saved.append(safe_name)
                            except Exception as e:
                                st.error(f"저장 실패: {uploaded_file.name} — {e}")
                    if saved:
                        st.success(f"{len(saved)}개 파일 업로드 완료")
                        if auto_sync:
                            SyncController.trigger_sync_background(
                                parser_type=parser_type,
                                force=False,
                                clear_cache_callback=initialize_rag_system_callback,
                            )
                            st.rerun()
                        else:
                            st.session_state.should_rerun_app = True
            else:
                st.warning("선택된 파일이 없습니다.")

    with col_btn2:
        if st.button("데이터 파이프라인 가동 (Sync)", key="dialog_sync_btn", use_container_width=True):
            SyncController.trigger_sync_background(
                parser_type=parser_type,
                force=False,
                clear_cache_callback=initialize_rag_system_callback,
            )
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
    for ext in SupportedFormats.EXTENSIONS:
        current_files.extend(list(RAW_DATA_DIR.glob(f"**/*{ext}")))

    if not current_files:
        st.info("현재 등록된 문서가 없습니다.")
    else:
        orchestrator = PipelineOrchestrator()
        manifest = orchestrator.manifest_manager.load_manifest()
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

            rel_path = normalize_to_nfc(str(f.relative_to(RAW_DATA_DIR)))

            r_col2.text(rel_path)  # 파일명 대신 상대 경로 표시
            r_col3.write(format_size(f.stat().st_size))

            normalized_manifest_files = {normalize_to_nfc(k): v for k, v in manifest_files.items()}
            file_manifest_info = normalized_manifest_files.get(rel_path, {})
            parser_name = file_manifest_info.get("parser_type")

            chunks = db_manager.get_source_chunks(f.name)
            chunk_count = len(chunks)
            if not parser_name:
                parser_name = parser_type
                if chunks:
                    parser_name = chunks[0].get("metadata", {}).get("parser_type", parser_type)

            is_pdf = f.suffix.lower() == ".pdf"
            ext = f.suffix.lower().lstrip(".")

            # 동기화 상태 배지 및 display_parser 계산 (파일 형식 무관)
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

            # PDF만 파서 선택 가능, 나머지는 고정 파서 텍스트 표시
            if is_pdf:
                parser_options = ["manual", "docling"]

                try:
                    selected_idx = parser_options.index((display_parser or "manual").lower())
                except ValueError:
                    selected_idx = 0

                selected_parser = r_col5.selectbox(
                    "파서 선택",
                    options=parser_options,
                    index=selected_idx,
                    format_func=lambda x: "기본" if x == "manual" else "표 인식 강화",
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
            else:
                # MD, HWP, HWPX 등 고정 파서 사용 파일
                fixed_parser_label = {
                    "md": "markdown",
                    "markdown": "markdown",
                    "hwp": "hwp",
                    "hwpx": "hwp",
                }.get(ext, ext)
                selected_parser = fixed_parser_label
                display_label = (
                    "마크다운"
                    if fixed_parser_label == "markdown"
                    else "한글 (HWP)"
                    if fixed_parser_label == "hwp"
                    else fixed_parser_label
                )
                r_col5.selectbox(
                    "파서 선택",
                    options=[fixed_parser_label],
                    index=0,
                    format_func=lambda x, dl=display_label: dl,
                    key=f"parser_select_disabled_{i}",
                    label_visibility="collapsed",
                    disabled=True,
                )

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

                SyncController.trigger_single_file_sync(
                    file_name=f.name,
                    active_parser=active_parser,
                    clear_cache_callback=initialize_rag_system_callback,
                )

            if r_col8.button("🗑️", key=f"del_btn_{i}", help=f"'{f.name}' 삭제") and f.exists():
                f.unlink()
                st.toast(f"파일 삭제됨: {f.name}")
                if auto_sync:
                    SyncController.trigger_sync_background(
                        parser_type=parser_type,
                        force=False,
                        clear_cache_callback=initialize_rag_system_callback,
                    )
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
                old_parser = file_chunks[0].get("metadata", {}).get("parser_type", "manual")
            old_display = (
                "기본" if old_parser == "manual" else "표 인식 강화" if old_parser == "docling" else old_parser
            )
            new_display = (
                "기본" if new_parser == "manual" else "표 인식 강화" if new_parser == "docling" else new_parser
            )
            change_details.append(f"- {file_name}: {old_display} -> {new_display}")

        st.markdown("\n".join(change_details))

        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("예, 변경 및 재색인 실행", key="confirm_parser_change_btn", use_container_width=True):
                pending_copy = pending.copy()
                st.session_state.parser_change_pending = None

                SyncController.trigger_multiple_files_sync(
                    pending_copy=pending_copy,
                    clear_cache_callback=initialize_rag_system_callback,
                )

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
            st.success("모든 데이터가 정합성을 유지하고 있습니다.")
        else:
            if ghosts:
                st.error(f"유령 청크 감지: {len(ghosts)}개 파일의 데이터가 DB에 남아있습니다.")
            if mismatches:
                st.warning(f"업데이트 필요: {len(mismatches)}개 파일의 내용이 DB와 다릅니다.")
            if duplicates:
                st.error(f"중복 적재 감지: {len(duplicates)}개 파일에 여러 파서 데이터가 공존합니다.")

            if st.button("정합성 자동 복구 (Repair)", type="primary", use_container_width=True):
                with st.spinner("복구 작업 진행 중..."):
                    try:
                        repair_integrity(anomalies)
                        del st.session_state.health_report
                        initialize_rag_system_callback()
                    except Exception as repair_err:
                        st.error(f"복구 중 오류가 발생했습니다: {repair_err}")
                    else:
                        st.success("복구가 완료되었습니다. 상태를 재확인하세요.")
                        time.sleep(0.5)
                        st.rerun()

    st.divider()

    # Issue 28: ChromaDB 자동 백업 및 복원
    st.subheader("ChromaDB 백업 및 시스템 관리")
    from src.utils.backup import create_backup, list_backups, restore_backup

    col_bk1, col_bk2 = st.columns(2)

    with col_bk1:
        st.markdown("**백업 생성**")
        if st.button("현재 DB 상태 백업하기", use_container_width=True):
            with st.spinner("백업 중..."):
                path = create_backup()
            if path:
                st.success(f"백업 완료: {path.name}")
            else:
                st.error("백업 실패. 로그를 확인하세요.")

    with col_bk2:
        st.markdown("**백업 복구**")
        backups = list_backups()
        if backups:
            selected_backup = st.selectbox(
                "복구할 백업 파일", options=backups, format_func=lambda x: x.name, label_visibility="collapsed"
            )
            if st.button("선택 파일로 복구", use_container_width=True, type="primary"):
                st.warning("주의: 원본 파일(PDF) 목록과 DB 상태가 불일치할 수 있습니다.")
                with st.spinner(f"{selected_backup.name} 복구 중..."):
                    ok = restore_backup(selected_backup)
                if ok:
                    st.success("복구 성공! (앱 재시작 권장)")
                else:
                    st.error("복구 실패. 서버 로그 확인 요망.")
        else:
            st.info("저장된 백업 파일이 없습니다.")

    st.divider()
    if st.button("관리 시스템 종료 (닫기)", use_container_width=True):
        st.session_state.admin_active = False
        st.rerun()
