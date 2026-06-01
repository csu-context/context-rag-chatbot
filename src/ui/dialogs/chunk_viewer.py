import html
import json
import logging
import math
from pathlib import Path

import streamlit as st

from src.pipeline import PipelineOrchestrator
from src.utils.unicode import normalize_to_nfc

logger = logging.getLogger(__name__)


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
    orchestrator = PipelineOrchestrator()
    manifest = orchestrator.manifest_manager.load_manifest()
    manifest_files = manifest.get("files", {})

    target_hash = None
    normalized_file_name = normalize_to_nfc(file_name)
    for rel_path, info in manifest_files.items():
        normalized_rel_name = normalize_to_nfc(Path(rel_path).name)
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

                escaped_content = html.escape(chunk["content"])
                st.markdown(
                    f'<div class="chunk-child-container">{escaped_content}</div>',
                    unsafe_allow_html=True,
                )

                # 매핑된 상위 부모 텍스트가 있을 경우 익스팬더로 표시
                parent_text = parent_text_map.get(c_id)
                if parent_text:
                    with st.expander("상위 부모 문맥 (Parent Context)"):
                        escaped_parent = html.escape(parent_text)
                        st.markdown(
                            f'<div class="chunk-parent-container">{escaped_parent}</div>',
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
