from typing import Any

import streamlit as st


def init_session_state():
    """Streamlit 애플리케이션의 세션 상태 변수들을 중앙 집중식으로 초기화합니다."""
    defaults = {
        "admin_active": False,
        "dialog_doc_to_show": None,
        "dialog_chunks_file_to_show": None,
        "parser_change_pending": None,
        "is_generating": False,
        "should_rerun_app": False,
        "stop_generation": False,
        "current_prompt": "",
        "stream_iter": None,
        "stream_steps": [],
        "full_response": "",
        "docs": [],
        "final_docs": [],
        "start_time": None,
        "show_expert_mode": False,
        "chunk_viewer_page": 0,
        # 백그라운드 동기화 작업 상태
        "_current_sync_job": None,
        "reranker_eager_load_started": False,
    }

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_session_val(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)


def set_session_val(key: str, val: Any) -> None:
    st.session_state[key] = val


def pop_session_val(key: str, default: Any = None) -> Any:
    return st.session_state.pop(key, default)


def clear_session_val(key: str) -> None:
    if key in st.session_state:
        del st.session_state[key]
