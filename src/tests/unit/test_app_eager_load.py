import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st


class MockSessionState(dict):
    def __getattr__(self, item):
        return self.get(item)

    def __setattr__(self, key, value):
        self[key] = value


@pytest.fixture
def mock_st_session_state():
    with patch("streamlit.session_state", MockSessionState()) as mock_state:
        yield mock_state


def test_reranker_eager_load(mock_st_session_state):
    # Test bug_027 fix

    with (
        patch("src.core.reranker.CrossEncoderReranker") as mock_reranker,
        patch("src.vector_db.chroma_manager.ChromaDBManager"),
        patch("src.core.retriever.RetrieverFactory"),
        patch("src.core.chains.get_rag_chain"),
        patch("src.app.st.set_page_config"),
        patch("src.app.st.sidebar"),
        patch("src.app.st.chat_message"),
        patch("src.app.st.chat_input"),
        patch("src.app.init_session_state"),
        patch("src.common.config.settings") as mock_settings,
    ):
        mock_settings.RERANKER_TYPE = "local"

        if "src.app" in sys.modules:
            importlib.reload(sys.modules["src.app"])
        else:
            pass

        # The background thread should have been started
        mock_reranker.eager_load_background.assert_called_once()
        assert st.session_state.get("reranker_eager_load_started") is True


def test_llm_warmup_guard(mock_st_session_state):
    # Test bug_044 fix

    with (
        patch("src.app.st.set_page_config"),
        patch("src.app.st.sidebar"),
        patch("src.app.st.chat_message"),
        patch("src.app.st.chat_input"),
        patch("src.vector_db.chroma_manager.ChromaDBManager"),
        patch("src.core.retriever.RetrieverFactory"),
        patch("src.core.chains.get_rag_chain"),
        patch("src.app.init_session_state"),
        patch("src.models.factory.LLMFactory") as mock_factory,
        patch("src.common.config.settings") as mock_settings,
    ):
        mock_settings.MODEL_TYPE = "ollama"
        mock_llm = MagicMock()
        mock_factory.create_llm.return_value = mock_llm
        mock_llm.is_model_available.return_value = True

        if "src.app" in sys.modules:
            importlib.reload(sys.modules["src.app"])
        else:
            pass

        # warmup should be called
        assert st.session_state.get("llm_warmup_done") is True
