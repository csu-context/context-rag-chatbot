from unittest.mock import MagicMock, patch

import pytest
import torch
from langchain_core.documents import Document

from src.common.config import settings
from src.core.reranker import CrossEncoderReranker


# ---------------------------------------------------------------------------
# 채팅 이력 최대 길이 제한 (메모리 누수 방지)
# ---------------------------------------------------------------------------
def test_chat_history_trim_enforces_max_turns():
    """메시지 수가 MAX_CHAT_HISTORY_TURNS * 2 초과 시 오래된 메시지가 잘리는지 검증."""
    from src.ui.stream_responder import _trim_chat_history

    max_msgs = settings.MAX_CHAT_HISTORY_TURNS * 2
    all_messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(max_msgs + 10)
    ]

    mock_state = MagicMock()
    mock_state.messages = all_messages.copy()

    with patch("src.ui.stream_responder.st.session_state", mock_state):
        _trim_chat_history()

    assert len(mock_state.messages) == max_msgs
    assert mock_state.messages[-1] == all_messages[-1]


def test_chat_history_trim_no_op_when_under_limit():
    """메시지 수가 한도 이하면 트리밍이 발생하지 않는지 검증."""
    from src.ui.stream_responder import _trim_chat_history

    max_msgs = settings.MAX_CHAT_HISTORY_TURNS * 2
    all_messages = [{"role": "user", "content": f"msg {i}"} for i in range(max_msgs - 2)]

    mock_state = MagicMock()
    mock_state.messages = all_messages.copy()

    with patch("src.ui.stream_responder.st.session_state", mock_state):
        _trim_chat_history()

    assert len(mock_state.messages) == len(all_messages)


# ---------------------------------------------------------------------------
# Reranker batch_size 제한으로 VRAM spike 방지
# ---------------------------------------------------------------------------
def test_reranker_batch_size_passed_to_predict():
    """model.predict 호출 시 settings.RERANKER_BATCH_SIZE가 batch_size로 전달되는지 검증."""
    CrossEncoderReranker.reset_instance()
    try:
        reranker = CrossEncoderReranker.get_instance(device="cpu")
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8, 0.3, 0.6]

        docs = [Document(page_content=f"문서 {i}", metadata={}) for i in range(3)]

        with patch.object(reranker, "_load_model", return_value=mock_model):
            reranker.rerank("테스트 쿼리", docs, top_k=3, threshold=0.0)

        _, call_kwargs = mock_model.predict.call_args
        assert call_kwargs.get("batch_size") == settings.RERANKER_BATCH_SIZE
    finally:
        CrossEncoderReranker.reset_instance()


# ---------------------------------------------------------------------------
# BGE 임베더 fp16 로드 검증
# ---------------------------------------------------------------------------
def test_embedder_fp16_used_on_cuda():
    """CUDA 환경에서 BGEEmbedder가 torch.float16으로 모델을 로드하는지 검증."""
    from src.models.embedder import BGEEmbedder

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.backends.mps.is_available", return_value=False),
        patch("src.models.embedder.SentenceTransformer") as mock_st,
    ):
        mock_st.return_value = MagicMock()
        BGEEmbedder(model_name="BAAI/bge-m3")

        _, call_kwargs = mock_st.call_args
        model_kwargs = call_kwargs.get("model_kwargs", {})
        assert model_kwargs.get("torch_dtype") == torch.float16


def test_embedder_no_fp16_on_cpu():
    """CPU 환경에서 BGEEmbedder가 fp16 없이 기본 로드하는지 검증."""
    from src.models.embedder import BGEEmbedder

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
        patch("src.models.embedder.SentenceTransformer") as mock_st,
    ):
        mock_st.return_value = MagicMock()
        BGEEmbedder(model_name="BAAI/bge-m3")

        _, call_kwargs = mock_st.call_args
        model_kwargs = call_kwargs.get("model_kwargs", {})
        assert model_kwargs == {}


# ---------------------------------------------------------------------------
# CrossEncoder Reranker fp16 automodel_args 검증
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "device, expect_fp16",
    [
        ("cuda", True),
        ("mps", True),
        ("cpu", False),
    ],
)
def test_reranker_fp16_automodel_args_by_device(device, expect_fp16):
    """device에 따라 CrossEncoderReranker._load_model이 올바른 automodel_args를 사용하는지 검증."""
    CrossEncoderReranker.reset_instance()
    CrossEncoderReranker._model = None
    try:
        reranker = CrossEncoderReranker(device=device)

        with patch("src.core.reranker.CrossEncoder") as mock_ce:
            mock_ce.return_value = MagicMock()
            reranker._load_model()

            _, call_kwargs = mock_ce.call_args
            automodel_args = call_kwargs.get("automodel_args", {})

            if expect_fp16:
                assert automodel_args.get("torch_dtype") == torch.float16
            else:
                assert automodel_args == {}
    finally:
        CrossEncoderReranker.reset_instance()
        CrossEncoderReranker._model = None
