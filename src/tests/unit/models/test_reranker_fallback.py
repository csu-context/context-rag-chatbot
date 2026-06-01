import math
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.core.reranker import CrossEncoderReranker


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def test_reranker_cpu_fallback_on_gpu_error():
    CrossEncoderReranker.reset_instance()

    reranker = CrossEncoderReranker.get_instance(device="cuda")
    assert reranker.device == "cuda"

    mock_model = MagicMock()
    call_count = 0

    def mock_predict(pairs, batch_size=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("CUDA out of memory error in pytorch")
        return [0.8, 0.5]

    mock_model.predict = mock_predict

    with patch.object(reranker, "_load_model", return_value=mock_model):
        docs = [
            Document(page_content="첫 번째 문서", metadata={"chunk_id": "1"}),
            Document(page_content="두 번째 문서", metadata={"chunk_id": "2"}),
        ]

        result = reranker.rerank(query="질문", documents=docs, top_k=2, threshold=0.1)

        assert reranker.device == "cpu"
        assert call_count == 2
        assert len(result.documents) == 2
        assert result.scores[0] == pytest.approx(_sigmoid(0.8))
        # R2: 서킷브레이커 - GPU 장애 시각이 기록되어야 함
        assert CrossEncoderReranker._gpu_failure_time is not None


def test_circuit_breaker_gpu_recovery():
    """R2: GPU 복구 인터벌 경과 후 GPU 재시도 검증."""
    CrossEncoderReranker.reset_instance()

    reranker = CrossEncoderReranker.get_instance(device="cuda")
    reranker.device = "cpu"
    reranker._original_device = "cuda"
    # 복구 인터벌보다 오래 된 장애 시각 주입
    CrossEncoderReranker._gpu_failure_time = time.time() - 400

    mock_model = MagicMock()
    mock_model.predict.return_value = [0.9, 0.7]

    with patch.object(reranker, "_load_model", return_value=mock_model):
        docs = [
            Document(page_content="문서1", metadata={}),
            Document(page_content="문서2", metadata={}),
        ]
        reranker.rerank(query="질문", documents=docs, top_k=2, threshold=0.0)

        # 복구 시도 후 GPU 장치로 전환되었어야 함
        assert reranker.device == "cuda"
        # 장애 기록 초기화
        assert CrossEncoderReranker._gpu_failure_time is None


def test_rerank_with_timeout_returns_fallback_on_timeout():
    """R7: 타임아웃 시 원본 문서 즉시 반환 검증."""
    CrossEncoderReranker.reset_instance()

    reranker = CrossEncoderReranker.get_instance(device="cpu")

    # 공유 executor 워커를 테스트 종료 시 즉시 해제 → 다음 테스트로 누수 방지
    release = threading.Event()

    def slow_rerank(*args, **kwargs):
        release.wait(timeout=30)
        return MagicMock()

    docs = [Document(page_content=f"문서{i}", metadata={}) for i in range(3)]

    try:
        with (
            patch.object(reranker, "rerank", side_effect=slow_rerank),
            patch("src.core.reranker.settings") as mock_settings,
        ):
            mock_settings.RERANKER_TIMEOUT_SEC = 1
            mock_settings.RERANKER_THRESHOLD = 0.5
            result = reranker.rerank_with_timeout("질문", docs, top_k=3)
    finally:
        release.set()

    assert len(result.documents) <= 3
    assert all(s == 0.0 for s in result.scores)
