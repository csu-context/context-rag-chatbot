"""스트리밍 keepalive 단위 테스트.

토큰 사이 유휴가 keepalive_interval을 넘으면 빈 청크("")가 발사되어
Proxy 유휴 타임아웃에 의한 연결 단절을 방지하는지 검증한다.
"""

import time
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from src.core.chains import RAGPipeline


def _pipeline_with_stream(stream_fn):
    """__init__ 의존성(LLM/리트리버 등)을 우회하고 llm.stream만 주입한 파이프라인 생성."""
    pipeline = RAGPipeline.__new__(RAGPipeline)
    mock_llm = MagicMock()
    mock_llm.stream = stream_fn
    pipeline.llm = mock_llm
    return pipeline


def test_keepalive_emits_empty_chunk_on_idle():
    """토큰 사이 유휴가 interval을 넘으면 빈 청크가 1개 이상 발사된다."""

    def slow_stream(*args, **kwargs):
        yield AIMessage(content="A")
        time.sleep(0.5)  # interval(0.2초)보다 긴 유휴 → keepalive 발사 구간
        yield AIMessage(content="B")

    pipeline = _pipeline_with_stream(slow_stream)

    collected = list(pipeline._stream_with_keepalive(None, keepalive_interval=0.2))

    # 유휴 동안 빈 청크가 최소 1번 발사되어야 함
    assert "" in collected, f"keepalive 빈 청크 미발사: {collected}"
    # 실제 토큰은 순서대로 모두 전달되어야 함 (빈 청크는 본문에 영향 없음)
    assert "".join(c for c in collected if c) == "AB"


def test_no_keepalive_when_stream_is_fast():
    """토큰이 interval 내에 연속 도착하면 빈 청크는 발사되지 않는다."""

    def fast_stream(*args, **kwargs):
        yield AIMessage(content="A")
        yield AIMessage(content="B")

    pipeline = _pipeline_with_stream(fast_stream)

    collected = list(pipeline._stream_with_keepalive(None, keepalive_interval=5))

    assert collected == ["A", "B"]


def test_producer_exception_is_reraised_to_consumer():
    """생산자 스레드에서 발생한 예외가 소비자 측으로 재발생된다."""

    def err_stream(*args, **kwargs):
        yield AIMessage(content="A")
        raise RuntimeError("boom")

    pipeline = _pipeline_with_stream(err_stream)

    received = []
    with pytest.raises(RuntimeError, match="boom"):
        for chunk in pipeline._stream_with_keepalive(None, keepalive_interval=5):
            received.append(chunk)

    # 예외 전 정상 토큰은 전달되었어야 함
    assert "A" in received
