import time
from unittest.mock import MagicMock, patch

import pytest

from src.ui.stream_responder import StreamResponder, _trim_chat_history


@pytest.fixture
def mock_st():
    with patch("src.ui.stream_responder.st") as mock_st:
        mock_st.session_state = MagicMock()
        mock_st.session_state.messages = []
        mock_st.session_state.show_expert_mode = True
        mock_st.session_state.is_generating = True
        mock_st.session_state.stream_iter = None
        mock_st.session_state.current_prompt = "Test query"
        mock_st.session_state.start_time = time.time()
        yield mock_st


class TestStreamResponder:
    def test_trim_chat_history(self, mock_st):
        with patch("src.ui.stream_responder.settings") as mock_settings:
            mock_settings.MAX_CHAT_HISTORY_TURNS = 2

            mock_st.session_state.messages = [1, 2, 3, 4, 5, 6]
            _trim_chat_history()
            # 2 turns * 2 = 4 max messages
            assert len(mock_st.session_state.messages) == 4
            assert mock_st.session_state.messages == [3, 4, 5, 6]

    def test_process_step_streaming(self, mock_st):
        responder = StreamResponder(MagicMock(), MagicMock())
        responder.process_step({"stage": "generation", "status": "streaming", "output": "Hello"})
        assert responder.full_response == "Hello"
        responder.response_container.markdown.assert_called_with("Hello▌")

    def test_process_step_complete(self, mock_st):
        responder = StreamResponder(MagicMock(), MagicMock())
        responder.full_response = "Hello World"
        responder.process_step({"stage": "generation", "status": "complete"})
        responder.response_container.markdown.assert_called_with("Hello World")

    def test_process_step_latencies(self, mock_st):
        responder = StreamResponder(MagicMock(), MagicMock())
        responder.process_step({"stage": "retrieval", "status": "running"})
        assert "retrieval" in responder.stage_latencies
        assert "start" in responder.stage_latencies["retrieval"]

        responder.process_step({"stage": "retrieval", "status": "complete"})
        assert "end" in responder.stage_latencies["retrieval"]

    def test_display_latencies_in_flight_generation(self, mock_st):
        """진행 중 단계(LLM 준비/생성)는 '생성 중'으로 표기되어 멈춤 오인을 막는다. (#200)"""
        responder = StreamResponder(MagicMock(), MagicMock())
        responder.process_step({"stage": "retrieval", "status": "running"})
        responder.process_step({"stage": "retrieval", "status": "complete"})
        # 생성 단계 시작(첫 토큰 전) — 끝나지 않은 in-flight 상태
        responder.process_step({"stage": "generation", "status": "running"})

        rendered = responder.latency_placeholder.info.call_args[0][0]
        assert "응답 생성 중" in rendered  # 헤더 진행 표시
        assert "generation: (생성 중...)" in rendered  # 진행 중 단계 라인
        assert "retrieval:" in rendered  # 완료 단계는 소요시간 유지

    def test_display_latencies_all_complete_no_busy_marker(self, mock_st):
        """모든 단계 완료 시 진행 표시 없이 소요시간만 노출한다. (회귀 가드)"""
        responder = StreamResponder(MagicMock(), MagicMock())
        responder.process_step({"stage": "retrieval", "status": "running"})
        responder.process_step({"stage": "retrieval", "status": "complete"})

        rendered = responder.latency_placeholder.info.call_args[0][0]
        assert "생성 중" not in rendered
        assert "처리 중" not in rendered
        assert "총 소요시간" in rendered

    def test_format_docs(self, mock_st):
        responder = StreamResponder(MagicMock(), MagicMock())

        class DummyDoc:
            def __init__(self, content, score):
                self.page_content = content
                self.metadata = {"score": score}

        docs = [DummyDoc("test content", 0.9)]
        responder.process_step({"stage": "citation", "status": "complete", "source_documents": docs})

        assert len(responder.final_docs) == 1
        assert responder.final_docs[0]["content"] == "test content"
        assert responder.final_docs[0]["score"] == 0.9

    def test_handle_error(self, mock_st):
        responder = StreamResponder(MagicMock(), MagicMock())
        responder.full_response = "Partial response"
        res = responder.handle_error("Network timeout")
        assert "Partial response" in res
        assert "통신 오류" in res
        responder.response_container.markdown.assert_called()

        responder.full_response = ""
        res2 = responder.handle_error("Crash")
        assert "Crash" in res2

    def test_finalize_stream_success(self, mock_st):
        mock_logger = MagicMock()
        mock_stats_fn = MagicMock(return_value={"cpu": 10})
        responder = StreamResponder(mock_logger, mock_stats_fn)
        responder.full_response = "Final Answer"
        responder.final_docs = [{"content": "doc1", "score": 0.8}]

        with patch("src.ui.stream_responder.settings") as mock_settings:
            mock_settings.REDIS_URL = None
            mock_settings.MAX_CHAT_HISTORY_TURNS = 2
            responder.finalize_stream("success")

        assert not mock_st.session_state.is_generating
        assert len(mock_st.session_state.messages) == 1
        assert mock_st.session_state.messages[0]["content"] == "Final Answer"
        assert mock_st.session_state.messages[0]["role"] == "assistant"

        mock_logger.log.assert_called_once()
        assert mock_logger.log.call_args[1]["status"] == "success"
        assert mock_logger.log.call_args[1]["answer"] == "Final Answer"

    def test_consume_stream_success(self, mock_st):
        mock_logger = MagicMock()
        responder = StreamResponder(mock_logger, MagicMock())

        # Mock iterator
        def mock_iter():
            yield {"stage": "generation", "status": "streaming", "output": "a"}
            yield {"stage": "generation", "status": "complete"}

        mock_st.session_state.stream_iter = mock_iter()
        mock_st.session_state.stream_steps = []
        mock_st.session_state.stop_generation = False

        with patch("src.ui.stream_responder.settings") as mock_settings:
            mock_settings.REDIS_URL = None
            mock_settings.MAX_CHAT_HISTORY_TURNS = 2
            responder.consume_stream()

        assert responder.full_response == "a"
        mock_st.rerun.assert_called_once()
