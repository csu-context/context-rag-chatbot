import contextlib
import logging
import time
from typing import Any

import streamlit as st

from src.common.config import settings

logger = logging.getLogger(__name__)


def _trim_chat_history() -> None:
    max_messages = settings.MAX_CHAT_HISTORY_TURNS * 2
    if len(st.session_state.messages) > max_messages:
        st.session_state.messages = st.session_state.messages[-max_messages:]


class StreamResponder:
    """답변 스트리밍 생성 프로세스를 총괄하여 처리하는 UI 응답 클래스"""

    def __init__(self, perf_logger: Any, get_system_stats_fn: Any):
        self.perf_logger = perf_logger
        self.get_system_stats_fn = get_system_stats_fn
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

    @property
    def _stage_durations(self) -> dict[str, float]:
        """display_latencies/finalize_stream 중복 제거."""
        return {s: v["end"] - v["start"] for s, v in self.stage_latencies.items() if "end" in v}

    def display_latencies(self):
        total_latency = time.time() - self.start_time
        over_limit = total_latency > 5.0

        if st.session_state.show_expert_mode:
            header = f"**총 소요시간: {total_latency:.2f}초**"
            if over_limit:
                header = f"⚠️ {header} (5초 초과)"

            stage_durations = self._stage_durations
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

    def finalize_stream(self, status: str, error: Exception | None = None) -> None:
        """스트림 종료 후 메시지 저장, 로깅, 세션 정리를 수행합니다."""
        st.session_state.is_generating = False
        latency = time.time() - st.session_state.start_time
        content = self.handle_error(error) if error else self.full_response

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": content,
                "citations": self.final_docs,
                "latency": latency,
            }
        )
        _trim_chat_history()

        # Redis에 대화 기록 동기화 (REDIS_URL 설정 시)
        from src.common.config import settings

        if settings.REDIS_URL:
            from src.utils.redis_session import RedisSessionStore

            session_id = st.session_state.get("_redis_session_id", "")
            if session_id:
                # citations는 _format_docs에서 이미 plain dict로 변환됨 → 그대로 저장
                RedisSessionStore.save_messages(session_id, list(st.session_state.messages))

        log_kwargs: dict[str, Any] = {
            "total_latency": latency,
            "system_stats": self.get_system_stats_fn(),
            "status": status,
        }
        log_kwargs["query"] = st.session_state.current_prompt
        if error:
            log_kwargs["error"] = str(error)
        else:
            log_kwargs["answer"] = self.full_response
            log_kwargs["latencies_per_stage"] = self._stage_durations
            log_kwargs["confidence_scores"] = [
                doc.get("metadata", {}).get("rerank_score", doc.get("score", 0.0)) for doc in self.final_docs
            ]
        self.perf_logger.log(**log_kwargs)

        st.session_state.stream_iter = None
        st.session_state.current_prompt = ""

    def consume_stream(self):
        """스트리밍 생성 제네레이터를 읽어서 UI를 실시간 업데이트하고 결과를 로깅합니다."""
        show_expert_mode = st.session_state.show_expert_mode
        if not st.session_state.stream_iter:
            return

        try:
            spinner_ctx = (
                st.spinner("답변을 생성하고 있습니다...") if not show_expert_mode else contextlib.nullcontext()
            )
            with spinner_ctx:
                for step in st.session_state.stream_iter:
                    if st.session_state.stop_generation:
                        break

                    st.session_state.stream_steps.append(step)

                    self.process_step(step)

            if st.session_state.stop_generation:
                # 이터레이터 명시적 종료로 LLM HTTP 요청 취소
                _iter = st.session_state.stream_iter
                st.session_state.stop_generation = False
                self.finalize_stream("interrupted")
                with contextlib.suppress(Exception):
                    if _iter is not None:
                        _iter.close()
            else:
                self.finalize_stream("success")
            st.rerun()

        except Exception as e:
            logger.error(f"채팅 중 오류 발생: {e}", exc_info=True)
            self.finalize_stream("error", error=e)
            st.rerun()
