import logging
import time
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)


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


class StreamUIHandler:
    """답변 스트리밍 시 단계별 시간 측정 및 마크다운 UI 갱신을 담당하는 보조 클래스"""

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


class ChatController:
    """답변 스트리밍 생성 프로세스를 총괄하여 처리하는 컨트롤러 클래스"""

    @staticmethod
    def consume_stream(ui_handler: StreamUIHandler, perf_logger: Any, get_system_stats_fn: Any):
        """스트리밍 생성 제네레이터를 읽어서 UI를 실시간 업데이트하고 결과를 로깅합니다."""
        show_expert_mode = st.session_state.show_expert_mode
        if not st.session_state.stream_iter:
            return

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
                    system_stats=get_system_stats_fn(),
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
                    system_stats=get_system_stats_fn(),
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
                system_stats=get_system_stats_fn(),
                status="error",
            )
            st.session_state.stream_iter = None
            st.session_state.current_prompt = ""
            st.rerun()
