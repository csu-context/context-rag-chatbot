import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import streamlit as st

from src.pipeline import PipelineOrchestrator

logger = logging.getLogger(__name__)


class SyncCancelledError(BaseException):
    """백그라운드 동기화 사용자 취소 신호.

    BaseException 을 상속하여 파이프라인 전반의 `except Exception` 블록에 흡수되지 않고
    sync_controller 의 취소 핸들러까지 전파되도록 한다.
    """


@dataclass
class SyncJobState:
    """스레드 안전 동기화 작업 상태. 백그라운드 스레드와 Streamlit 메인 스레드 간 공유."""

    running: bool = True
    current: int = 0
    total: int = 0
    file_name: str = ""
    percent: int = 0
    completed: bool = False
    cancelled: bool = False
    error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, compare=False, repr=False)
    _cancel: threading.Event = field(default_factory=threading.Event, compare=False, repr=False)

    def request_cancel(self) -> None:
        self._cancel.set()

    @property
    def should_cancel(self) -> bool:
        return self._cancel.is_set()

    def update_progress(self, current: int, total: int, file_name: str) -> None:
        with self._lock:
            self.current = current
            self.total = total
            self.file_name = file_name
            self.percent = int(current / total * 100) if total > 0 else 0

    def complete(self) -> None:
        with self._lock:
            self.running = False
            self.completed = True
            self.percent = 100

    def fail(self, error: str) -> None:
        with self._lock:
            self.running = False
            self.error = error

    def mark_cancelled(self) -> None:
        with self._lock:
            self.running = False
            self.cancelled = True

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "running": self.running,
                "current": self.current,
                "total": self.total,
                "file_name": self.file_name,
                "percent": self.percent,
                "completed": self.completed,
                "cancelled": self.cancelled,
                "error": self.error,
            }


class SyncController:
    """데이터 동기화 및 파이프라인 연동 비즈니스 논리를 총괄 제어하는 컨트롤러 클래스"""

    @staticmethod
    def trigger_sync_background(
        parser_type: str,
        force: bool = False,
        clear_cache_callback: Callable | None = None,
    ) -> SyncJobState:
        """비동기 백그라운드 동기화. 이미 실행 중이면 기존 job 반환."""
        existing = st.session_state.get("_current_sync_job")
        if existing and existing.running:
            return existing

        job = SyncJobState()
        st.session_state._current_sync_job = job

        def run(j: SyncJobState) -> None:
            try:
                from src.utils.health_check import repair_integrity, run_full_diagnostics

                _, report = run_full_diagnostics(silent=True, check_model=False)
                anomalies = report.get("db", {}).get("anomalies", {})
                if anomalies and any(anomalies.values()):
                    repair_integrity(anomalies, target_parser=parser_type)

                def progress_callback(current: int, total: int, file_name: str) -> None:
                    if j.should_cancel:
                        raise SyncCancelledError("사용자 취소")
                    j.update_progress(current, total, file_name)

                orchestrator = PipelineOrchestrator()
                orchestrator.run_ingestion(force=force, parser_type=parser_type, progress_callback=progress_callback)
                j.complete()
                if clear_cache_callback:
                    try:
                        clear_cache_callback()
                    except Exception as cache_err:
                        logger.error(f"캐시 초기화 콜백 실패: {cache_err}")
            except SyncCancelledError:
                j.mark_cancelled()
            except Exception as e:
                j.fail(str(e))

        thread = threading.Thread(target=run, args=(job,), daemon=True, name="sync-background")
        thread.start()
        return job

    @staticmethod
    def trigger_sync(parser_type: str, force: bool = False, clear_cache_callback=None):
        """데이터베이스 전체 동기화를 실행하고 진행 피드백을 UI에 업데이트합니다."""
        progress_bar = st.progress(0)

        with st.status("데이터 정합성 점검 및 복구 중...", expanded=True) as status:
            from src.utils.health_check import repair_integrity, run_full_diagnostics

            # 1. 정합성 점검 및 자가 치유
            st.write("DB 상태를 스캔하고 있습니다...")
            _, report = run_full_diagnostics(silent=True, check_model=False)
            anomalies = report.get("db", {}).get("anomalies", {})
            if anomalies and any(anomalies.values()):
                st.write("이상 데이터 감지: 자동 복구를 진행합니다.")
                repair_integrity(anomalies, target_parser=parser_type)
                st.write("정합성 복구 완료.")
            else:
                st.write("데이터 정합성 이상 없음.")

            # 2. 파이프라인 동기화 진행
            status.update(label="새 데이터 동기화 중...", state="running")

            def sync_callback(current, total, file_name, pb=progress_bar, st_status=status):
                if total > 0:
                    percent = int((current / total) * 100)
                    percent = min(100, max(0, percent))
                    pb.progress(percent, text=f"[{current}/{total}] {file_name} 처리 중... ({percent}%)")
                    st_status.update(label=f"진행 중: {file_name}", state="running")
                    st.write(f"[{current}/{total}] {file_name} 처리 완료 ({percent}%)")
                else:
                    pb.progress(0, text="대기 중...")

            # 싱글톤 오케스트레이터 가동
            orchestrator = PipelineOrchestrator()
            orchestrator.run_ingestion(force=force, parser_type=parser_type, progress_callback=sync_callback)
            progress_bar.progress(100, text="모든 파일 처리 완료 (100%)")
            status.update(label="동기화 완료", state="complete", expanded=False)

        # 캐싱된 리소스를 초기화하기 위한 콜백 수행
        if clear_cache_callback:
            clear_cache_callback()

        st.success("DB 동기화 완료")
        time.sleep(0.5)
        progress_bar.empty()
        st.rerun()

    @staticmethod
    def trigger_single_file_sync(file_name: str, active_parser: str, clear_cache_callback=None):
        """특정 개별 문서에 대한 동기화 및 재색인을 실행합니다."""

        progress_bar = st.progress(0)
        with st.status(f"{file_name} 개별 동기화 중...", expanded=True) as status:

            def single_file_sync_callback(current, total, name, pb=progress_bar, st_status=status):
                if total > 0:
                    percent = int((current / total) * 100)
                    percent = min(100, max(0, percent))
                    pb.progress(percent, text=f"[{current}/{total}] {name} 처리 중... ({percent}%)")
                    st_status.update(label=f"진행 중: {name}", state="running")
                    st.write(f"[{current}/{total}] {name} 처리 완료 ({percent}%)")

            orchestrator = PipelineOrchestrator()
            orchestrator.update_file_parser(file_name, active_parser, progress_callback=single_file_sync_callback)
            progress_bar.progress(100, text="동기화 완료")
            status.update(label="개별 동기화 및 재색인 완료", state="complete", expanded=False)

        if clear_cache_callback:
            clear_cache_callback()

        st.toast(f"동기화 완료: {file_name}")
        time.sleep(0.5)
        progress_bar.empty()
        st.rerun()

    @staticmethod
    def trigger_multiple_files_sync(pending_copy: dict[str, str], clear_cache_callback=None):
        """복수 문서에 대해 한꺼번에 파서 지정 변경 및 재색인을 가동합니다."""
        progress_bar = st.progress(0)
        with st.status("지정된 파일들의 파서 전환 및 재색인 중...", expanded=True) as status:

            def multi_file_sync_callback(current, total, name, pb=progress_bar, st_status=status):
                if total > 0:
                    percent = int((current / total) * 100)
                    percent = min(100, max(0, percent))
                    pb.progress(percent, text=f"[{current}/{total}] {name} 처리 중... ({percent}%)")
                    st_status.update(label=f"진행 중: {name}", state="running")

            orchestrator = PipelineOrchestrator()
            orchestrator.update_multiple_file_parsers(pending_copy, progress_callback=multi_file_sync_callback)
            progress_bar.progress(100, text="파서 전환 완료")
            status.update(label="파서 전환 및 재색인 완료", state="complete", expanded=False)

        if clear_cache_callback:
            clear_cache_callback()

        st.toast("선택한 파일들의 파서가 성공적으로 전환되었습니다.")
        time.sleep(0.5)
        progress_bar.empty()
        st.rerun()
