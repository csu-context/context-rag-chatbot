"""백그라운드 파싱 중 Streamlit 메인 스레드 논블로킹 검증 (Issue #4 / DoD B).

DoD: "10개 PDF 파싱 중 Streamlit UI의 다른 탭·버튼이 메인 스레드 블로킹 없이 반응".

핵심 불변식: 무거운 인제스트(파싱)는 `trigger_sync_background` 가 띄운 별도 데몬 스레드
(`sync-background`)에서 수행되며, Streamlit 스크립트 실행(메인) 스레드는 즉시 반환되어
다른 UI 상호작용을 자유롭게 처리할 수 있다.

(참고: PDF 끼리는 메모리 보호를 위해 순차 처리된다 — `IngestionPipeline._process_pdf_sequential`.
 따라서 "동시"는 'PDF 병렬'이 아니라 'UI ↔ 백그라운드 파싱의 동시성'을 의미한다.)
"""

import threading
import time
from unittest.mock import MagicMock, patch

from src.controllers.sync_controller import SyncController

# 메인 스레드가 블로킹되지 않았다면 trigger 호출은 사실상 즉시 반환되어야 한다.
NON_BLOCKING_BUDGET_S = 0.5


def _run_with_blocking_ingestion(release: threading.Event, recorder: dict):
    """run_ingestion 이 release 이벤트까지 블로킹하도록 구성한 컨텍스트를 반환.

    파싱이 진행 중인 동안 메인 스레드의 동작을 관찰하기 위함.
    """

    def blocking_run_ingestion(*args, **kwargs):
        recorder["worker_thread"] = threading.current_thread()
        recorder["started"] = time.perf_counter()
        release.wait(timeout=10)  # 메인 스레드가 검증을 마칠 때까지 백그라운드에서 대기

    mock_orchestrator = MagicMock()
    mock_orchestrator.return_value.run_ingestion.side_effect = blocking_run_ingestion

    mock_st = MagicMock()
    mock_st.session_state = MagicMock()
    mock_st.session_state.get.return_value = None

    return patch.multiple(
        "src.controllers.sync_controller",
        st=mock_st,
        PipelineOrchestrator=mock_orchestrator,
    )


def test_trigger_returns_immediately_while_parsing_runs():
    """trigger_sync_background 는 파싱 완료를 기다리지 않고 즉시 반환한다 (논블로킹)."""
    release = threading.Event()
    recorder: dict = {}

    with (
        _run_with_blocking_ingestion(release, recorder),
        patch("src.utils.health_check.run_full_diagnostics", return_value=(True, {"db": {"anomalies": {}}})),
    ):
        start = time.perf_counter()
        job = SyncController.trigger_sync_background("docling", force=True)
        elapsed = time.perf_counter() - start

        try:
            # 호출은 즉시 반환 — 파싱(블로킹)이 끝나기를 기다리지 않음
            assert elapsed < NON_BLOCKING_BUDGET_S, f"메인 스레드 블로킹 의심: trigger 가 {elapsed:.2f}초 소요"
            # 반환 시점에 백그라운드 작업은 아직 진행 중이어야 함
            assert job.running is True
        finally:
            release.set()


def test_parsing_executes_off_main_thread():
    """파싱은 메인 스레드가 아닌 데몬 백그라운드 스레드(sync-background)에서 실행된다."""
    release = threading.Event()
    recorder: dict = {}
    main_thread = threading.current_thread()

    with (
        _run_with_blocking_ingestion(release, recorder),
        patch("src.utils.health_check.run_full_diagnostics", return_value=(True, {"db": {"anomalies": {}}})),
    ):
        SyncController.trigger_sync_background("docling", force=True)

        # 워커 스레드가 기동해 run_ingestion 에 진입할 때까지 대기
        deadline = time.perf_counter() + 2
        while "worker_thread" not in recorder and time.perf_counter() < deadline:
            time.sleep(0.01)
        release.set()

    worker = recorder.get("worker_thread")
    assert worker is not None, "백그라운드 워커가 run_ingestion 에 진입하지 못함"
    assert worker is not main_thread, "파싱이 메인 스레드에서 실행됨 (블로킹 위험)"
    assert worker.daemon is True
    assert worker.name == "sync-background"


def test_main_thread_stays_responsive_during_parse():
    """파싱이 백그라운드에서 도는 동안 메인 스레드는 즉각 다른 작업을 처리한다."""
    release = threading.Event()
    recorder: dict = {}

    with (
        _run_with_blocking_ingestion(release, recorder),
        patch("src.utils.health_check.run_full_diagnostics", return_value=(True, {"db": {"anomalies": {}}})),
    ):
        job = SyncController.trigger_sync_background("docling", force=True)
        try:
            # 워커가 블로킹 상태에 진입할 때까지 대기
            deadline = time.perf_counter() + 2
            while "started" not in recorder and time.perf_counter() < deadline:
                time.sleep(0.01)

            # 메인 스레드에서 UI 상호작용을 흉내내는 다수 작업 수행 — 모두 즉시 끝나야 함
            op_start = time.perf_counter()
            for _ in range(100):
                _ = job.snapshot()  # 진행 상태 폴링 (UI 렌더가 하는 일)
            ops_elapsed = time.perf_counter() - op_start

            # 백그라운드는 여전히 블로킹 중인데도 메인 작업은 즉시 완료 → 메인 미블로킹 증명
            assert job.running is True
            assert ops_elapsed < NON_BLOCKING_BUDGET_S, f"메인 스레드 응답 지연: 100회 폴링 {ops_elapsed:.2f}초"
        finally:
            release.set()
