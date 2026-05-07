"""
프로젝트 로깅 가이드 및 유틸리티

이 모듈은 프로젝트 전체의 일관된 로깅 스타일을 유지하기 위해 작성되었습니다.
모든 팀원은 아래의 로깅 원칙을 준수해야 합니다.

[로깅 원칙]
1. 개별 모듈에서 logging.basicConfig() 호출 금지:
   - 각 파일에서 설정을 하드코딩하면 로그 포맷이 파편화됩니다.
2. 로거 인스턴스 생성:
   - 각 모듈 상단에서 `logger = logging.getLogger(__name__)`을 사용하여 로거를 생성합니다.
3. 전역 설정 적용:
   - 애플리케이션의 진입점(main.py, app.py)에서만 `setup_global_logging()`을 호출합니다.
4. 성능 기록:
   - 실행 시간 등 성능 지표는 `PerformanceLogger().log(...)`를 통해 별도 관리합니다.
5. 트레이싱 기록:
   - RAG 파이프라인 전 과정은 `TracingLogger().log_trace(...)`를 통해 구조화된 JSON으로 기록합니다.
"""

import json
import logging
import os
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.paths import LOGS_DIR


class PerformanceLogger:
    """성능 데이터를 중복 없이 확실히 기록하기 위한 전용 클래스 (싱글톤)"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._setup()
            return cls._instance

    def _setup(self):
        # 디렉토리가 없으면 생성
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.log_file = LOGS_DIR / "performance.log"
        # 파일이 없으면 헤더 생성
        if not self.log_file.exists():
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("Timestamp,Type,Duration,Info\n")

    def log(self, log_type: str, duration: float, info: str = ""):
        """한 줄의 성능 로그를 파일에 직접 기록합니다."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 콤마나 개행 문자 제거하여 CSV 형식 유지
        info = info.replace(",", " ").replace("\n", " ").strip()
        log_entry = f"{timestamp},{log_type},{duration:.2f},{info}\n"

        # Thread-safe하게 파일 쓰기
        with self._lock, open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)


class TraceSession:
    """트레이싱 세션을 관리하는 객체"""

    def __init__(self, logger: "TracingLogger", **initial_data):
        self.logger = logger
        self.data = {
            "steps": [],
            "total_latency_ms": 0,
            **initial_data,
        }
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.data["total_latency_ms"] = round((time.time() - self.start_time) * 1000, 2)
        if exc_type:
            self.data["status"] = "error"
            self.data["error_message"] = str(exc_val)
        else:
            self.data["status"] = self.data.get("status", "success")

        self.logger.log_trace(self.data)

    @contextmanager
    def trace_step(self, step_name: str) -> Generator[dict[str, Any]]:
        """개별 단계의 지연 시간을 측정하고 데이터를 수집합니다."""
        step_data = {"step": step_name}
        start = time.time()
        try:
            yield step_data
        finally:
            latency = (time.time() - start) * 1000
            step_data["latency_ms"] = round(latency, 2)
            self.data["steps"].append(step_data)


class TracingLogger:
    """RAG 파이프라인의 전 과정을 구조화된 JSON으로 기록하는 로거 (싱글톤)"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._setup()
            return cls._instance

    @classmethod
    def reset_instance(cls):
        """테스트용 싱글톤 리셋"""
        with cls._lock:
            cls._instance = None

    def _setup(self):
        self.trace_dir = LOGS_DIR / "trace"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.is_debug = os.getenv("DEBUG", "false").lower() == "true"

    def _get_log_file(self) -> Path:
        """오늘 날짜의 로그 파일 경로 반환"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.trace_dir / f"trace_{date_str}.jsonl"

    def start_session(self, **initial_data) -> TraceSession:
        """새로운 트레이싱 세션을 시작합니다."""
        return TraceSession(self, **initial_data)

    def log_trace(self, trace_data: dict[str, Any]):
        """구조화된 추적 데이터를 JSONL 형식으로 저장합니다."""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            **trace_data,
        }

        # 민감 정보 필터링
        filtered_entry = self._filter_sensitive_data(log_entry)

        # 디버그 모드인 경우 Pretty Print
        if self.is_debug:
            print("\n" + "=" * 20 + " [DEBUG TRACE] " + "=" * 20)
            print(json.dumps(filtered_entry, indent=2, ensure_ascii=False))
            print("=" * 55 + "\n")

        # 파일 쓰기
        log_file = self._get_log_file()
        with self._lock, open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(filtered_entry, ensure_ascii=False) + "\n")

    def _filter_sensitive_data(self, data: Any) -> Any:
        """API 키 등 민감 정보가 포함된 필드를 필터링합니다."""
        if isinstance(data, dict):
            # 키 이름에 'key', 'token', 'secret' 등이 포함되면 값을 마스킹
            filtered = {}
            for k, v in data.items():
                if any(secret in k.lower() for secret in ["key", "token", "secret", "auth"]):
                    filtered[k] = "********"
                else:
                    filtered[k] = self._filter_sensitive_data(v)
            return filtered
        if isinstance(data, list):
            return [self._filter_sensitive_data(i) for i in data]
        return data


def setup_global_logging():
    """시스템 기본 로깅 설정 (app.log 용)"""
    # 디렉토리가 없으면 생성
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "app.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
        force=True,
    )
    # 노이즈 제거
    for name in ["httpx", "google", "langchain"]:
        logging.getLogger(name).setLevel(logging.WARNING)
