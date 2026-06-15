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
import re
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.paths import LOGS_DIR

# 민감 정보 필터링을 위한 정규표현식 정의
EMAIL_REGEX = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
PHONE_REGEX = re.compile(r"\b(?:\+82[-.\s]?)?\(?0[1-9]\d{0,2}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}\b")
RRN_REGEX = re.compile(r"\b\d{6}[-.\s]?[1-48]\d{6}\b")
API_KEY_REGEX = re.compile(
    r"\b(?:AIzaSy[a-zA-Z0-9_\-]{33}|"
    r"sk-ant-sid\d+-[a-zA-Z0-9_\-]{40,}|"
    r"sk-ant-[a-zA-Z0-9_\-]{40,}|"
    r"sk-[a-zA-Z0-9_\-]{32,})\b"
)


_PERF_LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_PERF_LOG_BACKUP_COUNT = 5


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
        from logging.handlers import RotatingFileHandler

        self.log_file = LOGS_DIR / "performance.jsonl"
        self.perf_logger = logging.getLogger("performance_logger")
        self.perf_logger.setLevel(logging.INFO)
        self.perf_logger.propagate = False

        if not self.perf_logger.handlers:
            try:
                LOGS_DIR.mkdir(parents=True, exist_ok=True)
                handler = RotatingFileHandler(
                    self.log_file, maxBytes=_PERF_LOG_MAX_BYTES, backupCount=_PERF_LOG_BACKUP_COUNT, encoding="utf-8"
                )
                handler.setFormatter(logging.Formatter("%(message)s"))
                self.perf_logger.addHandler(handler)
            except OSError as e:
                # 성능 로그 파일을 쓸 수 없어도(권한 등) 앱 기동/동작을 막지 않는다(no-op 핸들러로 폴백).
                self.log_file = None
                self.perf_logger.addHandler(logging.NullHandler())
                logging.getLogger(__name__).warning(f"성능 로깅 비활성화 — 파일 기록 불가(원인: {e}).")

    def log(self, **kwargs):
        """성능 로그를 JSONL 형식으로 파일에 기록합니다."""
        timestamp = datetime.now().isoformat()
        log_entry = {"timestamp": timestamp, **kwargs}
        self.perf_logger.info(json.dumps(log_entry, ensure_ascii=False))

    def log_inference(self, prompt: str, response: str, status: str, info: str = "", duration: float = 0.0):
        """추론 성능과 관련된 요약 정보를 performance.log에 기록합니다."""
        log_info = f"Status: {status} | Info: {info} | Prompt Len: {len(prompt)} | Resp Len: {len(response)}"
        self.log(log_type="inference", duration=duration, info=log_info)


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

    def _cleanup_old_logs(self, keep_days: int = 7):
        """설정된 기간보다 오래된 로그 파일을 삭제하여 디스크 공간을 관리합니다."""
        try:
            current_time = time.time()
            for log_file in self.trace_dir.glob("trace_*.jsonl"):
                file_age_days = (current_time - log_file.stat().st_mtime) / (24 * 3600)
                if file_age_days > keep_days:
                    log_file.unlink()
        except Exception as e:
            # 로깅 초기화 중 에러가 발생해도 프로세스가 중단되지 않도록 예외 처리
            print(f"오래된 로그 삭제 중 오류 발생: {e}")

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
        """API 키 등 민감 정보가 포함된 필드와 문자열 데이터를 마스킹 처리합니다."""
        if isinstance(data, dict):
            filtered = {}
            for k, v in data.items():
                if any(secret in k.lower() for secret in ["key", "token", "secret", "auth"]):
                    filtered[k] = "********"
                else:
                    filtered[k] = self._filter_sensitive_data(v)
            return filtered
        if isinstance(data, list):
            return [self._filter_sensitive_data(i) for i in data]
        if isinstance(data, str):
            data = EMAIL_REGEX.sub("[EMAIL_MASKED]", data)
            data = PHONE_REGEX.sub("[PHONE_MASKED]", data)
            data = RRN_REGEX.sub("[RRN_MASKED]", data)
            data = API_KEY_REGEX.sub("[API_KEY_MASKED]", data)
            return data
        return data


def setup_global_logging():
    """시스템 기본 로깅 설정 (app.log 용).

    파일 로깅(로그 디렉토리/파일)을 쓸 수 없는 경우(권한 등)에도 앱 기동을 죽이지 않고
    콘솔 로깅으로 graceful-degrade 한다. (init-host-dirs.sh 가 권한을 선제 처리하지만 방어선)
    """
    from logging.handlers import RotatingFileHandler

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    file_logging_error: Exception | None = None
    try:
        # 디렉토리가 없으면 생성
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOGS_DIR / "app.log"
        handlers.append(RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"))
    except OSError as e:
        # 로그 디렉토리/파일에 쓸 수 없어도(권한 등) 콘솔 로깅으로 계속 진행해 기동 중단을 막는다.
        file_logging_error = e

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )
    if file_logging_error is not None:
        logging.getLogger(__name__).warning(
            f"파일 로깅 비활성화 — 콘솔 전용으로 진행합니다(원인: {file_logging_error})."
        )
    # 노이즈 제거
    for name in ["httpx", "google", "langchain"]:
        logging.getLogger(name).setLevel(logging.WARNING)
    # sentence-transformers는 cache_folder 사용 시 cache_dir deprecation 경고를 데코레이터 로거로
    # 모델 로드마다 출력한다(warnings가 아닌 logging이라 filterwarnings로는 억제 불가). ERROR로 올려 억제.
    logging.getLogger("sentence_transformers.util.decorators").setLevel(logging.ERROR)
