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
"""

import logging
import threading
from datetime import datetime

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
