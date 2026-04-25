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
    log_file = LOGS_DIR / "app.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8")
        ],
        force=True
    )
    # 노이즈 제거
    for name in ["httpx", "google", "langchain"]:
        logging.getLogger(name).setLevel(logging.WARNING)
