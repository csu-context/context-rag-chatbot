import logging
import os
from pathlib import Path

# 로깅 설정
logger = logging.getLogger(__name__)

# 1. BASE_DIR 정의: src/utils/paths.py 기준으로 프로젝트 루트를 가리킴
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# 2. 주요 디렉토리 상수화 (상대 경로 기반 resolve() 처리)
DATA_DIR = (BASE_DIR / "data").resolve()
RAW_DATA_DIR = (DATA_DIR / "raw").resolve()
PROCESSED_DATA_DIR = (DATA_DIR / "processed").resolve()
EVAL_DATA_DIR = (DATA_DIR / "eval").resolve()

VECTOR_DB_DIR = (BASE_DIR / "vector_db").resolve()
BACKUP_DIR = (DATA_DIR / "backups").resolve()
MODELS_DIR = (BASE_DIR / "models").resolve()
LOGS_DIR = (BASE_DIR / "logs").resolve()
EVAL_LOGS_DIR = (LOGS_DIR / "eval").resolve()

# 4. 설정 및 사전 파일 경로
# 동의어 사전은 도메인 종속 어휘이므로 패키지 코드 밖(data/config/)에서 주입한다.
# SYNONYMS_PATH 환경변수로 외부 볼륨 경로를 오버라이드할 수 있고, 파일이 없으면
# BM25Tokenizer가 빈 사전으로 폴백한다(범용 매뉴얼 RAG 기본 동작).
SYNONYMS_FILE = Path(os.getenv("SYNONYMS_PATH") or (DATA_DIR / "config" / "synonyms.json")).resolve()
CACHE_DIR = (BASE_DIR / ".cache").resolve()
BM25_CACHE_FILE = CACHE_DIR / "bm25_index.pkl"  # legacy — kept for reference only
BM25_CACHE_DIR = CACHE_DIR / "bm25_v2"
CROSS_ENCODER_CACHE_DIR = CACHE_DIR / "cross-encoder"
PROMPTS_DIR = (BASE_DIR / "prompts").resolve()


# 3. 디렉토리 목록 (자동 생성용)
REQUIRED_DIRECTORIES = [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    VECTOR_DB_DIR,
    BACKUP_DIR,
    MODELS_DIR,
    LOGS_DIR,
    CACHE_DIR,
]


def ensure_directories():
    """
    프로젝트 실행 시 필요한 모든 디렉토리가 없을 경우 자동으로 생성함.
    parents=True: 부모 디렉토리가 없으면 함께 생성
    exist_ok=True: 이미 디렉토리가 존재해도 에러를 발생시키지 않음
    """
    for directory in REQUIRED_DIRECTORIES:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        else:
            # 선택사항: 이미 존재할 경우 로그를 남기지 않거나 디버깅용으로만 사용
            pass


if __name__ == "__main__":
    # 유틸리티 단독 실행 시 테스트 및 초기화 수행
    logger.info(f"Project Base Directory: {BASE_DIR}")
    ensure_directories()
    logger.info("All required directories are verified/created.")
