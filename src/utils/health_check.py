import hashlib
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

# paths.py를 임포트하여 환경 부트스트랩 (sys.path 자동 설정됨)
try:
    from src.utils.paths import BASE_DIR, RAW_DATA_DIR, REQUIRED_DIRECTORIES, ensure_directories
except ImportError:
    # 직접 실행 시 src를 찾지 못할 경우를 대비한 로컬 임포트 (paths.py가 같은 폴더에 있으므로 가능)
    from paths import BASE_DIR, RAW_DATA_DIR, REQUIRED_DIRECTORIES, ensure_directories
from src.common.config import settings
from src.common.constants import MetadataFields
from src.utils.file_utils import generate_file_hash

# 로깅 설정
logger = logging.getLogger(__name__)


def check_env():
    """환경 변수 및 필수 설정 점검"""
    logger.info("[1/4] 환경 변수 및 설정 점검")
    load_dotenv()

    # 필수 키: 실제 동작에 필요한 API 키만 체크 (EMBEDDING_MODEL_NAME은 config 기본값으로 동작)
    # MODEL_TYPE에 따라 필요한 API 키가 달라지므로 model_type 기반으로 동적 체크
    model_type = settings.MODEL_TYPE.lower()
    all_pass = True

    # model_type별 필수 API 키 매핑
    required_by_model = {
        "gemini": ["GOOGLE_API_KEY"],
        "claude": ["ANTHROPIC_API_KEY"],
        "ollama": [],  # 로컬 모델이므로 API 키 불필요
    }
    required_keys = required_by_model.get(model_type, [])

    logger.info(f"  모델 타입: {model_type} | 체크 대상 키: {required_keys or '없음(로컬)'}")

    for key in required_keys:
        val = getattr(settings, key, None)
        if val and val != f"your_{key.lower()}_here":
            display_val = f"{val[:8]}****"
            logger.info(f"  [PASS] {key}: {display_val}")
        else:
            logger.error(f"  [FAIL] {key}: 누락됨 또는 기본값!")
            all_pass = False

    return all_pass


def check_data_integrity():
    """데이터 파일 및 경로 점검 (paths.py 표준 준수)"""
    logger.info("[2/4] 데이터 및 경로 정합성 점검")
    ensure_directories()

    logger.info(f"  프로젝트 루트: {BASE_DIR}")

    for path in REQUIRED_DIRECTORIES:
        if path.exists():
            count = len(list(path.glob("*"))) if path.is_dir() else 1
            logger.info(f"  [OK] {path.name} 경로 확인: {path} (항목 수: {count})")
        else:
            logger.error(f"  [MISSING] {path.name} 경로 누락: {path}")

    raw_files = []
    for ext in [".pdf", ".md", ".markdown"]:
        raw_files.extend(list(RAW_DATA_DIR.glob(f"**/*{ext}")))

    if raw_files:
        logger.info(f"  탐지된 원본 파일 ({len(raw_files)}개):")
        for f in raw_files:
            logger.info(f"    - {f.relative_to(RAW_DATA_DIR)}")
    else:
        logger.warning("  RAW_DATA_DIR가 비어 있거나 지원하는 파일이 없습니다.")


def check_model_loading():
    """임베딩 모델 로딩 상태 점검"""
    logger.info("[3/4] 로컬 임베딩 모델 로딩 점검")

    try:
        from src.models.embedder import BGEEmbedder

        model_name = settings.EMBEDDING_MODEL_NAME

        start_time = time.time()
        embedder = BGEEmbedder(model_name=model_name)
        embedder.encode(["Health check test"])
        duration = time.time() - start_time

        logger.info(f"  모델 로드 성공: {model_name}")
        logger.info(f"  테스트 인코딩 완료 (소요시간: {duration:.2f}s)")
        return True
    except Exception as e:
        logger.error(f"  모델 로드 중 오류 발생: {e}")
        return False


def check_database_status():
    """ChromaDB 연결 및 데이터 정합성 점검"""
    logger.info("[4/4] 벡터 데이터베이스(ChromaDB) 상태 점검")

    try:
        from src.vector_db.chroma_manager import ChromaDBManager

        db_manager = ChromaDBManager(collection_name="rag_collection")

        count = db_manager.get_count()
        logger.info(f"  DB 연결 성공 (컬렉션: {db_manager.collection_name})")
        logger.info(f"  총 적재된 청크 수: {count}")

        if count == 0:
            logger.warning("  DB가 비어 있습니다. 전처리가 필요합니다.")
            return

        # TODO: 향후 데이터 증가 시 페이징(limit, offset) 처리 필요
        all_data = db_manager.collection.get(include=["metadatas"])
        metadatas = all_data["metadatas"]

        db_file_map = {}
        for meta in metadatas:
            src_name = meta.get(MetadataFields.SRC_NAME, "UNKNOWN")
            src_id = meta.get(MetadataFields.SOURCE_ID, "UNKNOWN")
            if src_name not in db_file_map:
                db_file_map[src_name] = {"source_id": src_id, "count": 0}
            db_file_map[src_name]["count"] += 1

        logger.info("[정합성 비교] 로컬 파일 vs 벡터 DB")

        raw_files = []
        for ext in [".pdf", ".md", ".markdown"]:
            raw_files.extend(list(RAW_DATA_DIR.glob(f"**/*{ext}")))

        local_files_info = {f.name: generate_file_hash(f, parser_type=settings.PARSER_TYPE) for f in raw_files}
        all_filenames = set(local_files_info.keys()) | set(db_file_map.keys())

        for fname in sorted(all_filenames):
            local_id = local_files_info.get(fname)
            db_info = db_file_map.get(fname)

            if local_id and db_info:
                if local_id == db_info["source_id"]:
                    logger.info(f"  [MATCH] {fname:30} | 일치 (청크: {db_info['count']:3d})")
                else:
                    logger.error(f"  [MISMATCH] {fname:30} | 내용 변경됨 (Update 필요)")
            elif local_id:
                logger.warning(f"  [MISSING_IN_DB] {fname:30} | DB에 없음 (Ingest 필요)")
            else:
                logger.warning(f"  [DELETED_LOCALLY] {fname:30} | 파일 삭제됨 (DB 정리 필요)")

    except Exception as e:
        logger.error(f"  DB 정합성 점검 중 오류 발생: {e}")


def run_full_diagnostics(silent: bool = False, check_model: bool = True) -> tuple[bool, dict]:
    """
    전체 자가 진단 실행
    :param silent: True일 경우 로그 출력을 억제함 (에러 제외)
    :param check_model: 로컬 모델 로딩 및 테스트 인코딩 수행 여부 (시간 소요 방지)
    """
    if not silent:
        logger.info("=" * 60)
        logger.info("RAG 시스템 자가 진단 (Health Check) 시작")
        logger.info("=" * 60)

    current_level = logger.getEffectiveLevel()
    if silent:
        logger.setLevel(logging.ERROR)

    env_ok = check_env()
    check_data_integrity()

    # 모델 로드 체크는 옵션에 따라 수행 (시간 절약)
    model_ok = True
    if check_model:
        model_ok = check_model_loading()

    db_report = {"connected": False, "count": 0}
    try:
        from src.vector_db.chroma_manager import ChromaDBManager

        db_manager = ChromaDBManager(collection_name="rag_collection")
        db_report["count"] = db_manager.get_count()
        db_report["connected"] = True
        if not silent:
            check_database_status()
    except Exception:
        pass

    is_healthy = env_ok and model_ok and db_report["connected"]

    if silent:
        logger.setLevel(current_level)

    if not silent:
        logger.info("=" * 60)
        if is_healthy:
            logger.info("진단 완료: 핵심 컴포넌트가 정상 동작 중입니다.")
        else:
            logger.error("진단 완료: 일부 설정이나 모델 로드에 문제가 있습니다.")
        logger.info("=" * 60)

    return is_healthy, {"env": env_ok, "model": model_ok, "db": db_report}


if __name__ == "__main__":
    run_full_diagnostics()
