import logging
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
            return {}

        # [정합성 분석] 로컬 파일 vs 벡터 DB vs 물리 데이터
        logger.info("[정합성 분석] 로컬 파일 vs 벡터 DB")

        local_files_info = _get_local_files_info()
        db_rel_path_map = _get_db_rel_path_map(db_manager)

        return _analyze_anomalies(local_files_info, db_rel_path_map)
    except Exception as e:
        logger.error(f"  DB 정합성 점검 중 오류 발생: {e}")
        return {}


def _get_local_files_info() -> dict[str, dict]:
    """로컬 raw 데이터 파일들의 정보(상대 경로, 해시)를 추출"""
    raw_files = []
    for ext in [".pdf", ".md", ".markdown"]:
        raw_files.extend(list(RAW_DATA_DIR.glob(f"**/*{ext}")))

    local_files_info = {}
    for f in raw_files:
        rel_path = str(f.relative_to(RAW_DATA_DIR))
        local_files_info[rel_path] = {
            "name": f.name,
            "hash": generate_file_hash(f, parser_type=settings.PARSER_TYPE),
            "path": f,
        }
    return local_files_info


def _get_db_rel_path_map(db_manager, batch_size: int = 1000) -> dict[str, list]:
    """DB의 메타데이터를 가져와 상대 경로별로 그룹화"""
    db_rel_path_map = {}
    offset = 0

    while True:
        batch = db_manager.collection.get(include=["metadatas"], limit=batch_size, offset=offset)
        db_metas = batch["metadatas"]

        for meta in db_metas:
            rel_path = meta.get(MetadataFields.RELATIVE_PATH)
            # 하위 호환성: relative_path가 없으면 src_name 활용
            if not rel_path:
                rel_path = meta.get(MetadataFields.SRC_NAME, "UNKNOWN")
            if rel_path not in db_rel_path_map:
                db_rel_path_map[rel_path] = []
            db_rel_path_map[rel_path].append(meta)

        offset += batch_size
        if len(db_metas) < batch_size:
            break

    return db_rel_path_map


def _analyze_anomalies(local_files_info: dict, db_rel_path_map: dict) -> dict:
    """로컬 정보와 DB 정보를 비교하여 이상 징후 탐지"""
    anomalies = {
        "ghost_chunks": [],  # 로컬에 없는데 DB에 있음
        "mismatched_hash": [],  # 로컬과 DB의 해시가 다름
        "duplicate_parsers": [],  # 동일 파일에 여러 파서 타입이 공존
        "missing_in_db": [],  # 로컬에는 있는데 DB에 없음
    }

    all_rel_paths = set(local_files_info.keys()) | set(db_rel_path_map.keys())

    for rel_path in sorted(all_rel_paths):
        local_info = local_files_info.get(rel_path)
        db_chunks = db_rel_path_map.get(rel_path, [])

        if local_info and db_chunks:
            _check_local_db_mismatch(rel_path, local_info, db_chunks, anomalies)
        elif local_info:
            logger.warning(f"  [MISSING] {rel_path:30} | DB에 없음 (Ingest 필요)")
            anomalies["missing_in_db"].append(rel_path)
        else:
            logger.warning(f"  [GHOST] {rel_path:30} | 파일 삭제됨 (DB 정리 필요)")
            anomalies["ghost_chunks"].append(rel_path)

    return anomalies


def _get_parser_type(meta: dict) -> str:
    """메타데이터에서 파서 타입을 추출 (구 스키마 호환)"""
    return meta.get(MetadataFields.PARSER_TYPE) or meta.get("parser") or settings.PARSER_TYPE


def _check_local_db_mismatch(rel_path: str, local_info: dict, db_chunks: list, anomalies: dict):
    """동일 파일에 대해 로컬과 DB의 메타데이터 일치 여부 세부 확인"""
    parser_types = {_get_parser_type(m) for m in db_chunks}
    source_ids = {m.get(MetadataFields.SOURCE_ID) for m in db_chunks}

    if len(parser_types) > 1:
        logger.error(f"  [DUP_PARSER] {rel_path:30} | 여러 파서 공존: {parser_types}")
        anomalies["duplicate_parsers"].append(rel_path)

    # DB에 저장된 실제 파서 정보를 바탕으로 해시값 다시 동적 계산
    active_parser = next(iter(parser_types)) if parser_types else settings.PARSER_TYPE
    current_sid = generate_file_hash(local_info["path"], parser_type=active_parser)

    if current_sid not in source_ids:
        logger.error(f"  [MISMATCH] {rel_path:30} | 해시 불일치 (Update 필요)")
        anomalies["mismatched_hash"].append(rel_path)
    else:
        logger.info(f"  [MATCH] {rel_path:30} | 일치 (청크: {len(db_chunks):3d})")


def _repair_ghost_chunks(pipeline, ghosts: list[str]):
    """유령 청크 정리 로직 분리"""
    logger.info(f"유령 청크 {len(ghosts)}개 정리 중...")
    for rel_path in ghosts:
        # relative_path가 없는 legacy 청크 대응: src_name으로 조회하여 relative_path가 없는 것들 수집
        db_data = pipeline.db_manager.collection.get(
            where={MetadataFields.SRC_NAME: Path(rel_path).name}, include=["metadatas"]
        )
        legacy_sids = [
            m.get(MetadataFields.SOURCE_ID) for m in db_data["metadatas"] if not m.get(MetadataFields.RELATIVE_PATH)
        ]

        if legacy_sids:
            logger.info(f"  Legacy 유령 청크 발견 ({rel_path}): {len(legacy_sids)}개 삭제")
            pipeline.cleanup_db(source_ids_to_delete=list(set(legacy_sids)))

        # 정상적인 relative_path 기반 삭제 시도 (idempotent)
        pipeline.cleanup_db(source_ids_to_delete=[], relative_paths_to_delete=[rel_path])


def _repair_duplicate_parsers(pipeline, duplicates: list[str]):
    """중복 파서 데이터 정리 로직 분리"""
    logger.info(f"중복 파서 데이터 {len(duplicates)}개 정리 중...")
    for rel_path in duplicates:
        # relative_path가 있는 경우와 없는(legacy) 경우 모두 고려하여 src_name으로 조회
        db_data = pipeline.db_manager.collection.get(
            where={MetadataFields.SRC_NAME: Path(rel_path).name}, include=["metadatas"]
        )

        sids_to_delete = []
        for meta in db_data["metadatas"]:
            # relative_path가 일치하거나 (신규), relative_path가 없으면서 이름이 같은 경우 (Legacy)
            is_match = meta.get(MetadataFields.RELATIVE_PATH) == rel_path or not meta.get(MetadataFields.RELATIVE_PATH)
            if is_match and _get_parser_type(meta) != settings.PARSER_TYPE:
                sids_to_delete.append(meta.get(MetadataFields.SOURCE_ID))

        if sids_to_delete:
            logger.info(f"  중복 파서 데이터 삭제 ({rel_path}): {len(sids_to_delete)}개")
            pipeline.cleanup_db(source_ids_to_delete=list(set(sids_to_delete)))


def repair_integrity(anomalies: dict, target_parser: str | None = None):
    """
    탐지된 이상 징후를 바탕으로 DB 정합성을 자동 복구합니다.
    - ghost_chunks: DB에서만 존재하는 찌꺼기 삭제
    - duplicate_parsers: 파서 중복 청크 클린업
    - mismatched_hash: 해시 불일치 파일 재색인 및 업데이트
    """
    if not anomalies:
        return

    from src.pipeline import PipelineOrchestrator

    orchestrator = PipelineOrchestrator()
    pipeline = orchestrator.ingestion_pipeline

    # 1. 유령 청크 제거
    ghosts = anomalies.get("ghost_chunks", [])
    if ghosts:
        _repair_ghost_chunks(pipeline, ghosts)

    # 2. 중복 파서 데이터 정리
    duplicates = anomalies.get("duplicate_parsers", [])
    if duplicates:
        _repair_duplicate_parsers(pipeline, duplicates)

    # 3. 해시 불일치 파일 재색인 및 업데이트
    mismatches = anomalies.get("mismatched_hash", [])
    if mismatches:
        logger.info(f"해시 불일치 파일 {len(mismatches)}개 동기화 및 재색인 실행 중...")
        for rel_path in mismatches:
            file_name = Path(rel_path).name
            # DB에 현재 적재되어 있던 파서 정보 확인
            db_data = pipeline.db_manager.collection.get(
                where={MetadataFields.SRC_NAME: file_name}, include=["metadatas"]
            )
            parser_type = settings.PARSER_TYPE
            if target_parser:
                parser_type = target_parser
            elif db_data and db_data["metadatas"]:
                parser_type = _get_parser_type(db_data["metadatas"][0])

            logger.info(f"  재색인 파일 ({rel_path}) | 적용 파서: {parser_type}")
            orchestrator.update_file_parser(file_name=file_name, new_parser_type=parser_type)

    logger.info("정합성 복구 작업이 완료되었습니다.")


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

    db_report = {"connected": False, "count": 0, "anomalies": {}}
    try:
        from src.vector_db.chroma_manager import ChromaDBManager

        db_manager = ChromaDBManager(collection_name="rag_collection")
        db_report["count"] = db_manager.get_count()
        db_report["connected"] = True

        # 정합성 상세 리포트 생성
        db_report["anomalies"] = check_database_status()
    except Exception as e:
        logger.error(f"진단 중 오류 발생: {e}")
        pass

    # 건강 상태 정의: 환경 변수 OK + DB 연결 OK + 중대한 정합성 오류(유령 청크 등) 없음
    has_critical_anomaly = len(db_report["anomalies"].get("ghost_chunks", [])) > 0
    is_healthy = env_ok and model_ok and db_report["connected"] and not has_critical_anomaly

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
