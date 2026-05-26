import json
import logging
import threading
from pathlib import Path
from typing import Any

from src.common.config import settings
from src.core.cache import SemanticCache
from src.core.storage import StorageManager
from src.pipeline.ingestion import IngestionPipeline
from src.pipeline.strategies import DoclingPDFParserStrategy, ManualParserStrategy, ParserStrategy
from src.utils.file_utils import generate_file_hash
from src.utils.logger import TracingLogger
from src.utils.paths import CACHE_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.utils.unicode import normalize_path_to_nfc, normalize_to_nfc

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """전체 파이프라인(Ingestion & Inference)을 총괄하는 오케스트레이터 (싱글톤)"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                # 초기화는 한 번만 수행
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            logger.info("PipelineOrchestrator 초기화...")
            self.parser_type = settings.PARSER_TYPE.lower()
            self.strategy = self._get_parser_strategy()
            # StorageManager 인스턴스 생성 및 IngestionPipeline에 전달
            self.storage_manager = StorageManager(PROCESSED_DATA_DIR, CACHE_DIR)
            self.ingestion_pipeline = IngestionPipeline(self.strategy, storage_manager=self.storage_manager)
            self.tracing_logger = TracingLogger()
            self.manifest_path = PROCESSED_DATA_DIR / "manifest.json"
            self.cache = SemanticCache()
            self._initialized = True
            logger.info("PipelineOrchestrator 초기화 완료.")

    def _load_manifest(self) -> dict[str, Any]:
        """manifest.json 파일을 로드합니다. 파일이 없거나 손상된 경우, 빈 2.0 매니페스트를 반환합니다."""
        default_manifest = {"version": "2.0", "global_parser_type": self.parser_type, "files": {}}
        if not self.manifest_path.exists():
            logger.warning("Manifest 파일이 없어 전체 재색인을 수행합니다.")
            return default_manifest
        try:
            with open(self.manifest_path, encoding="utf-8") as f:
                data = json.load(f)

            # 하위 호환성 처리 (1.0 규격인 경우 자동 변환)
            if not isinstance(data, dict) or "version" not in data:
                logger.info("이전 규격(1.0)의 Manifest가 감지되어 2.0 규격으로 자동 전환합니다.")
                files_converted = {}
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, str):
                            files_converted[k] = {"hash": v, "parser_type": self.parser_type}
                data = {"version": "2.0", "global_parser_type": self.parser_type, "files": files_converted}
            return data
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning("Manifest 파일이 손상되었거나 찾을 수 없어 전체 재색인을 수행합니다.")
            if self.manifest_path.exists():
                self.manifest_path.unlink()
            return default_manifest

    def _save_manifest(self, manifest: dict[str, Any]):
        """처리 완료 후 새로운 manifest 상태를 저장합니다."""
        try:
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            logger.info(f"Manifest 업데이트 완료: {self.manifest_path}")
        except Exception as e:
            logger.error(f"Manifest 저장 실패: {e}")

    def _get_parser_strategy(self) -> ParserStrategy:
        """설정된 파서 타입에 따라 전략을 반환하며 가용성을 검증합니다."""
        if self.parser_type == "docling":
            try:
                import docling  # noqa: F401

                return DoclingPDFParserStrategy()
            except ImportError:
                logger.error(
                    "'docling' 파서용 'docling' 라이브러리가 없습니다. "
                    "'manual'로 강제 전환합니다.\n"
                    "설치: pip install docling"
                )
                return ManualParserStrategy()

        return ManualParserStrategy()

    def _calculate_delta(
        self, all_files: list[Path], old_manifest: dict[str, Any]
    ) -> tuple[list[Path], list[str], dict[str, Any]]:
        """현재 파일 상태와 이전 상태를 비교하여 변경 사항(Delta)을 계산합니다."""
        old_files = old_manifest.get("files", {})

        new_files = {}
        for f in all_files:
            rel_path = normalize_path_to_nfc(f.relative_to(RAW_DATA_DIR))
            old_info = old_files.get(rel_path, {})
            if not old_info:
                # NFD-NFC 매치 백업
                for k, v in old_files.items():
                    if normalize_to_nfc(k) == rel_path:
                        old_info = v
                        break
            file_parser_type = old_info.get("parser_type", self.parser_type)
            h = generate_file_hash(f, file_parser_type)
            new_files[rel_path] = {"hash": h, "parser_type": file_parser_type}

        new_manifest = {"version": "2.0", "global_parser_type": self.parser_type, "files": new_files}

        files_to_process = []
        for rel_path, info in new_files.items():
            old_info = old_files.get(rel_path)
            if not old_info or old_info.get("hash") != info["hash"]:
                files_to_process.append(RAW_DATA_DIR / rel_path)

        source_ids_to_delete = []
        for rel_path, old_info in old_files.items():
            new_info = new_files.get(rel_path)
            if not new_info or old_info.get("hash") != new_info["hash"]:
                old_hash = old_info.get("hash")
                if old_hash:
                    source_ids_to_delete.append(old_hash)

        return files_to_process, source_ids_to_delete, new_manifest

    def _process_changes(  # noqa: C901
        self,
        files_to_process: list[Path],
        source_ids_to_delete: list[str],
        session: Any,
        relative_paths_to_delete: list[str] | None = None,
        progress_callback=None,
        new_manifest: dict[str, Any] | None = None,
    ):
        """도출된 변경 사항(DB 삭제, 파싱, 업서트, 인덱스 갱신)을 순차적으로 수행합니다."""
        if files_to_process or source_ids_to_delete or relative_paths_to_delete:
            with session.trace_step("cache_flush"):
                logger.info("데이터 변경이 감지되어 시맨틱 캐시를 초기화합니다.")
                self.cache.flush()

        if source_ids_to_delete or relative_paths_to_delete:
            with session.trace_step("db_cleanup"):
                filenames_to_delete = []
                try:
                    old_manifest = self._load_manifest()
                    old_files = old_manifest.get("files", {})
                    for rel_path, info in old_files.items():
                        if info.get("hash") in source_ids_to_delete:
                            filenames_to_delete.append(Path(rel_path).name)
                except Exception as e:
                    logger.error(f"삭제 파일명 추출 중 오류 발생: {e}")

                self.ingestion_pipeline.cleanup_db(
                    source_ids_to_delete=source_ids_to_delete,
                    relative_paths_to_delete=relative_paths_to_delete,
                    filenames_to_delete=filenames_to_delete,
                )

        if files_to_process:
            with session.trace_step("parse_and_chunk") as step:
                file_parser_types = {}
                if new_manifest and "files" in new_manifest:
                    for rel_path, info in new_manifest["files"].items():
                        file_parser_types[rel_path] = info.get("parser_type", self.parser_type)

                processed_data = self.ingestion_pipeline.process_and_chunk(
                    files_to_process, progress_callback=progress_callback, file_parser_types=file_parser_types
                )
                step["parent_chunk_count"] = len(processed_data)

            if processed_data:
                with session.trace_step("save_json") as step:
                    save_paths = self.ingestion_pipeline.save_processed_data(processed_data)
                    step["saved_files"] = [p.name for p in save_paths]

                if progress_callback:
                    try:
                        progress_callback(
                            len(files_to_process),
                            len(files_to_process),
                            "임베딩 변환 및 벡터 적재 중 (시간이 소요될 수 있습니다)",
                        )
                    except Exception as cb_e:
                        logger.error(f"진행 상황 콜백 호출 실패: {cb_e}")

                with session.trace_step("db_upsert"):
                    self.ingestion_pipeline.upsert_to_db(processed_data)

        if files_to_process or source_ids_to_delete or relative_paths_to_delete:
            with session.trace_step("bm25_update") as step:
                try:
                    from src.vector_db.bm25_manager import BM25Manager

                    BM25Manager()  # Rebuilds 수퍼 클래스
                    step["status"] = "success"
                except Exception as e:
                    step["status"] = f"failed: {e}"

    def process_single_file(self, file_path: Path, force: bool = False) -> bool:
        """특정 단일 파일에 대해 즉시 색인 업데이트를 수행합니다. (Atomic Update)"""
        if not file_path.exists():
            logger.error(f"파일을 찾을 수 없습니다: {file_path}")
            return False

        try:
            relative_path = str(file_path.relative_to(RAW_DATA_DIR))
        except ValueError:
            logger.error(f"파일이 RAW_DATA_DIR 외부에 있습니다: {file_path}")
            return False

        logger.info(f"단일 파일 개별 동기화 시작: {relative_path}")

        with self.tracing_logger.start_session(type="single_file_sync", file=relative_path) as session:
            # 1. 기존 데이터 삭제 대상 식별 (동일 상대 경로의 모든 데이터)
            # Delete-before-Insert 원칙에 따라 이전 파서 타입 데이터까지 포함하여 삭제
            relative_paths_to_delete = [relative_path]

            # 2. 변경 사항 처리 실행
            self._process_changes(
                files_to_process=[file_path],
                source_ids_to_delete=[],
                session=session,
                relative_paths_to_delete=relative_paths_to_delete,
            )

            # 3. 매니페스트 업데이트
            manifest = self._load_manifest()
            manifest[relative_path] = generate_file_hash(file_path, self.parser_type)
            self._save_manifest(manifest)

        logger.info(f"단일 파일 개별 동기화 완료: {relative_path}")
        return True

    def run_ingestion(  # noqa: C901
        self,
        force: bool = False,
        parser_type: str | None = None,
        progress_callback=None,
        target_files: list[Path] | None = None,
    ):
        """전체 데이터 구축 파이프라인 실행"""
        if parser_type:
            parser_type_lower = parser_type.lower()
            if self.parser_type != parser_type_lower:
                logger.info(f"파서 타입 변경 감지: {self.parser_type} -> {parser_type_lower}")
                self.parser_type = parser_type_lower
                self.strategy = self._get_parser_strategy()
                self.ingestion_pipeline.strategy = self.strategy

        logger.info(f"데이터 구축 파이프라인을 시작합니다. (전략: {self.parser_type}, 강제 재색인: {force})")

        with self.tracing_logger.start_session(type="ingestion", parser_type=self.parser_type) as session:
            with session.trace_step("scan_and_check_updates") as step:
                all_files = self.ingestion_pipeline.scan_files()
                old_manifest = self._load_manifest()
                relative_paths_to_delete = []

                if not all_files and not old_manifest:
                    logger.warning("처리할 파일이 없고 이전 기록도 없습니다.")
                    session.data["status"] = "no_files"
                    return

                if force:
                    logger.info(
                        "강제 동기화가 요청되었습니다. 모든 기존 데이터를 완전히 삭제하고 전체 재색인을 수행합니다."
                    )

                    # 강제 초기화 시 기존 데이터를 물리적으로 모두 정리 (vector DB 및 파싱 캐시 등)
                    logger.info("ChromaDB 컬렉션 및 가공 파일 물리적 초기화 시작...")
                    self.ingestion_pipeline.db_manager.reset_collection()

                    # processed_dir의 JSON 파일들 및 CACHE_DIR의 pkl 파일들 정리
                    for f in self.storage_manager.scan_processed_files():
                        try:
                            self.storage_manager.delete_file(f)
                        except Exception as e:
                            logger.error(f"JSON 파일 삭제 실패 {f}: {e}")

                    for f in CACHE_DIR.glob("*_parsed.pkl"):
                        try:
                            self.storage_manager.delete_file(f)
                        except Exception as e:
                            logger.error(f"캐시 파일 삭제 실패 {f}: {e}")

                    files_to_process = all_files
                    source_ids_to_delete = []  # 이미 물리적으로 모두 삭제했으므로 부분 삭제 프로세스는 건너뜀
                    relative_paths_to_delete = None

                    new_files = {}
                    for f in all_files:
                        rel_path = normalize_path_to_nfc(f.relative_to(RAW_DATA_DIR))
                        new_files[rel_path] = {
                            "hash": generate_file_hash(f, self.parser_type),
                            "parser_type": self.parser_type,
                        }
                    new_manifest = {"version": "2.0", "global_parser_type": self.parser_type, "files": new_files}
                else:
                    files_to_process, source_ids_to_delete, new_manifest = self._calculate_delta(
                        all_files, old_manifest
                    )
                    # 파서 변경 대응: 업데이트 대상 파일들은 상대 경로 기준으로도 삭제를 병행
                    # (Delete-before-Insert 원자성 확보)
                    files_to_process_relative = [str(f.relative_to(RAW_DATA_DIR)) for f in files_to_process]
                    # 이미 source_ids_to_delete로 처리되는 항목(변경분)은 제외하고,
                    # 삭제된 파일들만 relative_paths_to_delete에 추가
                    relative_paths_to_delete = [p for p in old_manifest if p not in new_manifest]
                    # 파서 변경 대응을 위해 현재 처리 대상인 파일들의 상대 경로도 추가 (중복되더라도 DB쪽은 안전)
                    relative_paths_to_delete += files_to_process_relative

                # target_files 부분 동기화 필터링 적용
                if target_files is not None:
                    target_paths = {p.resolve() for p in target_files}
                    filtered_files_to_process = [f for f in files_to_process if f.resolve() in target_paths]

                    # target_files에 해당하지 않는 문서들의 구 해시값은 delete 목록에서 보존
                    target_names = {p.name for p in target_files}
                    filtered_source_ids_to_delete = []
                    for sid in source_ids_to_delete:
                        old_files = old_manifest.get("files", {})
                        is_target = False
                        for rel_path, info in old_files.items():
                            if info.get("hash") == sid and Path(rel_path).name in target_names:
                                is_target = True
                                break
                        if is_target:
                            filtered_source_ids_to_delete.append(sid)

                    # new_manifest 내에서 처리되지 않은 문서의 메타데이터 보존
                    if new_manifest and "files" in new_manifest:
                        for rel_path in list(new_manifest["files"].keys()):
                            full_path = RAW_DATA_DIR / rel_path
                            if full_path.resolve() not in target_paths:
                                if rel_path in old_manifest.get("files", {}):
                                    new_manifest["files"][rel_path] = old_manifest["files"][rel_path]
                                else:
                                    new_manifest["files"].pop(rel_path, None)

                    files_to_process = filtered_files_to_process
                    source_ids_to_delete = filtered_source_ids_to_delete

                step["total_files"] = len(all_files)
                step["files_to_process"] = len(files_to_process)
                step["files_to_delete_in_db"] = len(source_ids_to_delete)

                logger.info(
                    f"파일 스캔 완료. 전체: {len(all_files)}, "
                    f"신규/변경/강제: {len(files_to_process)}, 삭제: {len(source_ids_to_delete)}"
                )

                if not files_to_process and not source_ids_to_delete:
                    logger.info("변경 사항이 없으므로 데이터 구축 작업을 건너뜁니다.")
                    session.data["status"] = "no_changes"
                    if force:
                        # 강제 초기화 후 파일이 없는 경우에도 manifest를 갱신하여 일관성 유지
                        self._save_manifest(new_manifest)
                    return

            self._process_changes(
                files_to_process,
                source_ids_to_delete,
                session,
                relative_paths_to_delete=relative_paths_to_delete,
                progress_callback=progress_callback,
                new_manifest=new_manifest,
            )

            with session.trace_step("update_manifest"):
                self._save_manifest(new_manifest)

        logger.info("데이터 구축 파이프라인 작업이 완료되었습니다.")

    def update_file_parser(self, file_name: str, new_parser_type: str, progress_callback=None):
        """특정 파일의 동기화 파서 타입을 변경하고, 해당 파일에 한해 즉각 재색인을 실행합니다."""
        new_parser_lower = new_parser_type.lower()
        old_manifest = self._load_manifest()
        old_files = old_manifest.get("files", {})

        normalized_file_name = normalize_to_nfc(file_name)
        target_rel_path = None
        for rel_path in old_files:
            normalized_rel_path = normalize_to_nfc(rel_path)
            is_match = (
                Path(normalized_rel_path).name == normalized_file_name or normalized_rel_path == normalized_file_name
            )
            if is_match:
                target_rel_path = rel_path
                break

        if not target_rel_path:
            for f in self.ingestion_pipeline.scan_files():
                normalized_f_name = normalize_to_nfc(f.name)
                if normalized_f_name == normalized_file_name:
                    target_rel_path = normalize_path_to_nfc(f.relative_to(RAW_DATA_DIR))
                    break

        if not target_rel_path:
            logger.warning(f"파서 변경 대상 파일을 찾을 수 없습니다: {file_name}")
            return

        logger.info(f"파일 {target_rel_path}의 파서를 {new_parser_lower}로 변경합니다.")

        if target_rel_path not in old_files:
            old_files[target_rel_path] = {}
        old_hash = old_files[target_rel_path].get("hash")
        hashes_to_delete = [old_hash] if old_hash else []
        logger.info(f"파서 변경 전 기존 ChromaDB 데이터 클린업 수행 (파일명: {file_name}, 해시: {hashes_to_delete})")
        self.ingestion_pipeline.cleanup_db(source_ids_to_delete=hashes_to_delete, filenames_to_delete=[file_name])
        old_files[target_rel_path]["hash"] = ""
        old_files[target_rel_path]["parser_type"] = new_parser_lower

        old_manifest["files"] = old_files
        self._save_manifest(old_manifest)

        # target_files를 전달하여 해당 파일만 동기화하도록 강제
        target_path = RAW_DATA_DIR / target_rel_path
        self.run_ingestion(force=False, progress_callback=progress_callback, target_files=[target_path])

    def update_multiple_file_parsers(self, file_parser_map: dict[str, str], progress_callback=None):  # noqa: C901
        """복수 파일의 동기화 파서 타입을 한꺼번에 변경하고, 대상 파일들을 즉각 재색인합니다."""
        old_manifest = self._load_manifest()
        old_files = old_manifest.get("files", {})

        updated_count = 0
        hashes_to_delete = []
        for file_name, new_parser in file_parser_map.items():
            new_parser_lower = new_parser.lower()

            normalized_file_name = normalize_to_nfc(file_name)

            target_rel_path = None
            for rel_path in old_files:
                normalized_rel_path = normalize_to_nfc(rel_path)
                is_match = (
                    Path(normalized_rel_path).name == normalized_file_name
                    or normalized_rel_path == normalized_file_name
                )
                if is_match:
                    target_rel_path = rel_path
                    break

            if not target_rel_path:
                for f in self.ingestion_pipeline.scan_files():
                    normalized_f_name = normalize_to_nfc(f.name)
                    if normalized_f_name == normalized_file_name:
                        target_rel_path = normalize_path_to_nfc(f.relative_to(RAW_DATA_DIR))
                        break

            if target_rel_path:
                logger.info(f"파일 {target_rel_path}의 파서를 {new_parser_lower}로 변경 등록합니다.")
                if target_rel_path not in old_files:
                    old_files[target_rel_path] = {}
                old_hash = old_files[target_rel_path].get("hash")
                if old_hash:
                    hashes_to_delete.append(old_hash)
                old_files[target_rel_path]["hash"] = ""
                old_files[target_rel_path]["parser_type"] = new_parser_lower
                updated_count += 1

        if hashes_to_delete:
            filenames_to_delete = list(file_parser_map.keys())
            logger.info(f"파서 일괄 변경 전 기존 ChromaDB 데이터 클린업 수행 (개수: {len(filenames_to_delete)})")
            self.ingestion_pipeline.cleanup_db(
                source_ids_to_delete=hashes_to_delete, filenames_to_delete=filenames_to_delete
            )

        if updated_count > 0:
            old_manifest["files"] = old_files
            self._save_manifest(old_manifest)

            # 변경된 파일들에 대해서만 일괄 부분 동기화 가동
            target_paths = []
            for file_name in file_parser_map:
                target_rel_path = None
                for rel_path in old_files:
                    if Path(rel_path).name == file_name or rel_path == file_name:
                        target_rel_path = rel_path
                        break
                if not target_rel_path:
                    for f in self.ingestion_pipeline.scan_files():
                        if f.name == file_name:
                            target_rel_path = str(f.relative_to(RAW_DATA_DIR))
                            break
                if target_rel_path:
                    target_paths.append(RAW_DATA_DIR / target_rel_path)

            self.run_ingestion(force=False, progress_callback=progress_callback, target_files=target_paths)
