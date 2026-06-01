import importlib.util
import logging
import threading
from pathlib import Path
from typing import Any

from src.common.config import settings
from src.core.cache import SemanticCache
from src.core.storage import StorageManager
from src.pipeline.ingestion import IngestionPipeline
from src.pipeline.manifest_manager import ManifestManager
from src.pipeline.strategies import DoclingPDFParserStrategy, ManualParserStrategy, ParserStrategy
from src.utils.file_utils import generate_file_hash
from src.utils.logger import TracingLogger
from src.utils.paths import CACHE_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.utils.unicode import normalize_path_to_nfc

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """전체 파이프라인(Ingestion & Inference)을 총괄하는 오케스트레이터 (싱글톤)"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
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
            self.storage_manager = StorageManager(PROCESSED_DATA_DIR, CACHE_DIR)
            self.ingestion_pipeline = IngestionPipeline(self.strategy, storage_manager=self.storage_manager)
            self.manifest_manager = ManifestManager(self.parser_type)
            self.tracing_logger = TracingLogger()
            self.cache = SemanticCache()
            self._initialized = True
            logger.info("PipelineOrchestrator 초기화 완료.")

    def _get_parser_strategy(self) -> ParserStrategy:
        """설정된 파서 타입에 따라 전략을 반환하며 가용성을 검증합니다."""
        if self.parser_type == "docling":
            if importlib.util.find_spec("docling") is not None:
                return DoclingPDFParserStrategy()
            logger.error(
                "'docling' 파서용 'docling' 라이브러리가 없습니다. 'manual'로 강제 전환합니다.\n"
                "설치: pip install docling"
            )
            return ManualParserStrategy()

        return ManualParserStrategy()

    def _cleanup_db(
        self, session: Any, source_ids_to_delete: list[str], relative_paths_to_delete: list[str] | None
    ) -> None:
        with session.trace_step("db_cleanup"):
            filenames_to_delete = []
            try:
                old_files = self.manifest_manager.load_manifest().get("files", {})
                filenames_to_delete = [
                    Path(rel).name for rel, info in old_files.items() if info.get("hash") in source_ids_to_delete
                ]
            except Exception as e:
                logger.error(f"삭제 파일명 추출 중 오류 발생: {e}")
            self.ingestion_pipeline.cleanup_db(
                source_ids_to_delete=source_ids_to_delete,
                relative_paths_to_delete=relative_paths_to_delete,
                filenames_to_delete=filenames_to_delete,
            )

    def _parse_and_upsert(
        self, session: Any, files_to_process: list[Path], new_manifest: dict[str, Any] | None, progress_callback
    ) -> None:
        with session.trace_step("parse_and_chunk") as step:
            file_parser_types = {
                rel: info.get("parser_type", self.parser_type)
                for rel, info in (new_manifest or {}).get("files", {}).items()
            }
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

    def _rebuild_bm25(self, session: Any) -> None:
        with session.trace_step("bm25_update") as step:
            try:
                from src.vector_db.bm25_manager import BM25Manager

                BM25Manager()
                step["status"] = "success"
            except Exception as e:
                step["status"] = f"failed: {e}"

    def _process_changes(
        self,
        files_to_process: list[Path],
        source_ids_to_delete: list[str],
        session: Any,
        relative_paths_to_delete: list[str] | None = None,
        progress_callback=None,
        new_manifest: dict[str, Any] | None = None,
    ):
        has_changes = bool(files_to_process or source_ids_to_delete or relative_paths_to_delete)

        if has_changes:
            with session.trace_step("cache_flush"):
                logger.info("데이터 변경이 감지되어 시맨틱 캐시를 초기화합니다.")
                self.cache.flush()

        if source_ids_to_delete or relative_paths_to_delete:
            self._cleanup_db(session, source_ids_to_delete, relative_paths_to_delete)

        if files_to_process:
            self._parse_and_upsert(session, files_to_process, new_manifest, progress_callback)

        if has_changes:
            self._rebuild_bm25(session)

    def _reset_all_data(self) -> None:
        logger.info("ChromaDB 컬렉션 및 가공 파일 물리적 초기화 시작...")
        self.ingestion_pipeline.db_manager.reset_collection()

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

    def process_single_file(self, file_path: Path, force: bool = False) -> bool:
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
            relative_paths_to_delete = [relative_path]

            self._process_changes(
                files_to_process=[file_path],
                source_ids_to_delete=[],
                session=session,
                relative_paths_to_delete=relative_paths_to_delete,
            )

            manifest = self.manifest_manager.load_manifest()
            manifest["files"][relative_path] = {
                "hash": generate_file_hash(file_path, self.parser_type),
                "parser_type": self.parser_type,
            }
            self.manifest_manager.save_manifest(manifest)

        logger.info(f"단일 파일 개별 동기화 완료: {relative_path}")
        return True

    def _update_parser_if_changed(self, parser_type: str | None) -> None:
        if parser_type:
            parser_type_lower = parser_type.lower()
            if self.parser_type != parser_type_lower:
                logger.info(f"파서 타입 변경 감지: {self.parser_type} -> {parser_type_lower}")
                self.parser_type = parser_type_lower
                self.manifest_manager.update_parser_type(self.parser_type)
                self.strategy = self._get_parser_strategy()
                self.ingestion_pipeline.strategy = self.strategy

    def _prepare_sync_plan(
        self, force: bool, all_files: list[Path], old_manifest: dict[str, Any]
    ) -> tuple[list[Path], list[str], list[str] | None, dict[str, Any]]:
        if force:
            logger.info("강제 동기화가 요청되었습니다. 모든 기존 데이터를 완전히 삭제하고 전체 재색인을 수행합니다.")
            self._reset_all_data()
            new_files = {
                normalize_path_to_nfc(f.relative_to(RAW_DATA_DIR)): {
                    "hash": generate_file_hash(f, self.parser_type),
                    "parser_type": self.parser_type,
                }
                for f in all_files
            }
            return all_files, [], None, self.manifest_manager.make_manifest(new_files)

        files_to_process, source_ids_to_delete, new_manifest = self.manifest_manager.calculate_delta(
            all_files, old_manifest, self.storage_manager, self.ingestion_pipeline.db_manager
        )
        files_to_process_relative = [str(f.relative_to(RAW_DATA_DIR)) for f in files_to_process]
        old_files_keys = old_manifest.get("files", {}).keys()
        new_files_keys = new_manifest.get("files", {}).keys()
        relative_paths_to_delete = [p for p in old_files_keys if p not in new_files_keys]
        relative_paths_to_delete += files_to_process_relative
        return files_to_process, source_ids_to_delete, relative_paths_to_delete, new_manifest

    def _apply_target_filter(
        self,
        target_files: list[Path],
        files_to_process: list[Path],
        source_ids_to_delete: list[str],
        new_manifest: dict[str, Any],
        old_manifest: dict[str, Any],
    ) -> tuple[list[Path], list[str]]:
        target_paths = {p.resolve() for p in target_files}
        target_names = {p.name for p in target_files}
        old_files = old_manifest.get("files", {})

        filtered_files = [f for f in files_to_process if f.resolve() in target_paths]
        filtered_ids = [
            sid
            for sid in source_ids_to_delete
            if any(info.get("hash") == sid and Path(rel).name in target_names for rel, info in old_files.items())
        ]

        if new_manifest and "files" in new_manifest:
            for rel_path in list(new_manifest["files"].keys()):
                if (RAW_DATA_DIR / rel_path).resolve() not in target_paths:
                    if rel_path in old_files:
                        new_manifest["files"][rel_path] = old_files[rel_path]
                    else:
                        new_manifest["files"].pop(rel_path, None)

        return filtered_files, filtered_ids

    def run_ingestion(
        self,
        force: bool = False,
        parser_type: str | None = None,
        progress_callback=None,
        target_files: list[Path] | None = None,
    ):
        self._update_parser_if_changed(parser_type)
        logger.info(f"데이터 구축 파이프라인을 시작합니다. (전략: {self.parser_type}, 강제 재색인: {force})")

        with self.tracing_logger.start_session(type="ingestion", parser_type=self.parser_type) as session:
            with session.trace_step("scan_and_check_updates") as step:
                all_files = self.ingestion_pipeline.scan_files()
                old_manifest = self.manifest_manager.load_manifest()

                if not all_files and not old_manifest:
                    logger.warning("처리할 파일이 없고 이전 기록도 없습니다.")
                    session.data["status"] = "no_files"
                    return

                files_to_process, source_ids_to_delete, relative_paths_to_delete, new_manifest = (
                    self._prepare_sync_plan(force, all_files, old_manifest)
                )

                if target_files is not None:
                    files_to_process, source_ids_to_delete = self._apply_target_filter(
                        target_files, files_to_process, source_ids_to_delete, new_manifest, old_manifest
                    )

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
                        self.manifest_manager.save_manifest(new_manifest)
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
                self.manifest_manager.save_manifest(new_manifest)

        logger.info("데이터 구축 파이프라인 작업이 완료되었습니다.")

    def update_file_parser(self, file_name: str, new_parser_type: str, progress_callback=None) -> None:
        new_parser_lower = new_parser_type.lower()
        old_manifest = self.manifest_manager.load_manifest()
        old_files = old_manifest.get("files", {})

        target_rel_path = self.manifest_manager.find_relative_path(
            file_name, old_files, self.ingestion_pipeline.scan_files()
        )

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
        self.manifest_manager.save_manifest(old_manifest)

        target_path = RAW_DATA_DIR / target_rel_path
        self.run_ingestion(force=False, progress_callback=progress_callback, target_files=[target_path])

    def update_multiple_file_parsers(self, file_parser_map: dict[str, str], progress_callback=None) -> None:
        old_manifest = self.manifest_manager.load_manifest()
        old_files = old_manifest.get("files", {})

        updated_count = 0
        hashes_to_delete = []
        for file_name, new_parser in file_parser_map.items():
            new_parser_lower = new_parser.lower()
            target_rel_path = self.manifest_manager.find_relative_path(
                file_name, old_files, self.ingestion_pipeline.scan_files()
            )

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
            self.manifest_manager.save_manifest(old_manifest)

            target_paths = []
            for file_name in file_parser_map:
                target_rel_path = self.manifest_manager.find_relative_path(
                    file_name, old_files, self.ingestion_pipeline.scan_files()
                )
                if target_rel_path:
                    target_paths.append(RAW_DATA_DIR / target_rel_path)

            self.run_ingestion(force=False, progress_callback=progress_callback, target_files=target_paths)
