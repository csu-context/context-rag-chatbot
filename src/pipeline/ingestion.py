import logging
import os
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.common.constants import MetadataFields
from src.core.chains import invalidate_source_json_cache
from src.core.storage import StorageManager
from src.pipeline.strategies import (
    DoclingPDFParserStrategy,
    ManualParserStrategy,
    MarkdownParserStrategy,
    ParserStrategy,
)
from src.processing.chunking import HierarchicalChunker, create_parent_child_chunks
from src.utils.paths import CACHE_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_directories
from src.utils.unicode import normalize_path_to_nfc, normalize_to_nfc, normalize_to_nfd
from src.vector_db.chroma_manager import ChromaDBManager

logger = logging.getLogger(__name__)


def _safe_invoke_progress(callback, current: int, total: int, name: str) -> None:
    if callback:
        try:
            callback(current, total, name)
        except Exception as cb_e:
            logger.error(f"진행 상황 콜백 호출 실패: {cb_e}")


def _get_parser_strategy_for_file(file_path: Path, file_parser_types: dict[str, str] | None) -> ParserStrategy:
    """파일 경로와 매니페스트 설정을 기반으로 적절한 파서 전략을 반환합니다."""
    if file_path.suffix.lower() != ".pdf":
        return MarkdownParserStrategy()

    from src.utils.paths import RAW_DATA_DIR
    from src.utils.unicode import normalize_to_nfc

    rel_path = normalize_path_to_nfc(file_path.relative_to(RAW_DATA_DIR))
    normalized_parser_types = {normalize_to_nfc(k): v for k, v in (file_parser_types or {}).items()}
    file_parser = normalized_parser_types.get(rel_path, "manual").lower()

    if file_parser == "docling":
        try:
            import docling  # noqa: F401

            return DoclingPDFParserStrategy()
        except ImportError:
            logger.error("docling 라이브러리가 없어 manual 전략으로 대체합니다.")
            return ManualParserStrategy()

    return ManualParserStrategy()


def _chunk_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """파싱된 섹션들을 계층적 청크 구조로 변환합니다."""
    if not sections:
        return []

    file_chunks_accum = []

    if sections[0].get("is_raw_markdown"):
        return create_parent_child_chunks(sections[0]["content"], sections[0]["metadata"])

    if sections[0].get("is_combined"):
        for sec in sections:
            file_chunks_accum.extend(create_parent_child_chunks(sec["content"], sec["metadata"]))
        return file_chunks_accum

    # 기존 ManualParser PDF 처리 방식 (Fallback 호환성)
    chunker = HierarchicalChunker()
    for sec in sections:
        parent_id = str(uuid.uuid4())
        meta_for_children = sec["metadata"].copy()
        sec_title = f"{sec.get('chapter', '기본 섹션')} > {sec.get('article', '기본 섹션')}"
        meta_for_children[MetadataFields.SEC_TITLE] = sec_title
        meta_for_children[MetadataFields.HEADER_PATH] = sec_title

        children = chunker.split_into_children(sec["content"], parent_id, meta_for_children)

        if children:
            parent_metadata = {
                MetadataFields.SOURCE_ID: sec["metadata"].get(MetadataFields.SOURCE_ID, "UNKNOWN"),
                MetadataFields.SRC_NAME: sec["metadata"].get(MetadataFields.SRC_NAME, "UNKNOWN"),
                MetadataFields.DOC_TYPE: sec["metadata"].get(MetadataFields.DOC_TYPE, "pdf"),
                MetadataFields.PG_NUM: sec["metadata"].get(MetadataFields.PG_NUM, 1),
                MetadataFields.SEC_TITLE: sec_title,
                MetadataFields.CHUNK_ID: parent_id,
                MetadataFields.PARENT_ID: None,
                MetadataFields.HEADER_PATH: sec_title,
                MetadataFields.IS_TABLE: False,
                MetadataFields.RELATIVE_PATH: sec["metadata"].get(MetadataFields.RELATIVE_PATH, "UNKNOWN"),
            }
            file_chunks_accum.append(
                {
                    "parent_id": parent_id,
                    "parent_text": sec["content"],
                    "metadata": parent_metadata,
                    "children": children,
                }
            )
    return file_chunks_accum


def _process_single_file_helper(
    file_path: Path,
    file_parser_types: dict[str, str] | None,
    processed_dir: Path,
    cache_dir: Path,
) -> list[dict[str, Any]]:
    try:
        storage_manager = StorageManager(processed_dir, cache_dir)
        active_strategy = _get_parser_strategy_for_file(file_path, file_parser_types)
        sections = active_strategy.parse(file_path, storage_manager=storage_manager)
        return _chunk_sections(sections)
    except Exception as e:
        logger.error(f"파일 처리 실패: {file_path.name} - {e!s}")
        return []


class IngestionPipeline:
    """데이터 전처리 및 인덱싱을 담당하는 클래스"""

    def __init__(
        self,
        strategy: ParserStrategy,
        raw_dir: Path = RAW_DATA_DIR,
        processed_dir: Path = PROCESSED_DATA_DIR,
        collection_name: str = "rag_collection",
        storage_manager: StorageManager = None,
    ):
        self.strategy = strategy
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.chunker = HierarchicalChunker()
        self.db_manager = ChromaDBManager(collection_name=collection_name)
        # StorageManager 연동
        self.storage_manager = storage_manager if storage_manager else StorageManager(processed_dir, CACHE_DIR)
        ensure_directories()

    def scan_files(self, supported_exts: list[str] | None = None) -> list[Path]:
        if supported_exts is None:
            supported_exts = [".pdf", ".md", ".markdown"]
        files = []
        for ext in supported_exts:
            # 모든 하위 디렉토리를 포함하여 검색
            all_files = list(self.raw_dir.glob(f"**/*{ext}"))
            # Mac 숨김 파일(._ 로 시작) 및 시스템 파일 필터링
            filtered_files = [f for f in all_files if not f.name.startswith("._") and f.name != ".DS_Store"]
            files.extend(filtered_files)
        return files

    def process_and_chunk(
        self, files: list[Path], progress_callback=None, file_parser_types: dict[str, str] | None = None
    ) -> list[dict[str, Any]]:
        all_hierarchical_data = []
        total_files = len(files)
        if total_files == 0:
            return []

        # 단일 파일인 경우 불필요한 프로세스 풀 생성 오버헤드 방지
        if total_files == 1:
            _safe_invoke_progress(progress_callback, 0, total_files, files[0].name)
            res = _process_single_file_helper(
                files[0],
                file_parser_types,
                self.storage_manager.processed_dir,
                self.storage_manager.cache_dir,
            )
            all_hierarchical_data.extend(res)
            _safe_invoke_progress(progress_callback, total_files, total_files, "모든 파일 처리 완료")
            return all_hierarchical_data

        # 하이브리드 처리: MD 파일은 프로세스 풀 병렬, PDF 파일은 OOM 방지를 위해 순차 처리
        pdf_files = [f for f in files if f.suffix.lower() == ".pdf"]
        non_pdf_files = [f for f in files if f.suffix.lower() != ".pdf"]
        completed_count = 0

        # Phase 1: 비-PDF 파일 병렬 처리
        if non_pdf_files:
            max_workers = min(len(non_pdf_files), os.cpu_count() or 4)
            logger.info(f"비-PDF 파일 병렬 파싱 활성화 ({len(non_pdf_files)}개 파일, Workers: {max_workers})")

            futures_map = {}
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for file_path in non_pdf_files:
                    future = executor.submit(
                        _process_single_file_helper,
                        file_path,
                        file_parser_types,
                        self.storage_manager.processed_dir,
                        self.storage_manager.cache_dir,
                    )
                    futures_map[future] = file_path

                for future in as_completed(futures_map):
                    file_path = futures_map[future]
                    completed_count += 1
                    try:
                        res = future.result()
                        all_hierarchical_data.extend(res)
                    except Exception as e:
                        logger.error(f"병렬 파일 처리 실패: {file_path.name} - {e}")

                    _safe_invoke_progress(progress_callback, completed_count, total_files, file_path.name)

        # Phase 2: PDF 파일 순차 처리 (Docling heavy AI 모델 OOM 방지)
        if pdf_files:
            logger.info(f"PDF 파일 순차 처리 시작 ({len(pdf_files)}개 파일, 메모리 보호 모드)")
            for file_path in pdf_files:
                try:
                    res = _process_single_file_helper(
                        file_path,
                        file_parser_types,
                        self.storage_manager.processed_dir,
                        self.storage_manager.cache_dir,
                    )
                    all_hierarchical_data.extend(res)
                except Exception as e:
                    logger.error(f"PDF 파일 처리 실패: {file_path.name} - {e}")

                completed_count += 1
                _safe_invoke_progress(progress_callback, completed_count, total_files, file_path.name)

        return all_hierarchical_data

    def save_processed_data(self, data: list[dict[str, Any]]) -> list[Path]:
        """전처리된 데이터를 source_id별 개별 JSON 파일로 저장합니다.

        (BM25 인덱스와 정합성 유지를 위해 개별 파일 관리가 필수적임)
        """
        saved_paths = []
        # 데이터를 source_id별로 그룹화
        grouped_data = {}
        for item in data:
            sid = item["metadata"].get(MetadataFields.SOURCE_ID, "unknown")
            if sid not in grouped_data:
                grouped_data[sid] = []
            grouped_data[sid].append(item)

        for sid, sid_data in grouped_data.items():
            save_path = self.storage_manager.save_processed_data(sid, sid_data)
            saved_paths.append(save_path)
            logger.info(f"   - 전처리 결과 저장: {save_path.name}")

        # JSON 파일이 갱신되었으므로 RAG 파이프라인의 LRU 캐시 무효화
        invalidate_source_json_cache()
        return saved_paths

    def _prepare_cleanup_targets(
        self, source_ids_to_delete: list[str] | None, filenames_to_delete: list[str] | None
    ) -> tuple[list[str], list[str]]:
        valid_ids = list({sid for sid in (source_ids_to_delete or []) if sid})

        target_filenames = []
        if filenames_to_delete:
            for fname in filenames_to_delete:
                if not fname:
                    continue
                name_only = Path(fname).name
                nfc_n = normalize_to_nfc(name_only)
                nfd_n = normalize_to_nfd(name_only)
                target_filenames.extend([nfc_n, nfd_n])

        target_filenames = list(set(target_filenames))

        if target_filenames:
            logger.info(f"파일명 기반 클린업 대상 확인: {filenames_to_delete}")
            try:
                results = self.db_manager.collection.get(
                    where={MetadataFields.SRC_NAME: {"$in": target_filenames}}, include=["metadatas"]
                )
                if results and results.get("metadatas"):
                    for meta in results["metadatas"]:
                        if meta and MetadataFields.SOURCE_ID in meta:
                            valid_ids.append(meta[MetadataFields.SOURCE_ID])
            except Exception as e:
                logger.error(f"파일명 기반 source_id 조회 중 오류 발생: {e}")

        return list(set(valid_ids)), target_filenames

    def cleanup_db(
        self,
        source_ids_to_delete: list[str] | None = None,
        relative_paths_to_delete: list[str] | None = None,
        filenames_to_delete: list[str] | None = None,
    ):
        """DB, 캐시 및 가공된 JSON 파일에서 삭제된 데이터를 제거합니다.

        (BM25Manager는 processed_dir의 모든 json을 읽으므로 물리적 파일 삭제가 곧 정합성임)
        """
        valid_ids, target_filenames = self._prepare_cleanup_targets(source_ids_to_delete, filenames_to_delete)

        if not valid_ids and not relative_paths_to_delete and not target_filenames:
            return

        logger.info("데이터 정합성 검증 및 클린업을 수행합니다.")

        # 1. 벡터 DB 데이터 삭제
        if target_filenames:
            try:
                self.db_manager.delete_documents(where={MetadataFields.SRC_NAME: {"$in": target_filenames}})
                logger.info(f"   - ChromaDB 파일명 기준 {len(target_filenames)}개 파일 삭제 완료")
            except Exception as e:
                logger.error(f"ChromaDB 파일명 기준 삭제 실패: {e}")

        self._delete_from_vector_db(valid_ids, relative_paths_to_delete)

        # 2. 물리적 캐시 및 JSON 파일 삭제
        deleted_count = self._delete_physical_files(valid_ids, relative_paths_to_delete)

        if deleted_count > 0:
            logger.info(f"물리적 데이터 파일 {deleted_count}개 삭제 완료")

        logger.info("데이터 정합성 검증 및 클린업 작업이 완료되었습니다.")

    def _delete_from_vector_db(self, source_ids_to_delete: list[str], relative_paths_to_delete: list[str] | None):
        """벡터 DB에서 데이터 삭제"""
        # 상대 경로 기반 삭제 (원자적 정리 강화)
        if relative_paths_to_delete:
            for rel_path in relative_paths_to_delete:
                logger.info(f"   - 상대 경로 기준 삭제: {rel_path}")
                self.db_manager.delete_documents(where={MetadataFields.RELATIVE_PATH: rel_path})

                # 윈도우/리눅스 경로 구분자 불일치 대응 (메타데이터 저장 시 경로가 다를 수 있음)
                alt_rel_path = rel_path.replace("\\", "/") if "\\" in rel_path else rel_path.replace("/", "\\")
                if alt_rel_path != rel_path:
                    self.db_manager.delete_documents(where={MetadataFields.RELATIVE_PATH: alt_rel_path})

        # Source ID 기반 삭제 (하위 호환성)
        if source_ids_to_delete:
            self.db_manager.delete_documents(where={MetadataFields.SOURCE_ID: {"$in": source_ids_to_delete}})
            logger.info(f"   - ChromaDB {len(source_ids_to_delete)}개 Source ID 삭제 완료")

    def _delete_physical_files(
        self, source_ids_to_delete: list[str], relative_paths_to_delete: list[str] | None
    ) -> int:
        """물리적 캐시 및 JSON 파일 삭제"""
        deleted_count = 0

        # a. Source ID 기반 삭제
        for sid in source_ids_to_delete:
            if self._remove_sid_files(sid):
                deleted_count += 1

        # b. 상대 경로 기반 물리 파일 삭제 (파서 변경 등 중복 방지)
        if relative_paths_to_delete:
            deleted_count += self._remove_files_by_rel_path(relative_paths_to_delete)

        return deleted_count

    def _remove_sid_files(self, sid: str) -> bool:
        """특정 Source ID와 관련된 물리 파일 삭제"""
        return self.storage_manager.delete_processed_and_cache(sid)

    def _remove_files_by_rel_path(self, relative_paths_to_delete: list[str]) -> int:
        """상대 경로 목록에 해당하는 물리 파일들을 찾아 삭제"""
        deleted_count = 0
        for json_file in self.storage_manager.scan_processed_files():
            if json_file.name == "manifest.json":
                continue
            try:
                data = self.storage_manager.load_processed_file(json_file)

                if not (data and isinstance(data, list) and len(data) > 0):
                    continue

                meta = data[0].get("metadata", {})
                if meta.get(MetadataFields.RELATIVE_PATH) in relative_paths_to_delete:
                    sid = json_file.stem
                    if self.storage_manager.delete_processed_and_cache(sid):
                        deleted_count += 1
            except Exception as e:
                logger.warning(f"파일 스캔 중 오류 ({json_file.name}): {e}")
        return deleted_count

    def upsert_to_db(self, data: list[dict[str, Any]]):
        ids, docs, metas = [], [], []
        for parent in data:
            for child in parent["children"]:
                ids.append(child["chunk_id"])
                docs.append(child["text"])
                metas.append(child["metadata"])

        if ids:
            logger.info(f"ChromaDB 업서트 시작 ({len(ids)}개 청크)...")
            self.db_manager.upsert_documents(ids=ids, documents=docs, metadatas=metas)
            logger.info("ChromaDB 업서트 완료!")
