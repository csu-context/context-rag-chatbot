import importlib.util
import logging
import os
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.common.constants import MetadataFields, SupportedFormats
from src.core.chains import invalidate_source_json_cache
from src.core.storage import StorageManager
from src.pipeline.strategies import (
    DoclingPDFParserStrategy,
    HwpParserStrategy,
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
    ext = file_path.suffix.lower()

    if ext in (".hwp", ".hwpx"):
        return HwpParserStrategy()

    if ext != ".pdf":
        return MarkdownParserStrategy()

    from src.utils.paths import RAW_DATA_DIR
    from src.utils.unicode import normalize_to_nfc

    rel_path = normalize_path_to_nfc(file_path.relative_to(RAW_DATA_DIR))
    normalized_parser_types = {normalize_to_nfc(k): v for k, v in (file_parser_types or {}).items()}
    file_parser = normalized_parser_types.get(rel_path, "manual").lower()

    if file_parser == "docling":
        if importlib.util.find_spec("docling") is not None:
            return DoclingPDFParserStrategy()
        logger.error("docling 라이브러리가 없어 manual 전략으로 대체합니다.")
        return ManualParserStrategy()

    return ManualParserStrategy()


def _chunk_raw_markdown(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return create_parent_child_chunks(sections[0]["content"], sections[0]["metadata"])


def _chunk_combined(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunker = HierarchicalChunker()
    result = []
    for sec in sections:
        result.extend(chunker.chunk(sec["content"], sec["metadata"]))
    return result


def _split_table_into_row_chunks(content: str, parent_id: str, base_meta: dict) -> list[dict[str, Any]]:
    """표를 행(row) 단위 청크로 분할. 각 청크에 표 제목 컨텍스트 + 헤더 행 보존.

    24행*10열 대형 표도 각 데이터 행이 독립 청크(200~400자)가 되어
    LLM 컨텍스트 한도 내에서 검색·답변 가능하게 한다.
    """
    # 표 제목(비-마크다운 텍스트)과 표 본체 분리
    parts = content.split("\n\n", 1)
    if len(parts) == 2 and "|" in parts[1]:
        title_ctx, table_md = parts[0].strip(), parts[1].strip()
    else:
        title_ctx, table_md = "", content.strip()

    lines = [ln for ln in table_md.split("\n") if ln.strip()]
    if len(lines) < 3:
        return []

    header, separator = lines[0], lines[1]
    data_rows = lines[2:]

    children = []
    for i, row in enumerate(data_rows):
        if not row.strip() or set(row.strip()) <= {"|", "-", " ", ":"}:
            continue
        row_content_parts = [header, separator, row]
        if title_ctx:
            row_content_parts = [title_ctx, "", *row_content_parts]
        row_text = "\n".join(row_content_parts).strip()
        child_id = f"{parent_id}_r{i}"
        child_meta = {
            **base_meta,
            MetadataFields.CHUNK_ID: child_id,
            MetadataFields.PARENT_ID: parent_id,
            MetadataFields.IS_TABLE: True,
        }
        children.append({"chunk_id": child_id, "metadata": child_meta, "text": row_text})

    return children


def _chunk_manual_pdf(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunker = HierarchicalChunker()
    result = []
    for sec in sections:
        parent_id = str(uuid.uuid4())
        meta_for_children = sec["metadata"].copy()
        sec_title = f"{sec.get('chapter', '기본 섹션')} > {sec.get('article', '기본 섹션')}"
        meta_for_children[MetadataFields.SEC_TITLE] = sec_title
        meta_for_children[MetadataFields.HEADER_PATH] = sec_title

        if sec["metadata"].get(MetadataFields.IS_TABLE, False):
            children = _split_table_into_row_chunks(sec["content"], parent_id, meta_for_children)
        else:
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
                MetadataFields.IS_TABLE: sec["metadata"].get(MetadataFields.IS_TABLE, False),
                MetadataFields.RELATIVE_PATH: sec["metadata"].get(MetadataFields.RELATIVE_PATH, "UNKNOWN"),
            }
            result.append(
                {
                    "parent_id": parent_id,
                    "parent_text": sec["content"],
                    "metadata": parent_metadata,
                    "children": children,
                }
            )
    return result


def _chunk_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not sections:
        return []
    first = sections[0]
    if first.get("is_raw_markdown"):
        return _chunk_raw_markdown(sections)
    if first.get("is_combined"):
        return _chunk_combined(sections)
    return _chunk_manual_pdf(sections)


def _process_single_file_helper(
    file_path: Path,
    file_parser_types: dict[str, str] | None,
    processed_dir: Path,
    cache_dir: Path,
) -> list[dict[str, Any]]:
    storage_manager = StorageManager(processed_dir, cache_dir)
    active_strategy = _get_parser_strategy_for_file(file_path, file_parser_types)
    sections = active_strategy.parse(file_path, storage_manager=storage_manager)
    return _chunk_sections(sections)


class IngestionPipeline:
    """데이터 전처리 및 인덱싱을 담당하는 클래스"""

    def __init__(
        self,
        strategy: ParserStrategy,
        raw_dir: Path = RAW_DATA_DIR,
        processed_dir: Path = PROCESSED_DATA_DIR,
        collection_name: str = "rag_collection",
        storage_manager: StorageManager | None = None,
    ):
        self.strategy = strategy
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.db_manager = ChromaDBManager(collection_name=collection_name)
        # StorageManager 연동
        self.storage_manager = storage_manager if storage_manager else StorageManager(processed_dir, CACHE_DIR)
        ensure_directories()

    def scan_files(self, supported_exts: list[str] | None = None) -> list[Path]:
        if supported_exts is None:
            supported_exts = SupportedFormats.EXTENSIONS
        files = []
        for ext in supported_exts:
            # 모든 하위 디렉토리를 포함하여 검색
            all_files = list(self.raw_dir.glob(f"**/*{ext}"))
            # Mac 숨김 파일(._ 로 시작) 및 시스템 파일 필터링
            filtered_files = [f for f in all_files if not f.name.startswith("._") and f.name != ".DS_Store"]
            files.extend(filtered_files)
        return files

    def _process_non_pdf_parallel(
        self, files: list[Path], file_parser_types: dict | None, total: int, offset: int, progress_callback
    ) -> tuple[list, int]:
        data, completed = [], offset
        max_workers = min(len(files), os.cpu_count() or 4)
        logger.info(f"비-PDF 파일 병렬 파싱 활성화 ({len(files)}개 파일, Workers: {max_workers})")
        futures_map = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for f in files:
                futures_map[
                    executor.submit(
                        _process_single_file_helper,
                        f,
                        file_parser_types,
                        self.storage_manager.processed_dir,
                        self.storage_manager.cache_dir,
                    )
                ] = f
            for future in as_completed(futures_map):
                file_path = futures_map[future]
                completed += 1
                try:
                    data.extend(future.result())
                except (TypeError, AttributeError) as e:
                    # 직렬화 관련 예외(PicklingError 포함) 시에만 안전하게 메인 프로세스에서 순차 재시도
                    logger.warning(f"병렬 직렬화 오류 감지: {file_path.name} ({e}). 순차 재시도를 수행합니다.")
                    try:
                        res = _process_single_file_helper(
                            file_path,
                            file_parser_types,
                            self.storage_manager.processed_dir,
                            self.storage_manager.cache_dir,
                        )
                        data.extend(res)
                    except Exception as seq_e:
                        logger.error(f"순차 재시도 처리 실패: {file_path.name} - {seq_e!s}")
                except Exception as e:
                    # 워커 OOM(BrokenProcessPool) 감지 시 즉시 중단하여 메인 프로세스 보호
                    from concurrent.futures.process import BrokenProcessPool
                    if isinstance(e, BrokenProcessPool):
                        logger.critical(
                            f"프로세스 풀이 파괴되었습니다(OOM 의심): {file_path.name} - {e!s}. "
                            "메인 프로세스 보호를 위해 즉시 중단합니다."
                        )
                        raise RuntimeError(
                            f"병렬 파싱 중 워커 OOM 발생으로 프로세스 풀이 손상되었습니다. "
                            f"파일: {file_path.name}"
                        ) from e
                    # 개별 파싱 오류 등은 재시도 없이 에러 처리 후 계속 진행
                    logger.error(f"병렬 파일 처리 실패 (재시도 안 함): {file_path.name} - {e!s}")
                _safe_invoke_progress(progress_callback, completed, total, file_path.name)
        return data, completed

    def _process_pdf_sequential(
        self, files: list[Path], file_parser_types: dict | None, total: int, offset: int, progress_callback
    ) -> tuple[list, int]:
        data, completed = [], offset
        logger.info(f"PDF 파일 순차 처리 시작 ({len(files)}개 파일, 메모리 보호 모드)")
        for file_path in files:
            try:
                data.extend(
                    _process_single_file_helper(
                        file_path,
                        file_parser_types,
                        self.storage_manager.processed_dir,
                        self.storage_manager.cache_dir,
                    )
                )
            except Exception as e:
                logger.error(f"PDF 파일 처리 실패: {file_path.name} - {e}")
            completed += 1
            _safe_invoke_progress(progress_callback, completed, total, file_path.name)
        return data, completed

    def process_and_chunk(
        self, files: list[Path], progress_callback=None, file_parser_types: dict[str, str] | None = None
    ) -> list[dict[str, Any]]:
        total = len(files)
        if total == 0:
            return []

        if total == 1:
            _safe_invoke_progress(progress_callback, 0, total, files[0].name)
            result = _process_single_file_helper(
                files[0],
                file_parser_types,
                self.storage_manager.processed_dir,
                self.storage_manager.cache_dir,
            )
            _safe_invoke_progress(progress_callback, total, total, "모든 파일 처리 완료")
            return result

        pdf_files = [f for f in files if f.suffix.lower() == ".pdf"]
        non_pdf_files = [f for f in files if f.suffix.lower() != ".pdf"]
        all_data, completed = [], 0

        if non_pdf_files:
            chunk, completed = self._process_non_pdf_parallel(
                non_pdf_files, file_parser_types, total, completed, progress_callback
            )
            all_data.extend(chunk)

        if pdf_files:
            chunk, completed = self._process_pdf_sequential(
                pdf_files, file_parser_types, total, completed, progress_callback
            )
            all_data.extend(chunk)

        return all_data

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

    def _delete_from_vector_db(
        self, source_ids_to_delete: list[str], relative_paths_to_delete: list[str] | None
    ) -> None:
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

    def upsert_to_db(self, data: list[dict[str, Any]]) -> None:
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
