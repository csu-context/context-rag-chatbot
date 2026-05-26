import json
import logging
import pickle
import threading
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.common.config import settings
from src.common.constants import MetadataFields
from src.core.cache import SemanticCache
from src.data.parser import ManualParser
from src.processing.chunking import HierarchicalChunker, create_parent_child_chunks
from src.processing.pdf_parser import DoclingPDFParser
from src.utils.file_utils import generate_file_hash
from src.utils.logger import TracingLogger, setup_global_logging
from src.utils.paths import CACHE_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_directories
from src.vector_db.chroma_manager import ChromaDBManager

# 로깅 설정
logger = logging.getLogger(__name__)


class ParserStrategy(ABC):
    """문서 파싱 전략을 위한 추상 베이스 클래스"""

    @abstractmethod
    def parse(self, file_path: Path) -> list[dict[str, Any]]:
        pass


class ManualParserStrategy(ParserStrategy):
    """기존 ManualParser를 사용하는 전략"""

    def parse(self, file_path: Path) -> list[dict[str, Any]]:
        # ManualParser는 RAW_DATA_DIR 기준 상대 경로를 받음
        relative_path = file_path.relative_to(RAW_DATA_DIR)
        parser = ManualParser(str(relative_path), parser_type="manual")
        return parser.parse()


class MarkdownParserStrategy(ParserStrategy):
    """Markdown 파일을 구조 파괴 없이 읽어오는 전략"""

    def parse(self, file_path: Path) -> list[dict[str, Any]]:
        with open(file_path, encoding="utf-8") as f:
            raw_md_text = f.read()

        if len(raw_md_text.strip()) < 5:
            logger.warning(f"마크다운 파일의 내용이 너무 짧아 건너뜁니다: {file_path.name}")
            return []

        parser_type = settings.PARSER_TYPE.lower()
        base_metadata = {
            MetadataFields.SOURCE_ID: generate_file_hash(file_path, parser_type),
            MetadataFields.SRC_NAME: file_path.name,
            MetadataFields.RELATIVE_PATH: str(file_path.relative_to(RAW_DATA_DIR)),
            MetadataFields.PARSER_TYPE: parser_type,
            MetadataFields.DOC_TYPE: file_path.suffix.lower().replace(".", ""),
            MetadataFields.PG_NUM: 1,
            MetadataFields.CATEGORY: file_path.parent.name if file_path.parent.name != "raw" else "일반",
        }

        return [{"is_raw_markdown": True, "content": raw_md_text, "metadata": base_metadata}]


class DoclingPDFParserStrategy(ParserStrategy):
    """
    IBM Docling 기반 고품질 PDF 파서 전략.
    - 표 구조 34개+ 자동 감지 및 마크다운 변환
    - 한글 OCR (RapidOCR) 지원
    - 레이아웃 구조 보존 (단일 Converter 인스턴스 재사용으로 메모리 효율 확보)
    """

    def __init__(self):
        self.pdf_parser = DoclingPDFParser()  # AI 모델 1회만 로드
        self.manual_parser = ManualParser  # PDF 외 파일 (MD 등) Fallback

    def parse(self, file_path: Path) -> list[dict[str, Any]]:
        if file_path.suffix.lower() == ".pdf":
            # 1. 캐시 파일 경로 설정
            source_id = generate_file_hash(file_path, parser_type="docling")
            cache_file = CACHE_DIR / f"{source_id}_parsed.pkl"

            # 2. 캐시 존재 시 무거운 파싱 생략
            if cache_file.exists():
                logger.info(f"캐시된 Docling 파싱 결과를 로드합니다: {file_path.name}")
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

            logger.info(f"DoclingPDFParser를 사용하여 PDF 파싱: {file_path.name}")
            start_time = time.time()
            parsed = self.pdf_parser.parse(file_path)  # {markdown, tables, page_count, table_count}
            elapsed = time.time() - start_time
            logger.info(f"파싱 완료: {file_path.name} (소요 시간: {elapsed:.2f}초, 표 {parsed['table_count']}개 감지)")

            # 3. Docling 결과를 pipeline 호환 포맷으로 매핑
            #    is_combined=True 플래그로 create_parent_child_chunks 경로를 타도록 지정
            results = [
                {
                    "is_combined": True,
                    "content": parsed["markdown"],
                    "metadata": {
                        MetadataFields.SOURCE_ID: source_id,
                        MetadataFields.SRC_NAME: file_path.name,
                        MetadataFields.RELATIVE_PATH: str(file_path.relative_to(RAW_DATA_DIR)),
                        MetadataFields.PARSER_TYPE: "docling",
                        MetadataFields.PG_NUM: 1,
                        MetadataFields.DOC_TYPE: "pdf",
                        MetadataFields.CATEGORY: file_path.parent.name,
                        # Docling 전용 메타데이터: 표 감지 정보
                        "table_count": parsed["table_count"],
                        "page_count": parsed["page_count"],
                        "has_table": parsed["table_count"] > 0,
                    },
                }
            ]

            # 4. 캐시 저장
            with open(cache_file, "wb") as f:
                pickle.dump(results, f)

            return results
        else:
            # PDF가 아닌 경우 ManualParser로 Fallback
            relative_path = file_path.relative_to(RAW_DATA_DIR)
            parser = self.manual_parser(str(relative_path), parser_type="docling")
            return parser.parse()


class IngestionPipeline:
    """데이터 전처리 및 인덱싱을 담당하는 클래스"""

    def __init__(
        self,
        strategy: ParserStrategy,
        raw_dir: Path = RAW_DATA_DIR,
        processed_dir: Path = PROCESSED_DATA_DIR,
        collection_name: str = "rag_collection",
    ):
        self.strategy = strategy
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.chunker = HierarchicalChunker()
        # ChromaDBManager 인스턴스를 초기화 시 한 번만 생성하여 재사용
        self.db_manager = ChromaDBManager(collection_name=collection_name)
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

    def process_and_chunk(  # noqa: C901
        self, files: list[Path], progress_callback=None, file_parser_types: dict[str, str] | None = None
    ) -> list[dict[str, Any]]:
        all_hierarchical_data = []
        total_files = len(files)
        for idx, file_path in enumerate(files):
            if progress_callback:
                try:
                    progress_callback(idx, total_files, file_path.name)
                except Exception as cb_e:
                    logger.error(f"진행 상황 콜백 호출 실패: {cb_e}")
            try:
                # 1. 파일 확장자에 따른 전략 동적 선택 및 파싱
                if file_path.suffix.lower() == ".pdf":
                    import unicodedata

                    rel_path = unicodedata.normalize("NFC", str(file_path.relative_to(RAW_DATA_DIR)))
                    normalized_parser_types = {
                        unicodedata.normalize("NFC", k): v for k, v in (file_parser_types or {}).items()
                    }
                    file_parser = normalized_parser_types.get(rel_path, "manual").lower()
                    logger.warning(
                        f"=== 디버깅 파서 선택 ===\n"
                        f"파일명: {file_path.name}\n"
                        f"rel_path (NFC): {rel_path}\n"
                        f"결정된 file_parser: {file_parser}\n"
                        f"전달된 file_parser_types: {file_parser_types}\n"
                        f"정규화된 parser_types: {normalized_parser_types}\n"
                        f"========================="
                    )
                    if file_parser == "docling":
                        try:
                            import docling  # noqa: F401

                            active_strategy = DoclingPDFParserStrategy()
                        except ImportError:
                            logger.error("docling 라이브러리가 없어 manual 전략으로 대체합니다.")
                            active_strategy = ManualParserStrategy()
                    else:
                        active_strategy = ManualParserStrategy()
                else:
                    active_strategy = MarkdownParserStrategy()

                sections = active_strategy.parse(file_path)

                if not sections:
                    continue

                # 2. 계층적 청킹 (결과 타입별 분기 처리)
                if sections[0].get("is_raw_markdown"):
                    file_chunks = create_parent_child_chunks(sections[0]["content"], sections[0]["metadata"])
                    all_hierarchical_data.extend(file_chunks)
                elif sections[0].get("is_combined"):
                    for sec in sections:
                        file_chunks = create_parent_child_chunks(sec["content"], sec["metadata"])
                        all_hierarchical_data.extend(file_chunks)
                else:
                    # 기존 ManualParser PDF 처리 방식 (Fallback 호환성)
                    for sec in sections:
                        parent_id = str(uuid.uuid4())
                        meta_for_children = sec["metadata"].copy()
                        sec_title = f"{sec.get('chapter', '기본 섹션')} > {sec.get('article', '기본 섹션')}"
                        meta_for_children[MetadataFields.SEC_TITLE] = sec_title
                        meta_for_children[MetadataFields.HEADER_PATH] = sec_title

                        children = self.chunker.split_into_children(sec["content"], parent_id, meta_for_children)

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
                            }
                            all_hierarchical_data.append(
                                {
                                    "parent_id": parent_id,
                                    "parent_text": sec["content"],
                                    "metadata": parent_metadata,
                                    "children": children,
                                }
                            )

            except Exception as e:
                logger.error(f"파일 처리 실패: {file_path.name} - {e!s}")

        if progress_callback and total_files > 0:
            try:
                progress_callback(total_files, total_files, "모든 파일 처리 완료")
            except Exception as cb_e:
                logger.error(f"진행 상황 콜백 호출 실패: {cb_e}")

        return all_hierarchical_data

    def save_processed_data(self, data: list[dict[str, Any]]) -> list[Path]:
        """
        전처리된 데이터를 source_id별 개별 JSON 파일로 저장합니다.
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
            save_path = self.processed_dir / f"{sid}.json"
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(sid_data, f, ensure_ascii=False, indent=2)
            saved_paths.append(save_path)
            logger.info(f"   - 전처리 결과 저장: {save_path.name}")

        return saved_paths

    def _prepare_cleanup_targets(
        self, source_ids_to_delete: list[str] | None, filenames_to_delete: list[str] | None
    ) -> tuple[list[str], list[str]]:
        import unicodedata
        from pathlib import Path

        valid_ids = list(set([sid for sid in (source_ids_to_delete or []) if sid]))

        target_filenames = []
        if filenames_to_delete:
            for fname in filenames_to_delete:
                if not fname:
                    continue
                name_only = Path(fname).name
                nfc_n = unicodedata.normalize("NFC", name_only)
                nfd_n = unicodedata.normalize("NFD", name_only)
                target_filenames.extend([nfc_n, nfd_n])

        target_filenames = list(set(target_filenames))

        if target_filenames:
            logger.info(f"파일명 기반 클린업 대상 확인: {filenames_to_delete}")
            try:
                results = self.db_manager.collection.get(
                    where={"src_name": {"$in": target_filenames}}, include=["metadatas"]
                )
                if results and results.get("metadatas"):
                    for meta in results["metadatas"]:
                        if meta and "source_id" in meta:
                            valid_ids.append(meta["source_id"])
            except Exception as e:
                logger.error(f"파일명 기반 source_id 조회 중 오류 발생: {e}")

        return list(set(valid_ids)), target_filenames

    def cleanup_db(
        self,
        source_ids_to_delete: list[str] | None = None,
        relative_paths_to_delete: list[str] | None = None,
        filenames_to_delete: list[str] | None = None
    ):
        """
        DB, 캐시 및 가공된 JSON 파일에서 삭제된 데이터를 제거합니다.
        (BM25Manager는 processed_dir의 모든 json을 읽으므로 물리적 파일 삭제가 곧 정합성임)
        """
        valid_ids, target_filenames = self._prepare_cleanup_targets(source_ids_to_delete, filenames_to_delete)

        if not valid_ids and not relative_paths_to_delete and not target_filenames:
            return

        logger.info("데이터 정합성 검증 및 클린업을 수행합니다.")

        # 1. 벡터 DB 데이터 삭제
        if target_filenames:
            try:
                self.db_manager.delete_documents(where={"src_name": {"$in": target_filenames}})
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
        removed = False
        # 파싱 캐시 삭제
        cache_file = CACHE_DIR / f"{sid}_parsed.pkl"
        if cache_file.exists():
            cache_file.unlink()

        # 가공된 JSON 삭제 (BM25 정합성 핵심)
        json_file = self.processed_dir / f"{sid}.json"
        if json_file.exists():
            json_file.unlink()
            removed = True
        return removed

    def _remove_files_by_rel_path(self, relative_paths_to_delete: list[str]) -> int:
        """상대 경로 목록에 해당하는 물리 파일들을 찾아 삭제"""
        deleted_count = 0
        for json_file in list(self.processed_dir.glob("*.json")):
            if json_file.name == "manifest.json":
                continue
            try:
                data = None
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)

                if not (data and isinstance(data, list) and len(data) > 0):
                    continue

                meta = data[0].get("metadata", {})
                if meta.get(MetadataFields.RELATIVE_PATH) in relative_paths_to_delete:
                    sid = json_file.stem
                    # 캐시 삭제
                    cache_file = CACHE_DIR / f"{sid}_parsed.pkl"
                    if cache_file.exists():
                        cache_file.unlink()
                    # JSON 삭제
                    if json_file.exists():
                        json_file.unlink()
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
            self.ingestion_pipeline = IngestionPipeline(self.strategy)
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
            import unicodedata

            rel_path = unicodedata.normalize("NFC", str(f.relative_to(RAW_DATA_DIR)))
            old_info = old_files.get(rel_path, {})
            if not old_info:
                # NFD-NFC 매치 백업
                for k, v in old_files.items():
                    if unicodedata.normalize("NFC", k) == rel_path:
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

    def _process_changes(
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

                    for f in self.ingestion_pipeline.processed_dir.glob("*.json"):
                        try:
                            f.unlink()
                        except Exception as e:
                            logger.error(f"JSON 파일 삭제 실패 {f}: {e}")

                    for f in CACHE_DIR.glob("*_parsed.pkl"):
                        try:
                            f.unlink()
                        except Exception as e:
                            logger.error(f"캐시 파일 삭제 실패 {f}: {e}")

                    files_to_process = all_files
                    source_ids_to_delete = []  # 이미 물리적으로 모두 삭제했으므로 부분 삭제 프로세스는 건너뜀
                    relative_paths_to_delete = None

                    new_files = {}
                    import unicodedata

                    for f in all_files:
                        rel_path = unicodedata.normalize("NFC", str(f.relative_to(RAW_DATA_DIR)))
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

        import unicodedata

        normalized_file_name = unicodedata.normalize("NFC", file_name)
        target_rel_path = None
        for rel_path in old_files:
            normalized_rel_path = unicodedata.normalize("NFC", rel_path)
            is_match = (
                Path(normalized_rel_path).name == normalized_file_name or normalized_rel_path == normalized_file_name
            )
            if is_match:
                target_rel_path = rel_path
                break

        if not target_rel_path:
            for f in self.ingestion_pipeline.scan_files():
                normalized_f_name = unicodedata.normalize("NFC", f.name)
                if normalized_f_name == normalized_file_name:
                    target_rel_path = unicodedata.normalize("NFC", str(f.relative_to(RAW_DATA_DIR)))
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
            import unicodedata

            normalized_file_name = unicodedata.normalize("NFC", file_name)

            target_rel_path = None
            for rel_path in old_files:
                normalized_rel_path = unicodedata.normalize("NFC", rel_path)
                is_match = (
                    Path(normalized_rel_path).name == normalized_file_name
                    or normalized_rel_path == normalized_file_name
                )
                if is_match:
                    target_rel_path = rel_path
                    break

            if not target_rel_path:
                for f in self.ingestion_pipeline.scan_files():
                    normalized_f_name = unicodedata.normalize("NFC", f.name)
                    if normalized_f_name == normalized_file_name:
                        target_rel_path = unicodedata.normalize("NFC", str(f.relative_to(RAW_DATA_DIR)))
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


# 하위 호환성을 위한 기존 클래스 래핑
class PreprocessingPipeline:
    def __init__(self, *args, **kwargs):
        self.orchestrator = PipelineOrchestrator()

    def run(self, *args, **kwargs):
        return self.orchestrator.run_ingestion()


if __name__ == "__main__":
    setup_global_logging()
    orchestrator = PipelineOrchestrator()
    orchestrator.run_ingestion()
