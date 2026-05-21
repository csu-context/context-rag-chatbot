import json
import logging
import os
import pickle
import threading
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from tqdm import tqdm

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

        parser_type = os.getenv("PARSER_TYPE", "manual").lower()
        base_metadata = {
            MetadataFields.SOURCE_ID: generate_file_hash(file_path, parser_type),
            MetadataFields.SRC_NAME: file_path.name,
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
                        MetadataFields.PG_NUM: 1,
                        MetadataFields.DOC_TYPE: "pdf",
                        MetadataFields.CATEGORY: file_path.parent.name,
                        # Docling 전용 메타데이터: 표 감지 정보
                        "parser": "docling",
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

    def process_and_chunk(self, files: list[Path]) -> list[dict[str, Any]]:
        all_hierarchical_data = []
        for file_path in tqdm(files, desc="Processing Files"):
            try:
                # 1. 파일 확장자에 따른 전략 동적 선택 및 파싱
                active_strategy = self.strategy if file_path.suffix.lower() == ".pdf" else MarkdownParserStrategy()

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

        return all_hierarchical_data

    def save_processed_data(self, data: list[dict[str, Any]]) -> list[Path]:
        """
        전처리된 데이터를 source_id별 개별 JSON 파일로 저장합니다.
        (BM25 인덱스와의 정합성 유지를 위해 개별 파일 관리가 필수적임)
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

    def cleanup_db(self, source_ids_to_delete: list[str]):
        """
        DB, 캐시 및 가공된 JSON 파일에서 삭제된 데이터를 제거합니다.
        (BM25Manager는 processed_dir의 모든 json을 읽으므로 물리적 파일 삭제가 곧 정합성임)
        """
        if not source_ids_to_delete:
            return

        logger.info(f"데이터 정합성 검증: {len(source_ids_to_delete)}개 소스 ID에 대한 클린업을 수행합니다.")

        # 1. 벡터 DB 데이터 삭제
        try:
            self.db_manager.delete_documents(where={"source_id": {"$in": source_ids_to_delete}})
            logger.info("   - ChromaDB 벡터 데이터 삭제 완료")
        except Exception as e:
            logger.error(f"   - ChromaDB 삭제 실패: {e}")

        # 2. 물리적 캐시 및 JSON 파일 삭제 (강화된 클린업)
        deleted_count = 0
        for sid in source_ids_to_delete:
            # 파싱 캐시 삭제
            cache_file = CACHE_DIR / f"{sid}_parsed.pkl"
            if cache_file.exists():
                cache_file.unlink()

            # 가공된 JSON 삭제 (BM25 정합성 핵심)
            json_file = self.processed_dir / f"{sid}.json"
            if json_file.exists():
                json_file.unlink()
                deleted_count += 1

        if deleted_count > 0:
            logger.info(f"   - 물리적 데이터 파일 {deleted_count}개 삭제 완료")

        logger.info("데이터 정합성 검증 및 클린업 작업이 완료되었습니다.")

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

    def _load_manifest(self) -> dict[str, str]:
        """manifest.json 파일을 로드합니다. 파일이 없거나 손상된 경우, 빈 dict를 반환합니다."""
        if not self.manifest_path.exists():
            logger.warning("Manifest 파일이 없어 전체 재색인을 수행합니다.")
            return {}
        try:
            with open(self.manifest_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning("Manifest 파일이 손상되었거나 찾을 수 없어 전체 재색인을 수행합니다.")
            if self.manifest_path.exists():
                self.manifest_path.unlink()
            return {}

    def _save_manifest(self, manifest: dict[str, str]):
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
        self, all_files: list[Path], old_manifest: dict[str, str]
    ) -> tuple[list[Path], list[str], dict[str, str]]:
        """현재 파일 상태와 이전 상태를 비교하여 변경 사항(Delta)을 계산합니다."""
        new_manifest = {str(f.relative_to(RAW_DATA_DIR)): generate_file_hash(f, self.parser_type) for f in all_files}

        files_to_process_relative = [p for p, h in new_manifest.items() if old_manifest.get(p) != h]
        files_to_process = [RAW_DATA_DIR / p for p in files_to_process_relative]

        source_ids_to_delete = [
            h for p, h in old_manifest.items() if p not in new_manifest or old_manifest[p] != new_manifest[p]
        ]

        return files_to_process, source_ids_to_delete, new_manifest

    def _process_changes(self, files_to_process: list[Path], source_ids_to_delete: list[str], session: Any):
        """도출된 변경 사항(DB 삭제, 파싱, 업서트, 인덱스 갱신)을 순차적으로 수행합니다."""
        if files_to_process or source_ids_to_delete:
            with session.trace_step("cache_flush"):
                logger.info("데이터 변경이 감지되어 시맨틱 캐시를 초기화합니다.")
                self.cache.flush()

        if source_ids_to_delete:
            with session.trace_step("db_cleanup"):
                self.ingestion_pipeline.cleanup_db(source_ids_to_delete)

        if files_to_process:
            with session.trace_step("parse_and_chunk") as step:
                processed_data = self.ingestion_pipeline.process_and_chunk(files_to_process)
                step["parent_chunk_count"] = len(processed_data)

            if processed_data:
                with session.trace_step("save_json") as step:
                    save_paths = self.ingestion_pipeline.save_processed_data(processed_data)
                    step["saved_files"] = [p.name for p in save_paths]

                with session.trace_step("db_upsert"):
                    self.ingestion_pipeline.upsert_to_db(processed_data)

        if files_to_process or source_ids_to_delete:
            with session.trace_step("bm25_update") as step:
                try:
                    from src.vector_db.bm25_manager import BM25Manager

                    BM25Manager()  # Rebuilds 수퍼 클래스
                    step["status"] = "success"
                except Exception as e:
                    step["status"] = f"failed: {e}"

    def run_ingestion(self, force: bool = False):
        """전체 데이터 구축 파이프라인 실행"""
        logger.info(f"데이터 구축 파이프라인을 시작합니다. (전략: {self.parser_type}, 강제 재색인: {force})")

        with self.tracing_logger.start_session(type="ingestion", parser_type=self.parser_type) as session:
            with session.trace_step("scan_and_check_updates") as step:
                all_files = self.ingestion_pipeline.scan_files()
                old_manifest = self._load_manifest()

                if not all_files and not old_manifest:
                    logger.warning("처리할 파일이 없고 이전 기록도 없습니다.")
                    session.data["status"] = "no_files"
                    return

                if force:
                    logger.info("강제 동기화가 요청되었습니다. 모든 기존 데이터를 삭제하고 전체 재색인을 수행합니다.")
                    files_to_process = all_files
                    source_ids_to_delete = list(old_manifest.values())
                    new_manifest = {
                        str(f.relative_to(RAW_DATA_DIR)): generate_file_hash(f, self.parser_type) for f in all_files
                    }
                else:
                    files_to_process, source_ids_to_delete, new_manifest = self._calculate_delta(
                        all_files, old_manifest
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
                    return

            self._process_changes(files_to_process, source_ids_to_delete, session)

            with session.trace_step("update_manifest"):
                self._save_manifest(new_manifest)

        logger.info("데이터 구축 파이프라인 작업이 완료되었습니다.")


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
