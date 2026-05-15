import gc
import json
import logging
import pickle
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.common.config import settings
from src.common.constants import MetadataFields
from src.data.parser import ManualParser
from src.processing.chunking import HierarchicalChunker, create_parent_child_chunks
from src.processing.pdf_parser import EnhancedPDFParser
from src.utils.file_utils import generate_file_hash
from src.utils.logger import TracingLogger
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


class EnhancedPDFParserStrategy(ParserStrategy):
    """Unstructured 기반 고도화된 PDF 파서를 사용하는 전략"""

    def __init__(self):
        self.pdf_parser = EnhancedPDFParser()
        self.manual_parser = ManualParser  # MD 파일 등을 위해 필요

    def parse(self, file_path: Path) -> list[dict[str, Any]]:
        if file_path.suffix.lower() == ".pdf":
            # 1. 캐시 파일 경로 설정 (공통 해시 유틸리티 + 파서 타입 명시)
            source_id = generate_file_hash(file_path, parser_type="enhanced")
            cache_file = CACHE_DIR / f"{source_id}_parsed.pkl"

            # 2. 캐시가 존재하면 무거운 파싱을 생략하고 바로 로드 (시간 단축)
            if cache_file.exists():
                logger.info(f"캐시된 파싱 결과를 로드합니다: {file_path.name}")
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

            logger.info(f"EnhancedPDFParser를 사용하여 PDF 파싱: {file_path.name}")
            start_time = time.time()
            documents = self.pdf_parser.parse(file_path)
            elapsed = time.time() - start_time
            logger.info(f"파싱 완료: {file_path.name} (소요 시간: {elapsed:.2f}초)")

            # unstructured의 반환값을 페이지 단위로 그룹화하여 마크다운 텍스트로 결합 (페이지 정보 보존)
            results = []
            current_page = 1
            current_content = []
            last_title = ""

            for doc in documents:
                pg = doc.metadata.get(MetadataFields.PG_NUM, 1)
                cat = doc.metadata.get(MetadataFields.CATEGORY, "")
                content = doc.page_content.strip()
                if not content:
                    continue

                if pg != current_page and current_content:
                    results.append(
                        {
                            "is_combined": True,
                            "content": "\n\n".join(current_content),
                            "metadata": {
                                MetadataFields.SOURCE_ID: source_id,
                                MetadataFields.SRC_NAME: file_path.name,
                                MetadataFields.PG_NUM: current_page,
                                MetadataFields.DOC_TYPE: "pdf",
                                MetadataFields.CATEGORY: file_path.parent.name,
                            },
                        }
                    )
                    current_content = []
                    # 다음 페이지에도 직전 타이틀(Context)을 상속시켜 계층 구조가 유지되도록 함
                    if last_title:
                        current_content.append(f"\n# {last_title}\n")

                if cat == "Title":
                    current_content.append(f"\n# {content}\n")
                    last_title = content
                elif cat == "Table":
                    current_content.append(f"\n{content}\n")
                else:
                    current_content.append(content)

                current_page = pg

            if current_content:
                results.append(
                    {
                        "is_combined": True,
                        "content": "\n\n".join(current_content),
                        "metadata": {
                            MetadataFields.SOURCE_ID: source_id,
                            MetadataFields.SRC_NAME: file_path.name,
                            MetadataFields.PG_NUM: current_page,
                            MetadataFields.DOC_TYPE: "pdf",
                            MetadataFields.CATEGORY: file_path.parent.name,
                        },
                    }
                )

            # 리소스 해제 (Memory Leak 방지)
            del documents
            gc.collect()

            # 4. 다음 실행을 위해 파싱 결과 캐시 저장
            with open(cache_file, "wb") as f:
                pickle.dump(results, f)

            return results
        else:
            # PDF가 아닌 경우 ManualParser로 Fallback (enhanced 파이프라인에서 돌아감을 명시)
            relative_path = file_path.relative_to(RAW_DATA_DIR)
            parser = self.manual_parser(str(relative_path), parser_type="enhanced")
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
                # 1. 파싱
                sections = self.strategy.parse(file_path)

                # 2. 계층적 청킹
                if file_path.suffix.lower() == ".pdf":
                    # EnhancedPDFParserStrategy 결과인 경우 마크다운 통째로 계층적 청킹 수행
                    if sections and sections[0].get("is_combined"):
                        for sec in sections:
                            base_metadata = sec["metadata"]
                            md_text = sec["content"]
                            file_chunks = create_parent_child_chunks(md_text, base_metadata)
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
                else:
                    # Markdown 또는 기타 포맷 처리
                    if sections:
                        # ManualParser는 리스트 형태이므로 첫 번째 요소를 기준으로 처리 (통상 1개 파일당 1개 content)
                        base_metadata = sections[0]["metadata"]
                        md_text = sections[0]["content"]
                        file_chunks = create_parent_child_chunks(md_text, base_metadata)
                        all_hierarchical_data.extend(file_chunks)

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
    """전체 파이프라인(Ingestion & Inference)을 총괄하는 오케스트레이터"""

    def __init__(self):
        self.parser_type = settings.PARSER_TYPE.lower()
        self.strategy = self._get_parser_strategy()
        self.ingestion_pipeline = IngestionPipeline(self.strategy)
        self.tracing_logger = TracingLogger()
        self.manifest_path = PROCESSED_DATA_DIR / "manifest.json"

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
        if self.parser_type == "enhanced":
            try:
                import unstructured  # noqa: F401

                return EnhancedPDFParserStrategy()
            except ImportError:
                logger.error("'enhanced' 파서용 'unstructured' 라이브러리가 없습니다. 'manual'로 강제 전환합니다.")
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
        # 1. DB Cleanup (삭제된 파일 처리)
        if source_ids_to_delete:
            with session.trace_step("db_cleanup"):
                self.ingestion_pipeline.cleanup_db(source_ids_to_delete)

        # 2. 파싱, 청킹, 저장, 업서트 (신규/변경된 파일만 처리)
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

        # 3. BM25 인덱스 갱신 (데이터 변경이 있었을 경우에만)
        if files_to_process or source_ids_to_delete:
            with session.trace_step("bm25_update") as step:
                try:
                    from src.vector_db.bm25_manager import BM25Manager

                    BM25Manager()  # Rebuilds the index from DB
                    step["status"] = "success"
                except Exception as e:
                    step["status"] = f"failed: {e}"

    def run_ingestion(self):
        """전체 데이터 구축 파이프라인 실행"""
        logger.info(f"데이터 구축 파이프라인을 시작합니다. (전략: {self.parser_type})")

        with self.tracing_logger.start_session(type="ingestion", parser_type=self.parser_type) as session:
            # 0. 상태 진단
            with session.trace_step("diagnostics") as step:
                from src.utils.health_check import run_full_diagnostics

                is_healthy, report = run_full_diagnostics(silent=True, check_model=False)
                step["status"] = "healthy" if is_healthy else "unhealthy"

                if not is_healthy:
                    logger.error(f"시스템 진단 실패: {report}")
                    session.data["status"] = "failed_diagnostics"
                    return

            # 1. 스캔 및 증분 업데이트 대상 식별
            with session.trace_step("scan_and_check_updates") as step:
                all_files = self.ingestion_pipeline.scan_files()
                old_manifest = self._load_manifest()

                if not all_files and not old_manifest:
                    logger.warning("처리할 파일이 없고 이전 기록도 없습니다.")
                    session.data["status"] = "no_files"
                    return

                files_to_process, source_ids_to_delete, new_manifest = self._calculate_delta(all_files, old_manifest)

                step["total_files"] = len(all_files)
                step["files_to_process"] = len(files_to_process)
                step["files_to_delete_in_db"] = len(source_ids_to_delete)

                logger.info(
                    f"파일 스캔 완료. 전체: {len(all_files)}, "
                    f"신규/변경: {len(files_to_process)}, 삭제: {len(source_ids_to_delete)}"
                )

                if not files_to_process and not source_ids_to_delete:
                    logger.info("변경 사항이 없으므로 데이터 구축 작업을 건너뜁니다.")
                    session.data["status"] = "no_changes"
                    return

            # 2. 식별된 변경 사항 일괄 처리
            self._process_changes(files_to_process, source_ids_to_delete, session)

            # 3. Manifest 업데이트
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
    orchestrator = PipelineOrchestrator()
    orchestrator.run_ingestion()
