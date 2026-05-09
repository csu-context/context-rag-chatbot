import gc
import hashlib
import json
import logging
import os
import pickle
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

from src.common.constants import MetadataFields
from src.data.parser import ManualParser
from src.processing.chunking import HierarchicalChunker, create_parent_child_chunks
from src.processing.pdf_parser import EnhancedPDFParser
from src.utils.logger import TracingLogger
from src.utils.paths import CACHE_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_directories
from src.vector_db.chroma_manager import ChromaDBManager

load_dotenv()

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
        parser = ManualParser(str(relative_path))
        return parser.parse()


class EnhancedPDFParserStrategy(ParserStrategy):
    """Unstructured 기반 고도화된 PDF 파서를 사용하는 전략"""

    def __init__(self):
        self.pdf_parser = EnhancedPDFParser()
        self.manual_parser = ManualParser  # MD 파일 등을 위해 필요

    def parse(self, file_path: Path) -> list[dict[str, Any]]:
        if file_path.suffix.lower() == ".pdf":
            # 📌 1. 캐시 파일 경로 설정 (파일 변경 시 source_id도 바뀌어 안전함)
            source_id = self._generate_source_id(file_path)
            cache_file = CACHE_DIR / f"{source_id}_parsed.pkl"

            # 📌 2. 캐시가 존재하면 무거운 파싱을 생략하고 바로 로드 (시간 단축)
            if cache_file.exists():
                logger.info(f"💾 캐시된 파싱 결과를 불러옵니다: {file_path.name}")
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

            # 📌 4. 다음 실행을 위해 파싱 결과 캐시 저장
            with open(cache_file, "wb") as f:
                pickle.dump(results, f)

            return results
        else:
            # PDF가 아닌 경우 ManualParser로 Fallback
            relative_path = file_path.relative_to(RAW_DATA_DIR)
            parser = self.manual_parser(str(relative_path))
            return parser.parse()

    def _generate_source_id(self, file_path: Path) -> str:
        stats = file_path.stat()
        unique_str = f"{file_path.name}_{stats.st_mtime}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:12]


class IngestionPipeline:
    """데이터 전처리 및 인덱싱을 담당하는 클래스"""

    def __init__(
        self,
        strategy: ParserStrategy,
        raw_dir: Path = RAW_DATA_DIR,
        processed_dir: Path = PROCESSED_DATA_DIR,
    ):
        self.strategy = strategy
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.chunker = HierarchicalChunker()
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

    def save_processed_data(self, data: list[dict[str, Any]], filename: str | None = None) -> Path:
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"preprocessed_{timestamp}.json"

        save_path = self.processed_dir / filename
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"전처리 결과 저장 완료: {save_path}")
        return save_path

    def upsert_to_db(self, data: list[dict[str, Any]], collection_name: str = "rag_collection"):
        db_manager = ChromaDBManager(collection_name=collection_name)
        ids, docs, metas = [], [], []
        for parent in data:
            for child in parent["children"]:
                ids.append(child["chunk_id"])
                docs.append(child["text"])
                metas.append(child["metadata"])

        if ids:
            logger.info(f"ChromaDB 업서트 시작 ({len(ids)}개 청크)...")
            db_manager.upsert_documents(ids=ids, documents=docs, metadatas=metas)
            logger.info("ChromaDB 업서트 완료!")


class PipelineOrchestrator:
    """전체 파이프라인(Ingestion & Inference)을 총괄하는 오케스트레이터"""

    def __init__(self):
        self.parser_type = os.getenv("PARSER_TYPE", "manual").lower()
        self.strategy = self._get_parser_strategy()
        self.ingestion_pipeline = IngestionPipeline(self.strategy)
        self.tracing_logger = TracingLogger()

    def _get_parser_strategy(self) -> ParserStrategy:
        if self.parser_type == "enhanced":
            return EnhancedPDFParserStrategy()
        return ManualParserStrategy()

    def run_ingestion(self):
        """전체 데이터 구축 파이프라인 실행"""
        logger.info(f"Ingestion 시작 (전략: {self.parser_type})")

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

            # 1. 스캔
            with session.trace_step("scan") as step:
                files = self.ingestion_pipeline.scan_files()
                step["file_count"] = len(files)

                if not files:
                    logger.warning("처리할 파일이 없습니다.")
                    session.data["status"] = "no_files"
                    return

            # 2. 파싱 및 청킹
            with session.trace_step("parse_and_chunk") as step:
                processed_data = self.ingestion_pipeline.process_and_chunk(files)
                step["parent_chunk_count"] = len(processed_data)

                if not processed_data:
                    logger.warning("가공된 데이터가 없습니다.")
                    session.data["status"] = "no_processed_data"
                    return

            # 3. 결과 저장
            with session.trace_step("save_json") as step:
                save_path = self.ingestion_pipeline.save_processed_data(processed_data)
                step["path"] = str(save_path)

            # 4. DB 업서트
            with session.trace_step("db_upsert"):
                self.ingestion_pipeline.upsert_to_db(processed_data)

            # 5. BM25 인덱스 갱신
            with session.trace_step("bm25_update") as step:
                try:
                    from src.vector_db.bm25_manager import BM25Manager

                    BM25Manager()
                    step["status"] = "success"
                except Exception as e:
                    step["status"] = f"failed: {e}"

            logger.info("Ingestion 완료!")


# 하위 호환성을 위한 기존 클래스 래핑
class PreprocessingPipeline:
    def __init__(self, *args, **kwargs):
        self.orchestrator = PipelineOrchestrator()

    def run(self, *args, **kwargs):
        return self.orchestrator.run_ingestion()


if __name__ == "__main__":
    orchestrator = PipelineOrchestrator()
    orchestrator.run_ingestion()
