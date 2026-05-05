import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.common.constants import MetadataFields
from src.data.parser import ManualParser
from src.processing.chunking import HierarchicalChunker, create_parent_child_chunks
from src.processing.pdf_parser import EnhancedPDFParser
from src.utils.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_directories
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
        parser = ManualParser(str(relative_path))
        return parser.parse()


class EnhancedPDFParserStrategy(ParserStrategy):
    """Unstructured 기반 고도화된 PDF 파서를 사용하는 전략"""

    def __init__(self):
        self.pdf_parser = EnhancedPDFParser()
        self.manual_parser = ManualParser  # MD 파일 등을 위해 필요

    def parse(self, file_path: Path) -> list[dict[str, Any]]:
        if file_path.suffix.lower() == ".pdf":
            logger.info(f"EnhancedPDFParser를 사용하여 PDF 파싱: {file_path.name}")
            documents = self.pdf_parser.parse(file_path)
            # ManualParser의 결과 형식(list[dict])에 맞춰 변환 필요
            # EnhancedPDFParser는 list[Document]를 반환함
            structured_data = []
            for doc in documents:
                structured_data.append(
                    {
                        "chapter": "추출된 섹션",  # EnhancedParser는 장/조 구분이 아직 모호할 수 있음
                        "article": f"페이지 {doc.metadata.get('page', 1)}",
                        "content": doc.page_content,
                        "metadata": {
                            MetadataFields.SOURCE_ID: self._generate_source_id(file_path),
                            MetadataFields.SRC_NAME: file_path.name,
                            MetadataFields.PG_NUM: doc.metadata.get("page", 1),
                            MetadataFields.DOC_TYPE: "pdf",
                            MetadataFields.CATEGORY: file_path.parent.name,
                        },
                    }
                )
            return structured_data
        else:
            # PDF가 아닌 경우 ManualParser로 Fallback
            relative_path = file_path.relative_to(RAW_DATA_DIR)
            parser = self.manual_parser(str(relative_path))
            return parser.parse()

    def _generate_source_id(self, file_path: Path) -> str:
        import hashlib

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
            files.extend(list(self.raw_dir.glob(f"**/*{ext}")))
        return files

    def process_and_chunk(self, files: list[Path]) -> list[dict[str, Any]]:
        all_hierarchical_data = []
        for file_path in tqdm(files, desc="Processing Files"):
            try:
                # 1. 파싱
                sections = self.strategy.parse(file_path)

                # 2. 계층적 청킹
                if file_path.suffix.lower() == ".pdf":
                    for sec in sections:
                        parent_id = str(uuid.uuid4())
                        meta_for_children = sec["metadata"].copy()
                        meta_for_children[MetadataFields.SEC_TITLE] = f"{sec['chapter']} > {sec['article']}"

                        children = self.chunker.split_into_children(sec["content"], parent_id, meta_for_children)

                        all_hierarchical_data.append(
                            {
                                MetadataFields.PARENT_ID: parent_id,
                                "parent_text": sec["content"],
                                "metadata": sec["metadata"],
                                "children": children,
                            }
                        )
                else:
                    # Markdown 처리
                    with open(file_path, encoding="utf-8") as f:
                        md_text = f.read()

                    # 꼼수: ManualParser의 결과를 사용하여 메타데이터 추출 (이미 strategy.parse에서 생성됨)
                    if sections:
                        base_metadata = sections[0]["metadata"]
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

    def _get_parser_strategy(self) -> ParserStrategy:
        if self.parser_type == "enhanced":
            return EnhancedPDFParserStrategy()
        return ManualParserStrategy()

    def run_ingestion(self):
        """전체 데이터 구축 파이프라인 실행"""
        logger.info(f"Ingestion 시작 (전략: {self.parser_type})")

        # 0. 상태 진단
        from src.utils.health_check import run_full_diagnostics

        is_healthy, report = run_full_diagnostics(silent=True, check_model=False)
        if not is_healthy:
            logger.error(f"시스템 진단 실패: {report}")
            return

        # 1. 스캔
        files = self.ingestion_pipeline.scan_files()
        if not files:
            logger.warning("처리할 파일이 없습니다.")
            return

        # 2. 파싱 및 청킹
        processed_data = self.ingestion_pipeline.process_and_chunk(files)
        if not processed_data:
            logger.warning("가공된 데이터가 없습니다.")
            return

        # 3. 결과 저장
        self.ingestion_pipeline.save_processed_data(processed_data)

        # 4. DB 업서트
        self.ingestion_pipeline.upsert_to_db(processed_data)

        # 5. BM25 인덱스 갱신 트리거 (BM25Manager는 초기화 시 파일 날짜를 체크함)
        try:
            from src.vector_db.bm25_manager import BM25Manager

            BM25Manager()
            logger.info("BM25 인덱스 갱신 완료")
        except Exception as e:
            logger.error(f"BM25 인덱스 갱신 실패: {e}")

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
