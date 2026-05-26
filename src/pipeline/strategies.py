import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.common.config import settings
from src.common.constants import MetadataFields
from src.data.parser import ManualParser
from src.processing.pdf_parser import DoclingPDFParser
from src.utils.file_utils import generate_file_hash
from src.utils.paths import RAW_DATA_DIR

logger = logging.getLogger(__name__)


class ParserStrategy(ABC):
    """문서 파싱 전략을 위한 추상 베이스 클래스"""

    @abstractmethod
    def parse(self, file_path: Path, storage_manager: Any = None) -> list[dict[str, Any]]:
        pass


class ManualParserStrategy(ParserStrategy):
    """기존 ManualParser를 사용하는 전략"""

    def parse(self, file_path: Path, storage_manager: Any = None) -> list[dict[str, Any]]:
        # ManualParser는 RAW_DATA_DIR 기준 상대 경로를 받음
        relative_path = file_path.relative_to(RAW_DATA_DIR)
        parser = ManualParser(str(relative_path), parser_type="manual")
        return parser.parse()


class MarkdownParserStrategy(ParserStrategy):
    """Markdown 파일을 구조 파괴 없이 읽어오는 전략"""

    def parse(self, file_path: Path, storage_manager: Any = None) -> list[dict[str, Any]]:
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
    """IBM Docling 기반 고품질 PDF 파서 전략.

    - 표 구조 자동 감지 및 마크다운 변환
    - 레이아웃 구조 보존
    - 파일 I/O는 StorageManager를 사용하도록 캡슐화 처리
    """

    def __init__(self):
        self.pdf_parser = DoclingPDFParser()  # AI 모델 1회만 로드
        self.manual_parser = ManualParser  # PDF 외 파일 Fallback

    def parse(self, file_path: Path, storage_manager: Any = None) -> list[dict[str, Any]]:
        if file_path.suffix.lower() == ".pdf":
            source_id = generate_file_hash(file_path, parser_type="docling")

            # StorageManager가 주입되었을 때 캐시 로드
            if storage_manager and storage_manager.has_cache(source_id):
                logger.info(f"캐시된 Docling 파싱 결과를 로드합니다: {file_path.name}")
                return storage_manager.load_cache(source_id)

            logger.info(f"DoclingPDFParser를 사용하여 PDF 파싱: {file_path.name}")
            start_time = time.time()
            parsed = self.pdf_parser.parse(file_path)  # {markdown, tables, page_count, table_count}
            elapsed = time.time() - start_time
            logger.info(f"파싱 완료: {file_path.name} (소요 시간: {elapsed:.2f}초, 표 {parsed['table_count']}개 감지)")

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
                        "table_count": parsed["table_count"],
                        "page_count": parsed["page_count"],
                        "has_table": parsed["table_count"] > 0,
                    },
                }
            ]

            # StorageManager가 주입되었을 때 캐시 저장
            if storage_manager:
                storage_manager.save_cache(source_id, results)

            return results
        else:
            # PDF가 아닌 경우 ManualParser로 Fallback
            relative_path = file_path.relative_to(RAW_DATA_DIR)
            parser = self.manual_parser(str(relative_path), parser_type="docling")
            return parser.parse()
