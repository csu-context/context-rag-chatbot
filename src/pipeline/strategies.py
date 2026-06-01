import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import fitz

from src.common.config import settings
from src.common.constants import MetadataFields
from src.data.parser import ManualParser
from src.processing.hwp_parser import HwpParser
from src.processing.pdf_parser import DoclingPDFParser
from src.utils.file_utils import generate_file_hash
from src.utils.paths import RAW_DATA_DIR

logger = logging.getLogger(__name__)


def _safe_relative_to(file_path: Path, base: Path) -> Path:
    """relative_to 실패 시 절대 경로 반환 (RAW_DATA_DIR 외부 파일 방어)"""
    try:
        return file_path.relative_to(base)
    except ValueError:
        logger.warning(f"파일이 RAW_DATA_DIR 외부에 위치: {file_path}. 절대 경로 사용.")
        return file_path


class ParserStrategy(ABC):
    """문서 파싱 전략을 위한 추상 베이스 클래스"""

    @abstractmethod
    def parse(self, file_path: Path, storage_manager: Any = None) -> list[dict[str, Any]]:
        pass


class ManualParserStrategy(ParserStrategy):
    """기존 ManualParser를 사용하는 전략"""

    def parse(self, file_path: Path, storage_manager: Any = None) -> list[dict[str, Any]]:
        relative_path = _safe_relative_to(file_path, RAW_DATA_DIR)
        parser = ManualParser(str(relative_path), parser_type="manual", doc_type=settings.DOC_TYPE)
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
            MetadataFields.RELATIVE_PATH: str(_safe_relative_to(file_path, RAW_DATA_DIR)),
            MetadataFields.PARSER_TYPE: parser_type,
            MetadataFields.DOC_TYPE: file_path.suffix.lower().replace(".", ""),
            MetadataFields.PG_NUM: 1,
            MetadataFields.CATEGORY: file_path.parent.name if file_path.parent.name != "raw" else "일반",
        }

        return [{"is_raw_markdown": True, "content": raw_md_text, "metadata": base_metadata}]


class DoclingPDFParserStrategy(ParserStrategy):
    """하이브리드 PDF 파서 전략: Manual(텍스트) + Docling(표 추출).

    - 텍스트: ManualParser(rawdict 문자 단위 정렬) → 숫자 분리 레이아웃 문제 없음
    - 표: DoclingPDFParser → AI 레이아웃 분석으로 복잡한 표 구조 정확히 추출
    - 표는 별도 청크로 저장 (IS_TABLE=True)
    """

    def __init__(self):
        self.doc_type = settings.DOC_TYPE
        self.pdf_parser = DoclingPDFParser(doc_type=self.doc_type)  # 표 추출 전용, AI 모델 1회만 로드

    def parse(self, file_path: Path, storage_manager: Any = None) -> list[dict[str, Any]]:
        if file_path.suffix.lower() != ".pdf":
            relative_path = _safe_relative_to(file_path, RAW_DATA_DIR)
            return ManualParser(str(relative_path), parser_type="docling", doc_type=self.doc_type).parse()

        source_id = generate_file_hash(file_path, parser_type="docling")

        if storage_manager and storage_manager.has_cache(source_id):
            logger.info(f"캐시된 하이브리드 파싱 결과를 로드합니다: {file_path.name}")
            return storage_manager.load_cache(source_id)

        relative_path = _safe_relative_to(file_path, RAW_DATA_DIR)

        # 1. 텍스트: ManualParser (rawdict 기반 정확한 읽기 순서)
        logger.info(f"ManualParser로 텍스트 추출: {file_path.name}")
        text_sections = ManualParser(str(relative_path), parser_type="docling", doc_type=self.doc_type).parse()
        for sec in text_sections:
            sec["metadata"][MetadataFields.SOURCE_ID] = source_id

        # 2. 표: Docling (AI 레이아웃 분석)
        logger.info(f"Docling으로 표 추출: {file_path.name}")
        start_time = time.time()
        parsed = self.pdf_parser.parse(file_path)
        elapsed = time.time() - start_time
        logger.info(f"Docling 표 추출 완료: {file_path.name} ({elapsed:.2f}초, 표 {parsed['table_count']}개)")

        table_sections = self._build_table_sections(file_path, relative_path, parsed, source_id)
        results = text_sections + table_sections

        if storage_manager:
            storage_manager.save_cache(source_id, results)

        return results

    def _build_table_sections(
        self, file_path: Path, relative_path: Path, parsed: dict[str, Any], source_id: str
    ) -> list[dict[str, Any]]:
        """Docling이 감지한 표를 페이지 컨텍스트(표 제목/설명)와 함께 표 청크 목록으로 변환한다."""
        page_title_cache: dict[int, str] = {}
        with fitz.open(str(file_path)) as pdf_doc:
            return [
                {
                    "chapter": f"표 (p.{tbl['page']})",
                    "article": f"표 {tbl['table_index'] + 1}",
                    "content": DoclingPDFParser.build_table_content(tbl, pdf_doc, page_title_cache, self.doc_type),
                    "metadata": {
                        MetadataFields.SOURCE_ID: source_id,
                        MetadataFields.SRC_NAME: file_path.name,
                        MetadataFields.RELATIVE_PATH: str(relative_path),
                        MetadataFields.PARSER_TYPE: "docling",
                        MetadataFields.PG_NUM: tbl["page"],
                        MetadataFields.DOC_TYPE: "pdf",
                        MetadataFields.CATEGORY: file_path.parent.name,
                        MetadataFields.IS_TABLE: True,
                    },
                }
                for tbl in parsed["tables"]
                if tbl.get("markdown", "").strip()
            ]


class HwpParserStrategy(ParserStrategy):
    """markitdown-hwp(docpler Rust 엔진)를 사용하는 HWP/HWPX 파서 전략.

    - HWP 5.0 바이너리 및 HWPX XML 형식 모두 지원
    - docpler가 표 레코드(HWPTAG_TABLE)의 행/열 좌표를 복원하여 Markdown pipe table로 출력
    - 변환된 Markdown을 MarkdownParserStrategy와 동일한 섹션 구조로 반환
    """

    def __init__(self):
        self.hwp_parser = HwpParser()  # Lazy initialization, 첫 파싱 시 converter 로드

    def parse(self, file_path: Path, storage_manager: Any = None) -> list[dict[str, Any]]:
        ext = file_path.suffix.lower()
        if ext not in (".hwp", ".hwpx"):
            logger.warning(f"HwpParserStrategy: 지원하지 않는 확장자({ext}), 건너뜁니다: {file_path.name}")
            return []

        parsed = self.hwp_parser.parse(file_path)
        markdown_text = parsed["markdown"]

        if len(markdown_text.strip()) < 5:
            logger.warning(f"HWP 파싱 결과가 너무 짧아 건너뜁니다: {file_path.name}")
            return []

        parser_type = "hwp"
        base_metadata = {
            MetadataFields.SOURCE_ID: generate_file_hash(file_path, parser_type),
            MetadataFields.SRC_NAME: file_path.name,
            MetadataFields.RELATIVE_PATH: str(_safe_relative_to(file_path, RAW_DATA_DIR)),
            MetadataFields.PARSER_TYPE: parser_type,
            MetadataFields.DOC_TYPE: parsed["extension"],
            MetadataFields.PG_NUM: 1,
            MetadataFields.CATEGORY: file_path.parent.name if file_path.parent.name != "raw" else "일반",
        }

        return [{"is_raw_markdown": True, "content": markdown_text, "metadata": base_metadata}]


class ParserFactory:
    """파일 타입 및 매니페스트 설정에 따라 적절한 파서 전략을 생성하는 팩토리 클래스"""

    @staticmethod
    def create(file_path: Path, file_parser_types: dict[str, str] | None = None) -> ParserStrategy:
        suffix = file_path.suffix.lower()
        if suffix in (".hwp", ".hwpx"):
            return HwpParserStrategy()
        if suffix != ".pdf":
            return MarkdownParserStrategy()

        import importlib.util

        from src.utils.unicode import normalize_path_to_nfc, normalize_to_nfc

        rel_path = normalize_path_to_nfc(file_path.relative_to(RAW_DATA_DIR))
        normalized_parser_types = {normalize_to_nfc(k): v for k, v in (file_parser_types or {}).items()}
        file_parser = normalized_parser_types.get(rel_path, "manual").lower()

        if file_parser == "docling":
            if importlib.util.find_spec("docling") is not None:
                return DoclingPDFParserStrategy()
            logger.error("docling 라이브러리가 없어 manual 전략으로 대체합니다.")
            return ManualParserStrategy()

        return ManualParserStrategy()
