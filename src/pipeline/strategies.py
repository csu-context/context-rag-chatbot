import logging
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import fitz

from src.common.config import settings
from src.common.constants import MetadataFields
from src.data.parser import ManualParser
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
        # ManualParser는 RAW_DATA_DIR 기준 상대 경로를 받음
        relative_path = _safe_relative_to(file_path, RAW_DATA_DIR)
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
        self.pdf_parser = DoclingPDFParser()  # 표 추출 전용, AI 모델 1회만 로드

    def parse(self, file_path: Path, storage_manager: Any = None) -> list[dict[str, Any]]:
        if file_path.suffix.lower() != ".pdf":
            relative_path = _safe_relative_to(file_path, RAW_DATA_DIR)
            return ManualParser(str(relative_path), parser_type="docling").parse()

        source_id = generate_file_hash(file_path, parser_type="docling")

        if storage_manager and storage_manager.has_cache(source_id):
            logger.info(f"캐시된 하이브리드 파싱 결과를 로드합니다: {file_path.name}")
            return storage_manager.load_cache(source_id)

        relative_path = _safe_relative_to(file_path, RAW_DATA_DIR)

        # 1. 텍스트: ManualParser (rawdict 기반 정확한 읽기 순서)
        logger.info(f"ManualParser로 텍스트 추출: {file_path.name}")
        text_sections = ManualParser(str(relative_path), parser_type="docling").parse()
        for sec in text_sections:
            sec["metadata"][MetadataFields.SOURCE_ID] = source_id

        # 2. 표: Docling (AI 레이아웃 분석)
        logger.info(f"Docling으로 표 추출: {file_path.name}")
        start_time = time.time()
        parsed = self.pdf_parser.parse(file_path)
        elapsed = time.time() - start_time
        logger.info(f"Docling 표 추출 완료: {file_path.name} ({elapsed:.2f}초, 표 {parsed['table_count']}개)")

        # 표 제목 컨텍스트 추출: PDF 페이지별 비-표 텍스트 캐싱
        page_title_cache: dict[int, str] = {}
        pdf_doc = fitz.open(str(file_path))

        table_sections = [
            {
                "chapter": f"표 (p.{tbl['page']})",
                "article": f"표 {tbl['table_index'] + 1}",
                "content": self._build_table_content(tbl, pdf_doc, page_title_cache),
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

        pdf_doc.close()
        results = text_sections + table_sections

        if storage_manager:
            storage_manager.save_cache(source_id, results)

        return results

    def _build_table_content(
        self,
        tbl: dict,
        pdf_doc: "fitz.Document",
        page_title_cache: dict[int, str],
    ) -> str:
        """표 마크다운 앞에 해당 페이지의 비-표 텍스트(표 제목/설명)를 붙여 반환한다."""
        page_num = tbl["page"]
        if page_num not in page_title_cache:
            page_title_cache[page_num] = self._extract_non_table_text(pdf_doc, page_num)
        title_ctx = page_title_cache[page_num]
        md = tbl["markdown"]
        return f"{title_ctx}\n\n{md}" if title_ctx else md

    @staticmethod
    def _extract_non_table_text(pdf_doc: "fitz.Document", page_num: int) -> str:
        """PDF 페이지에서 표 셀이 아닌 텍스트(표 제목/절 제목 등)를 추출한다.

        sort=True로 읽기 순서 확보 후, 표 구분자('|')가 없는 비-표 라인만 수집한다.
        페이지 헤더(반복 출현 패턴)와 개정일 등 메타 노이즈는 제거한다.
        """
        if page_num < 1 or page_num > len(pdf_doc):
            return ""
        page = pdf_doc[page_num - 1]
        raw = page.get_text("text", sort=True)

        non_table_lines = []
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if "|" in stripped:
                continue
            # 개정일/신설 메타 노이즈 제거
            if re.match(r"^<?(개정|신설|삭제)", stripped):
                continue
            non_table_lines.append(stripped)

        # 앞 1줄(페이지 헤더)은 제거
        if non_table_lines:
            non_table_lines = non_table_lines[1:]

        return " ".join(non_table_lines[:5]).strip()  # 최대 5줄 컨텍스트
