import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import fitz

from src.common.config import settings
from src.common.constants import MetadataFields
from src.processing.layout_utils import extract_block_info, get_table_bboxes, in_table, join_pdf_blocks
from src.processing.text_utils import clean_text, normalize_pdf_text, strip_leading_annotation
from src.utils.file_utils import generate_file_hash
from src.utils.paths import RAW_DATA_DIR

logger = logging.getLogger(__name__)

# 조문 헤더 패턴 (예: 제1조, 제조1). 핫 루프 재컴파일 방지를 위해 모듈 상수로 1회 컴파일.
_ARTICLE_PATTERN = re.compile(r"^제(\d+조|조\d+)")


@dataclass
class _ParseState:
    """PDF 순차 파싱 중 누적되는 장/조/본문 상태."""

    chapter: str = "기본(장 없음)"
    article: str = "기본(조 없음)"
    content: list[tuple[str, Any]] = field(default_factory=list)
    start_page: int = 1


class ManualParser:
    def __init__(self, file_name: str, parser_type: str | None = None, doc_type: str | None = None):
        self.file_path = RAW_DATA_DIR / file_name
        if not self.file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {self.file_path}")

        self.file_name = self.file_path.name
        self.category = self.file_path.parent.name if self.file_path.parent != RAW_DATA_DIR else "일반"
        self.extension = self.file_path.suffix.lower().replace(".", "")
        self.parser_type = parser_type or os.getenv("PARSER_TYPE", "manual").lower()
        self.doc_type = doc_type or settings.DOC_TYPE
        self.source_id = generate_file_hash(self.file_path, self.parser_type)

    def parse(self) -> list[dict[str, Any]]:
        if self.extension == "pdf":
            return self._parse_pdf()
        elif self.extension in ["md", "markdown"]:
            return self._parse_markdown()
        raise ValueError(f"지원하지 않는 파일 형식입니다: {self.extension}")

    def _add_chunk(
        self,
        data_list: list[dict[str, Any]],
        chapter: str,
        article: str,
        text: str,
        page: int,
    ) -> None:
        cleaned = clean_text(text, self.doc_type)
        if not cleaned or len(cleaned) < 5:
            return
        data_list.append(
            {
                "chapter": clean_text(chapter, self.doc_type),
                "article": clean_text(article, self.doc_type),
                "content": cleaned,
                "metadata": {
                    MetadataFields.SOURCE_ID: self.source_id,
                    MetadataFields.SRC_NAME: self.file_name,
                    MetadataFields.RELATIVE_PATH: str(self.file_path.relative_to(RAW_DATA_DIR)),
                    MetadataFields.PARSER_TYPE: self.parser_type,
                    MetadataFields.PG_NUM: page,
                    MetadataFields.DOC_TYPE: self.extension,
                    MetadataFields.CATEGORY: self.category,
                },
            }
        )

    def _flush_pdf_chunk(self, state: _ParseState, data_list: list[dict[str, Any]]) -> None:
        if not state.content:
            return
        self._add_chunk(
            data_list,
            state.chapter,
            state.article,
            normalize_pdf_text(join_pdf_blocks(state.content), self.doc_type),
            state.start_page,
        )
        state.content = []

    def _parse_pdf(self) -> list[dict[str, Any]]:
        doc = fitz.open(str(self.file_path))
        base_font_size = self._calculate_base_font_size(doc)
        state = _ParseState()
        logger.info(f"PDF 분석 완료 (본문 크기: {base_font_size}) - 파싱 시작")

        structured_data: list[dict[str, Any]] = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            table_bboxes = get_table_bboxes(page)
            for block in page.get_text("rawdict").get("blocks", []):
                if table_bboxes and "bbox" in block and in_table(block["bbox"], table_bboxes):
                    continue
                self._process_pdf_block(block, page_num, base_font_size, state, structured_data)

        self._flush_pdf_chunk(state, structured_data)
        doc.close()
        return structured_data

    def _process_pdf_block(
        self,
        block: dict,
        page_num: int,
        base_font_size: float,
        state: _ParseState,
        structured_data: list,
    ) -> None:
        block_text, max_size = extract_block_info(block)
        if not block_text or max_size < base_font_size - 0.5:
            return

        if max_size >= base_font_size + 1.5:
            self._flush_pdf_chunk(state, structured_data)
            state.chapter = block_text
            state.article = "기본(조 없음)"
            state.start_page = page_num + 1
            return

        stripped = strip_leading_annotation(block_text).strip()
        normalized = stripped.replace(" ", "")
        if self.doc_type == "legal" and _ARTICLE_PATTERN.match(normalized):
            self._flush_pdf_chunk(state, structured_data)
            state.article = clean_text(stripped, self.doc_type)
            state.start_page = page_num + 1
        else:
            state.content.append((stripped or block_text, block.get("bbox")))

    def _calculate_base_font_size(self, doc) -> float:
        """문서 전체에서 가장 많이 사용된 본문 폰트 크기 계산"""
        font_sizes = []
        for page_num in range(min(len(doc), 10)):
            for block in doc[page_num].get_text("dict").get("blocks", []):
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["text"].strip():
                            font_sizes.append(round(span["size"], 1))
        return Counter(font_sizes).most_common(1)[0][0] if font_sizes else 11.0

    def _parse_markdown(self) -> list[dict[str, Any]]:
        with open(self.file_path, encoding="utf-8") as f:
            content = f.read()

        structured_data: list[dict[str, Any]] = []
        chapter = "기본(장 없음)"
        article = "기본(조 없음)"
        content_lines: list[str] = []

        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("# "):
                if content_lines:
                    self._add_chunk(structured_data, chapter, article, " ".join(content_lines), 1)
                    content_lines = []
                    article = "기본(조 없음)"
                chapter = line.lstrip("#").strip()
            elif line.startswith("## ") or line.startswith("### "):
                if content_lines:
                    self._add_chunk(structured_data, chapter, article, " ".join(content_lines), 1)
                article = line.lstrip("#").strip()
                content_lines = []
            else:
                content_lines.append(line)

        if content_lines:
            self._add_chunk(structured_data, chapter, article, " ".join(content_lines), 1)

        return structured_data


if __name__ == "__main__":
    import json

    from src.utils.logger import setup_global_logging

    setup_global_logging()

    supported_files = [f for ext in [".pdf", ".md", ".markdown"] for f in RAW_DATA_DIR.glob(f"*{ext}")]
    if supported_files:
        test_file = supported_files[0].name
        logger.info(f"테스트 파일 탐지됨: {test_file}")
        try:
            parser = ManualParser(test_file)
            parsed_data = parser.parse()
            logger.info(f"파싱 성공: {len(parsed_data)} 청크 추출됨")
            if parsed_data:
                print(json.dumps(parsed_data[0], ensure_ascii=False, indent=2))
        except Exception as e:
            logger.error(f"파싱 중 에러 발생: {e}")
    else:
        logger.warning(f"지원 파일(.pdf, .md)이 {RAW_DATA_DIR} 에 없습니다.")
