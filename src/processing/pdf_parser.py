import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.common.config import settings

logger = logging.getLogger(__name__)

_PAGE_BREAK_PLACEHOLDER = "<!-- page break -->"
_HEADER_FOOTER_LINES = 2  # 각 페이지 앞뒤에서 헤더/푸터 후보로 수집할 줄 수


class DoclingPDFParser:
    """
    IBM Docling 라이브러리를 활용하여 표 구조를 보존하며 PDF를 파싱합니다.
    - 34개+ 표 자동 감지 및 마크다운 변환
    - 한글 OCR (RapidOCR) 지원
    - 레이아웃 구조 보존 (제목, 제1장 등 계층적 구조)
    """

    def __init__(self):
        self._converter = None  # Lazy initialization (첫 파싱 시만 로드)

    def _get_converter(self):
        """Docling DocumentConverter를 요청 시 초기화 (Heavy AI 모델 로드는 상태 유지)"""
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter
            except ImportError as err:
                raise ImportError("docling 라이브러리가 필요합니다.\n설치: pip install docling") from err

            logger.info("Docling DocumentConverter 초기화 중... (AI 모델 로드로 잠시 소요될 수 있음)")
            self._converter = DocumentConverter()
            logger.info("Docling DocumentConverter 준비 완료")

        return self._converter

    def parse(self, file_path: str | Path) -> dict[str, Any]:
        """
        PDF를 파싱하여 마크다운 텍스트와 표 메타데이터를 반환합니다.

        Returns:
            {
                "markdown": str,        # 전체 마크다운 텍스트 (표 구조 포함)
                "tables": list[dict],   # 개별 표 데이터 목록
                "page_count": int,      # 전체 페이지 수
                "table_count": int,     # 감지된 표 개수
            }
        """
        from docling.datamodel.base_models import DocItemLabel
        from docling_core.types.doc.document import DOCUMENT_TOKENS_EXPORT_LABELS

        file_path = Path(file_path)
        converter = self._get_converter()

        logger.info(f"Docling 파싱 시작: {file_path.name}")
        result = converter.convert(str(file_path))
        doc = result.document

        # [Option 1] Docling 레이아웃 분석: PAGE_HEADER / PAGE_FOOTER 블록을 마크다운 변환 전 원천 제외
        filtered_labels = DOCUMENT_TOKENS_EXPORT_LABELS - {
            DocItemLabel.PAGE_HEADER,
            DocItemLabel.PAGE_FOOTER,
        }
        markdown_text = doc.export_to_markdown(
            labels=filtered_labels,
            page_break_placeholder=_PAGE_BREAK_PLACEHOLDER,
        )

        # [Option 2] 페이지 간 반복 출현 라인 동적 제거 (레이아웃 분류 누락분 보완)
        markdown_text = self._remove_repeated_lines(
            markdown_text,
            threshold=settings.PDF_HEADER_FOOTER_THRESHOLD,
        )

        # [Fallback] 설정 기반 정규식 패턴 적용 (페이지 번호 등 범용 경량 패턴)
        markdown_text = self._clean_pdf_noise(markdown_text)

        # 2. 표 구조를 개별 메타데이터로 추출
        tables_metadata = self._extract_tables_metadata(doc)

        logger.info(f"Docling 파싱 완료: {file_path.name} (표 {len(tables_metadata)}개 감지)")

        return {
            "markdown": markdown_text,
            "tables": tables_metadata,
            "page_count": self._get_page_count(doc),
            "table_count": len(tables_metadata),
        }

    def _remove_repeated_lines(self, text: str, threshold: float) -> str:
        """페이지 간 반복 출현하는 헤더/푸터 라인을 동적으로 제거한다.

        export_to_markdown(page_break_placeholder=...) 가 삽입한 마커로 페이지를 분리한 뒤,
        각 페이지의 앞 2줄·뒤 2줄에서 후보를 수집하고 threshold 이상의 빈도로 반복되는
        라인을 전체 텍스트에서 제거한다.
        """
        pages = text.split(_PAGE_BREAK_PLACEHOLDER)

        # 단일 페이지는 반복 분석 불가 → 마커만 정리 후 반환
        if len(pages) < 2:
            return re.sub(r"\s*" + re.escape(_PAGE_BREAK_PLACEHOLDER) + r"\s*", "\n\n", text).strip()

        total_pages = len(pages)
        candidate_counter: Counter[str] = Counter()

        for page in pages:
            non_empty = [line.strip() for line in page.split("\n") if line.strip()]
            # 앞 2줄 + 뒤 2줄; 같은 페이지 내 중복 카운트 방지
            candidates = set(non_empty[:_HEADER_FOOTER_LINES] + non_empty[-_HEADER_FOOTER_LINES:])
            for line in candidates:
                candidate_counter[line] += 1

        noise_lines = {
            line for line, count in candidate_counter.items() if count / total_pages >= threshold and len(line) > 1
        }

        if noise_lines:
            logger.debug(f"동적 헤더/푸터 제거: {len(noise_lines)}개 반복 라인 감지 — {noise_lines}")

        result_pages = []
        for page in pages:
            cleaned = "\n".join(line for line in page.split("\n") if line.strip() not in noise_lines)
            result_pages.append(cleaned)

        combined = f"\n\n{_PAGE_BREAK_PLACEHOLDER}\n\n".join(result_pages)
        return combined.strip()

    def _clean_pdf_noise(self, text: str) -> str:
        """설정 기반 정규식 패턴으로 잔여 노이즈 제거 (페이지 번호 등 범용 패턴)."""
        cleaned = text
        for pattern in settings.PDF_NOISE_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned)

        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _extract_tables_metadata(self, doc) -> list[dict[str, Any]]:
        """Docling Document에서 표 메타데이터를 추출"""
        tables = []
        try:
            for table_idx, table in enumerate(doc.tables):
                table_data = {
                    "table_index": table_idx,
                    "page": getattr(table.prov[0], "page_no", 1) if table.prov else 1,
                    "markdown": table.export_to_markdown(doc=doc) if hasattr(table, "export_to_markdown") else "",
                    "row_count": len(table.data.grid) if hasattr(table, "data") and hasattr(table.data, "grid") else 0,
                    "col_count": (
                        len(table.data.grid[0])
                        if hasattr(table, "data") and hasattr(table.data, "grid") and table.data.grid
                        else 0
                    ),
                }
                tables.append(table_data)
        except Exception as e:
            logger.warning(f"표 메타데이터 추출 실패: {e}")
        return tables

    def _get_page_count(self, doc) -> int:
        """Docling Document에서 전체 페이지 수를 추출"""
        try:
            return len(doc.pages)
        except Exception:
            return 0
