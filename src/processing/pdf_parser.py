import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
        file_path = Path(file_path)
        converter = self._get_converter()

        logger.info(f"Docling 파싱 시작: {file_path.name}")
        result = converter.convert(str(file_path))
        doc = result.document

        # 1. 전체 마크다운 텍스트 추출
        markdown_text = doc.export_to_markdown()

        # 2. 표 구조를 개별 메타데이터로 추출
        tables_metadata = self._extract_tables_metadata(doc)

        logger.info(f"Docling 파싱 완료: {file_path.name} (표 {len(tables_metadata)}개 감지)")

        return {
            "markdown": markdown_text,
            "tables": tables_metadata,
            "page_count": self._get_page_count(doc),
            "table_count": len(tables_metadata),
        }

    def _extract_tables_metadata(self, doc) -> list[dict[str, Any]]:
        """Docling Document에서 표 메타데이터를 추출"""
        tables = []
        try:
            for table_idx, table in enumerate(doc.tables):
                table_data = {
                    "table_index": table_idx,
                    "page": getattr(table.prov[0], "page_no", 1) if table.prov else 1,
                    "markdown": table.export_to_markdown() if hasattr(table, "export_to_markdown") else "",
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
