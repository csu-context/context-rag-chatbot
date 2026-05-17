import logging
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.common.constants import MetadataFields

logger = logging.getLogger(__name__)


class EnhancedPDFParser:
    """
    Unstructured 라이브러리를 활용하여 표(Table) 구조를 보존하며 PDF를 파싱합니다.
    - 복잡한 표를 감지하고 HTML을 거쳐 Markdown 표로 변환
    - yolo 기반 Table Detection 모델 적용 (hi_res 전략 활용)
    """

    def __init__(self, strategy: str = "hi_res"):
        self.strategy = strategy

    def parse(self, file_path: str | Path) -> list[Document]:
        try:
            from unstructured.partition.pdf import partition_pdf
        except ImportError as err:
            raise ImportError(
                "unstructured 라이브러리가 필요합니다.\n설치: pip install unstructured[pdf] pandas"
            ) from err

        logger.info(f"파싱 시작: {file_path} (전략: {self.strategy})")

        elements = partition_pdf(
            filename=str(file_path),
            strategy=self.strategy,
            infer_table_structure=True,  # 표 구조 추론 활성화
            extract_images_in_pdf=False,  # 1차 목표에 따라 이미지 OCR은 비활성화
            languages=["kor", "eng"],  # 한국어 및 영어 인식 설정
        )

        docs = []

        for el in elements:
            el_type = el.category
            text = el.text

            # 표 요소인 경우 마크다운 표 구조로 강제 변환
            if el_type == "Table":
                html_table = el.metadata.text_as_html if hasattr(el.metadata, "text_as_html") else None
                text = self._html_to_markdown_table(html_table) if html_table else f"\n| {text} |\n|---|---|\n"

            metadata = {
                MetadataFields.SOURCE_ID: str(file_path),
                MetadataFields.PG_NUM: el.metadata.page_number if hasattr(el.metadata, "page_number") else 1,
                MetadataFields.CATEGORY: el_type,
            }

            docs.append(Document(page_content=text, metadata=metadata))

        return docs

    def _html_to_markdown_table(self, html: str) -> str:
        """HTML 표 데이터를 Pandas를 통해 깔끔한 마크다운으로 변환"""
        try:
            import pandas as pd

            dfs = pd.read_html(html)
            if dfs:
                return "\n" + dfs[0].to_markdown(index=False) + "\n"
        except Exception as e:
            logger.warning(f"HTML -> Markdown 표 변환 실패: {e}")
        return html


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
