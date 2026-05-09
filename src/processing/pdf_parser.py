import logging
from pathlib import Path

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
