import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class HwpParser:
    """
    markitdown-hwp 라이브러리를 활용하여 HWP/HWPX 파일을 Markdown으로 변환합니다.

    내부적으로 Rust 기반 docpler 엔진이 HWP 5.0 바이너리(OLE2 Compound File)를
    직접 파싱하여 표 구조를 Markdown pipe table 형식으로 복원합니다.
    """

    def __init__(self):
        self._converter = None  # Lazy initialization

    def _get_converter(self):
        """MarkItDown 변환기를 요청 시 초기화합니다."""
        if self._converter is None:
            try:
                from markitdown import MarkItDown
            except ImportError as err:
                raise ImportError("markitdown-hwp 라이브러리가 필요합니다.\n설치: pip install markitdown-hwp") from err

            logger.info("MarkItDown(HWP) 변환기 초기화 중...")
            self._converter = MarkItDown(enable_plugins=True)
            logger.info("MarkItDown(HWP) 변환기 준비 완료")

        return self._converter

    def parse(self, file_path: str | Path) -> dict[str, Any]:
        """
        HWP/HWPX 파일을 파싱하여 Markdown 텍스트를 반환합니다.

        Returns:
            {
                "markdown": str,   # 변환된 Markdown 전문 (표 포함)
                "extension": str,  # 원본 파일 확장자 (hwp / hwpx)
            }
        """
        file_path = Path(file_path)
        converter = self._get_converter()

        logger.info(f"HWP 파싱 시작: {file_path.name}")
        result = converter.convert(str(file_path))
        markdown_text = result.text_content or ""

        logger.info(f"HWP 파싱 완료: {file_path.name} ({len(markdown_text)}자)")

        return {
            "markdown": markdown_text,
            "extension": file_path.suffix.lower().replace(".", ""),
        }
