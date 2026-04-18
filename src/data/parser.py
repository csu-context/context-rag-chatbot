import hashlib
import logging
import re
from collections import Counter
from typing import Any

import fitz

from src.common.constants import MetadataFields
from src.utils.paths import RAW_DATA_DIR

# 로그 설정
logger = logging.getLogger(__name__)


class ManualParser:
    def __init__(self, file_name: str):
        """
        사내 매뉴얼 파서 초기화 (PDF, MD 지원)
        :param file_name: 파일명 (예: 'sample.pdf' 또는 'manual.md')
        """
        self.file_path = RAW_DATA_DIR / file_name
        if not self.file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {self.file_path}")

        # [표준화] 파일명 및 고유 ID 생성
        self.file_name = self.file_path.name

        # 카테고리 자동 추출 (상위 폴더명 활용)
        self.category = (
            self.file_path.parent.name
            if self.file_path.parent != RAW_DATA_DIR
            else "일반"
        )

        # 확장자 추출 (마침표 제외)
        self.extension = self.file_path.suffix.lower().replace(".", "")

        # 고유 source_id 생성
        self.source_id = self._generate_source_id()

    def _generate_source_id(self) -> str:
        """파일명과 수정 시간을 조합하여 고유 ID(Hash) 생성"""
        stats = self.file_path.stat()
        unique_str = f"{self.file_path.name}_{stats.st_mtime}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:12]

    def parse(self) -> list[dict[str, Any]]:
        """확장자에 따른 파싱 수행"""
        if self.extension == "pdf":
            return self._parse_pdf()
        elif self.extension in ["md", "markdown"]:
            return self._parse_markdown()
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {self.extension}")

    def _clean_text(self, text: str) -> str:
        """데이터 정제: 불필요한 공백 제거 및 규정 특화 정규화"""
        # 1. 중복 공백 제거
        text = re.sub(r"\s+", " ", text).strip()

        # 2. PDF 깨짐 보정 (예: '제 조 1' -> '제1조')
        text = re.sub(r"제\s*조\s*(\d+)", r"제\1조", text)
        text = re.sub(r"제\s*(\d+)\s*조", r"제\1조", text)

        # 3. 본문 내 괄호 안의 불필요한 공백 제거 (예: '( )' -> '()')
        text = re.sub(r"\(\s+\)", "()", text)

        return text

    def _add_chunk(
        self,
        data_list: list[dict[str, Any]],
        chapter: str,
        article: str,
        content: list[str],
        page: int,
    ):
        """구조화된 청크 데이터 생성 및 리스트 추가 (표준 규격 준수)"""
        cleaned_content = self._clean_text(" ".join(content))
        if not cleaned_content or len(cleaned_content) < 5:
            return

        data_list.append(
            {
                "chapter": self._clean_text(chapter),
                "article": self._clean_text(article),
                "content": cleaned_content,
                "metadata": {
                    MetadataFields.SOURCE_ID: self.source_id,
                    MetadataFields.SRC_NAME: self.file_name,
                    MetadataFields.PG_NUM: page,
                    MetadataFields.DOC_TYPE: self.extension,
                    MetadataFields.CATEGORY: self.category,
                },
            }
        )

    def _extract_block_info(self, block: dict) -> tuple[str, float]:
        """블록 내 텍스트와 최대 폰트 크기를 추출"""
        if "lines" not in block:
            return "", 0.0

        all_spans = []
        for line in block["lines"]:
            for span in line["spans"]:
                all_spans.append(span)

        all_spans.sort(key=lambda x: (x["origin"][1], x["origin"][0]))

        block_text = ""
        max_size = 0.0
        last_y = -1.0
        for span in all_spans:
            text = span["text"]
            if not text.strip():
                continue

            if last_y != -1 and abs(span["origin"][1] - last_y) > 2:
                block_text += " "

            block_text += text
            last_y = span["origin"][1]
            if span["size"] > max_size:
                max_size = span["size"]

        return block_text.strip(), round(max_size, 1)

    def _parse_pdf(self) -> list[dict[str, Any]]:
        """폰트 크기 분석 기반 PDF 파싱 로직"""
        doc = fitz.open(str(self.file_path))
        base_font_size = self._calculate_base_font_size(doc)

        structured_data = []
        state = {
            "chapter": "기본(장 없음)",
            "article": "기본(조 없음)",
            "content": [],
            "start_page": 1,
        }

        logger.info(f"PDF 분석 완료 (본문 크기: {base_font_size}) - 파싱 시작")

        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict").get("blocks", [])

            for block in blocks:
                self._process_pdf_block(
                    block, page_num, base_font_size, state, structured_data
                )

        if state["content"]:
            self._add_chunk(
                structured_data,
                state["chapter"],
                state["article"],
                state["content"],
                state["start_page"],
            )

        doc.close()
        return structured_data

    def _process_pdf_block(
        self,
        block: dict,
        page_num: int,
        base_font_size: float,
        state: dict,
        structured_data: list,
    ):
        """단일 PDF 블록을 분석하여 상태 업데이트 및 청크 추가"""
        block_text, max_size = self._extract_block_info(block)
        if not block_text or max_size < base_font_size - 0.5:
            return

        # 대제목(장) 탐지
        if max_size >= base_font_size + 1.5:
            if state["content"]:
                self._add_chunk(
                    structured_data,
                    state["chapter"],
                    state["article"],
                    state["content"],
                    state["start_page"],
                )
                state["content"] = []

            state["chapter"] = block_text
            state["article"] = "기본(조 없음)"
            state["start_page"] = page_num + 1
            return

        # 중제목(조) 탐지
        normalized_text = block_text.replace(" ", "")
        is_article = re.match(r"^제\d+조", normalized_text) or re.match(
            r"^제조\d+", normalized_text
        )

        if is_article:
            if state["content"]:
                self._add_chunk(
                    structured_data,
                    state["chapter"],
                    state["article"],
                    state["content"],
                    state["start_page"],
                )

            if normalized_text.startswith("제조"):
                state["article"] = re.sub(r"제조(\d+)", r"제\1조", block_text)
            else:
                state["article"] = block_text

            state["content"] = []
            state["start_page"] = page_num + 1
        else:
            state["content"].append(block_text)

    def _calculate_base_font_size(self, doc) -> float:
        """문서 전체에서 가장 많이 사용된 본문 폰트 크기 계산"""
        font_sizes = []
        sample_pages = min(len(doc), 10)
        for page_num in range(sample_pages):
            page = doc[page_num]
            for block in page.get_text("dict").get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["text"].strip():
                                font_sizes.append(round(span["size"], 1))
        return Counter(font_sizes).most_common(1)[0][0] if font_sizes else 11.0

    def _parse_markdown(self) -> list[dict[str, Any]]:
        """헤더 계층 기반 Markdown 파싱 로직"""
        with open(self.file_path, encoding="utf-8") as f:
            content = f.read()

        structured_data = []
        chapter = "기본(장 없음)"
        article = "기본(조 없음)"
        content_lines = []

        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.startswith("# "):
                if content_lines:
                    self._add_chunk(structured_data, chapter, article, content_lines, 1)
                    content_lines = []
                    article = "기본(조 없음)"
                chapter = line.lstrip("#").strip()
            elif line.startswith("## ") or line.startswith("### "):
                if content_lines:
                    self._add_chunk(structured_data, chapter, article, content_lines, 1)
                article = line.lstrip("#").strip()
                content_lines = []
            else:
                content_lines.append(line)

        if content_lines:
            self._add_chunk(structured_data, chapter, article, content_lines, 1)

        return structured_data


if __name__ == "__main__":
    import json

    # 단독 실행 시 테스트 로직
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # data/raw 디렉토리에서 테스트할 첫 번째 파일 자동 검색
    supported_files = []
    for ext in [".pdf", ".md", ".markdown"]:
        supported_files.extend(list(RAW_DATA_DIR.glob(f"*{ext}")))

    if supported_files:
        test_file = supported_files[0].name
        logger.info(f"테스트 파일 탐지됨: {test_file}")
        try:
            parser = ManualParser(test_file)
            parsed_data = parser.parse()
            logger.info(f"파싱 성공: {len(parsed_data)} 청크 추출됨")

            # 샘플 출력
            if parsed_data:
                logger.info("\n[첫 번째 청크 샘플]")
                print(json.dumps(parsed_data[0], ensure_ascii=False, indent=2))
        except Exception as e:
            logger.error(f"파싱 중 에러 발생: {e}")
    else:
        logger.warning(
            f"테스트를 위한 지원 파일(.pdf, .md)이 {RAW_DATA_DIR} 에 없습니다."
        )
