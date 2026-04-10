import json
import logging
import re
from collections import Counter

import fitz

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

        # 카테고리 자동 추출 (상위 폴더명 활용)
        self.category = self.file_path.parent.name if self.file_path.parent != RAW_DATA_DIR else "일반"
        self.extension = self.file_path.suffix.lower()

    def parse(self):
        """확장자에 따른 파싱 수행"""
        if self.extension == ".pdf":
            return self._parse_pdf()
        elif self.extension in [".md", ".markdown"]:
            return self._parse_markdown()
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {self.extension}")

    def _clean_text(self, text: str):
        """데이터 정제: 불필요한 공백 제거 및 규정 특화 정규화"""
        # 1. 중복 공백 제거
        text = re.sub(r"\s+", " ", text).strip()

        # 2. PDF 깨짐 보정 (예: '제 조 1' -> '제1조')
        text = re.sub(r"제\s*조\s*(\d+)", r"제\1조", text)
        text = re.sub(r"제\s*(\d+)\s*조", r"제\1조", text)

        # 3. 본문 내 괄호 안의 불필요한 공백 제거 (예: '( )' -> '()')
        text = re.sub(r"\(\s+\)", "()", text)

        return text

    def _add_chunk(self, data_list, chapter, article, content, page):
        """구조화된 청크 데이터 생성 및 리스트 추가"""
        cleaned_content = self._clean_text(" ".join(content))
        if not cleaned_content or len(cleaned_content) < 5:
            return

        data_list.append(
            {
                "chapter": self._clean_text(chapter),
                "article": self._clean_text(article),
                "content": cleaned_content,
                "metadata": {
                    "source": self.file_path.name,
                    "category": self.category,
                    "page": page,
                    "extension": self.extension,
                },
            }
        )

    def _extract_block_info(self, block):
        """블록 내 모든 span을 추출, 정렬하여 텍스트와 최대 폰트 크기 반환"""
        if "lines" not in block:
            return None, 0

        all_spans = []
        for line in block["lines"]:
            for s in line["spans"]:
                all_spans.append(s)

        # y좌표 우선, 그 다음 x좌표 순으로 정렬 (텍스트 순서 보정)
        all_spans.sort(key=lambda x: (x["origin"][1], x["origin"][0]))

        block_text = ""
        max_size = 0
        last_y = -1
        for s in all_spans:
            text = s["text"]
            if not text.strip():
                continue

            # 줄바꿈 감지 시 공백 추가
            if last_y != -1 and abs(s["origin"][1] - last_y) > 2:
                block_text += " "

            block_text += text
            last_y = s["origin"][1]
            if s["size"] > max_size:
                max_size = s["size"]

        return block_text.strip(), round(max_size, 1)

    def _process_pdf_block(
        self,
        block_text,
        max_size,
        base_font_size,
        structured_data,
        current_chapter,
        current_article,
        current_content,
        article_start_page,
        page_num,
    ):
        """블록의 폰트 크기 및 내용을 분석하여 챕터/조/본문으로 분류"""
        # 머리말/꼬리말 무시 (본문 크기보다 작을 경우)
        if max_size < base_font_size - 0.5:
            return current_chapter, current_article, current_content, article_start_page

        # 대제목(장) 탐지
        if max_size >= base_font_size + 1.5:
            if current_content:
                self._add_chunk(structured_data, current_chapter, current_article, current_content, article_start_page)
                current_content = []

            return block_text, "기본(조 없음)", current_content, page_num + 1

        # 중제목(조) 탐지 (유연한 패턴 매칭)
        normalized_text = block_text.replace(" ", "")
        is_article = re.match(r"^제\d+조", normalized_text) or re.match(r"^제조\d+", normalized_text)

        if is_article:
            if current_content:
                self._add_chunk(structured_data, current_chapter, current_article, current_content, article_start_page)

            # 깨진 텍스트 보정 (제조1 -> 제1조)
            new_article = (
                re.sub(r"제조(\d+)", r"제\1조", block_text) if normalized_text.startswith("제조") else block_text
            )
            return current_chapter, new_article, [], page_num + 1

        current_content.append(block_text)
        return current_chapter, current_article, current_content, article_start_page

    def _parse_pdf(self):
        """폰트 크기 분석 기반 PDF 파싱 로직"""
        doc = fitz.open(str(self.file_path))
        base_font_size = self._calculate_base_font_size(doc)

        structured_data = []
        current_chapter = "기본(장 없음)"
        current_article = "기본(조 없음)"
        current_content = []
        article_start_page = 1

        logger.info(f"PDF 분석 완료 (본문 크기: {base_font_size}) - 파싱 시작")

        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict").get("blocks", [])

            for b in blocks:
                block_text, max_size = self._extract_block_info(b)
                if not block_text:
                    continue

                current_chapter, current_article, current_content, article_start_page = self._process_pdf_block(
                    block_text,
                    max_size,
                    base_font_size,
                    structured_data,
                    current_chapter,
                    current_article,
                    current_content,
                    article_start_page,
                    page_num,
                )

        if current_content:
            self._add_chunk(structured_data, current_chapter, current_article, current_content, article_start_page)

        doc.close()
        return structured_data

    def _calculate_base_font_size(self, doc):
        """문서 전체에서 가장 많이 사용된 본문 폰트 크기 계산"""
        font_sizes = []
        sample_pages = min(len(doc), 10)
        for page_num in range(sample_pages):
            page = doc[page_num]
            for b in page.get_text("dict").get("blocks", []):
                if "lines" in b:
                    for line in b["lines"]:
                        for s in line["spans"]:
                            if s["text"].strip():
                                font_sizes.append(round(s["size"], 1))
        return Counter(font_sizes).most_common(1)[0][0] if font_sizes else 11.0

    def _parse_markdown(self):
        """헤더 계층 기반 Markdown 파싱 로직"""
        with open(self.file_path, encoding="utf-8") as f:
            content = f.read()

        structured_data = []
        current_chapter = "기본(장 없음)"
        current_article = "기본(조 없음)"
        current_content = []

        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("# "):
                if current_content:
                    self._add_chunk(structured_data, current_chapter, current_article, current_content, 1)
                    current_content = []
                    current_article = "기본(조 없음)"
                current_chapter = line.lstrip("#").strip()
            elif line.startswith("## ") or line.startswith("### "):
                if current_content:
                    self._add_chunk(structured_data, current_chapter, current_article, current_content, 1)
                current_article = line.lstrip("#").strip()
                current_content = []
            else:
                current_content.append(line)

        if current_content:
            self._add_chunk(structured_data, current_chapter, current_article, current_content, 1)

        return structured_data


if __name__ == "__main__":
    # 로깅 설정
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
                logger.info(json.dumps(parsed_data[0], ensure_ascii=False, indent=2))
        except Exception as e:
            logger.error(f"파싱 중 에러 발생: {e}")
    else:
        logger.warning(f"테스트를 위한 지원 파일(.pdf, .md)이 {RAW_DATA_DIR} 에 없습니다.")
