import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import fitz

from src.common.config import settings
from src.processing.layout_utils import (
    collect_chars_from_span,
    extract_block_info,
    get_table_bboxes,
    in_table,
    join_pdf_blocks,
    join_sorted_chars,
)
from src.processing.text_utils import (
    clean_text,
    normalize_pdf_text,
    normalize_table_markdown,
    strip_leading_annotation,
)

logger = logging.getLogger(__name__)

# 표 컨텍스트 추출 시 제외할 노이즈 패턴
_META_NOISE = re.compile(r"^<?(개정|신설|삭제)")
_PAGE_NUMBER = re.compile(r"^[-–]\s*\d+\s*[-–]$")  # noqa: RUF001


_PAGE_BREAK_PLACEHOLDER = "<!-- page break -->"
_CELL_LINE_TOL = 5.0  # 셀 내부 줄 분리 y 임계(pt) — 줄 간격보다 작고 한 줄 내 변동보다 큼
# value 줄/anchor 줄 비율이 이상이면 1:1 매칭(대응 없는 행은 빈칸), 미만이면 세로 병합으로 보고 전파
_ALIGN_ONE_TO_ONE_RATIO = 0.5
_HEADER_FOOTER_LINES = 2  # 각 페이지 앞뒤에서 헤더/푸터 후보로 수집할 줄 수


class DoclingPDFParser:
    """
    IBM Docling 라이브러리를 활용하여 표 구조를 보존하며 PDF를 파싱합니다.
    - 34개+ 표 자동 감지 및 마크다운 변환
    - 한글 OCR (RapidOCR) 지원
    - 레이아웃 구조 보존 (제목, 제1장 등 계층적 구조)
    """

    def __init__(self, doc_type: str = "legal"):
        self.doc_type = doc_type  # 표 셀 정규화에 doc_type 게이트 적용
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

        # 2. 표 구조를 개별 메타데이터로 추출 (PyMuPDF 셀 텍스트로 재추출)
        with fitz.open(str(file_path)) as fitz_doc:
            tables_metadata = self._extract_tables_metadata(doc, fitz_doc)

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

    def _extract_tables_metadata(self, doc, fitz_doc: "fitz.Document") -> list[dict[str, Any]]:
        """Docling으로 표 위치를 탐지하고 PyMuPDF 좌표 기반 추출로 셀 텍스트를 재구성한다.

        Docling의 export_to_markdown은 PDF span 인코딩 순서에 의존해 역순 오류가 발생할 수 있다.
        PyMuPDF rawdict 기반 문자 좌표 추출로 읽기 순서를 보장하며, 추출 실패 시 Docling 마크다운으로 폴백한다.
        """
        tables = []
        page_table_counters: dict[int, int] = {}
        try:
            for table_idx, table in enumerate(doc.tables):
                page_num = getattr(table.prov[0], "page_no", 1) if table.prov else 1
                idx_on_page = page_table_counters.get(page_num, 0)
                page_table_counters[page_num] = idx_on_page + 1

                docling_md = table.export_to_markdown(doc=doc) if hasattr(table, "export_to_markdown") else ""
                markdown = self._pymupdf_table_markdown(fitz_doc, page_num, idx_on_page, self.doc_type) or docling_md

                grid = table.data.grid if hasattr(table, "data") and hasattr(table.data, "grid") else []
                tables.append(
                    {
                        "table_index": table_idx,
                        "page": page_num,
                        "markdown": markdown,
                        "row_count": len(grid),
                        "col_count": len(grid[0]) if grid else 0,
                    }
                )
        except Exception as e:
            logger.warning(f"표 메타데이터 추출 실패: {e}")
        return tables

    @staticmethod
    def _extract_cell_text(blocks: list, cell_bbox: tuple) -> str:
        """rawdict blocks + join_sorted_chars로 셀 bbox 내 문자를 ManualParser 방식으로 추출한다."""
        x0, y0, x1, y1 = cell_bbox
        all_chars = []
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    for ch in collect_chars_from_span(span):
                        cy, cx0, _c, _size, cx1 = ch
                        cx_center = (cx0 + cx1) / 2
                        if x0 <= cx_center <= x1 and y0 <= cy <= y1:
                            all_chars.append(ch)
        if not all_chars:
            return ""
        all_chars.sort(key=lambda v: (v[0], v[1]))
        return join_sorted_chars(all_chars, cell_bbox=cell_bbox).strip()

    @staticmethod
    def _pymupdf_table_markdown(
        fitz_doc: "fitz.Document", page_num: int, idx_on_page: int, doc_type: str = "legal"
    ) -> str:
        """셀별 rawdict 추출로 마크다운을 재구성한다. 실패 시 빈 문자열.

        행 구분선이 없어 한 셀에 여러 항목이 멀티라인으로 들어간 표(예: 학과별 행이 나뉘지
        않고 한 칸에 나열되며 학위는 세로 병합된 경우)는, 셀 내부 줄의 y좌표를 정렬해 항목
        단위 논리 행으로 분할하고 빈 구간은 직전 값을 전파(Forward-fill)해 병합을 복원한다.
        """
        try:
            page = fitz_doc[page_num - 1]
            # Docling은 reading order, PyMuPDF는 detection order로 표를 열거하므로 인덱스 직매칭은
            # 다중 표 페이지에서 어긋날 수 있다. PyMuPDF 표를 (top-y, left-x)로 정렬해 reading order에 맞춘다.
            pymupdf_tables = sorted(page.find_tables().tables, key=lambda t: (round(t.bbox[1], 1), round(t.bbox[0], 1)))
            if idx_on_page >= len(pymupdf_tables):
                return ""
            table = pymupdf_tables[idx_on_page]
            blocks = page.get_text("rawdict").get("blocks", [])
            grid: list[list[str | None]] = []
            for table_row in table.rows:
                col_lines = [
                    DoclingPDFParser._extract_cell_lines(blocks, cell) if cell else None for cell in table_row.cells
                ]
                grid.extend(DoclingPDFParser._split_multiline_row(col_lines))
            DoclingPDFParser._propagate_merged_cells(grid)
            return DoclingPDFParser._cells_to_markdown(grid, doc_type)
        except Exception:
            return ""

    @staticmethod
    def _extract_cell_lines(blocks: list, cell_bbox: tuple) -> list[tuple[float, str]]:
        """셀 bbox 내 문자를 줄(y) 단위로 그룹핑해 [(y_center, text)]로 반환한다.

        한 셀에 여러 항목이 멀티라인으로 들어간 경우 항목별 정렬을 복원하기 위해 줄을 분리한다.
        각 줄 텍스트는 _extract_cell_text와 동일하게 join_sorted_chars로 한글 띄어쓰기를 보존한다.
        """
        x0, y0, x1, y1 = cell_bbox
        chars = []
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    for ch in collect_chars_from_span(span):
                        cy, cx0, _c, _size, cx1 = ch
                        if x0 <= (cx0 + cx1) / 2 <= x1 and y0 <= cy <= y1:
                            chars.append(ch)
        if not chars:
            return []
        chars.sort(key=lambda v: (v[0], v[1]))
        lines: list[tuple[float, str]] = []
        group = [chars[0]]
        for ch in chars[1:]:
            if abs(ch[0] - group[0][0]) <= _CELL_LINE_TOL:
                group.append(ch)
            else:
                lines.append(DoclingPDFParser._finalize_cell_line(group, cell_bbox))
                group = [ch]
        lines.append(DoclingPDFParser._finalize_cell_line(group, cell_bbox))
        return [(y, t) for y, t in lines if t]

    @staticmethod
    def _finalize_cell_line(group: list, cell_bbox: tuple) -> tuple[float, str]:
        y = sum(c[0] for c in group) / len(group)
        text = join_sorted_chars(sorted(group, key=lambda v: (v[0], v[1])), cell_bbox=cell_bbox).strip()
        return (y, text)

    @staticmethod
    def _align_to_anchor(
        anchor_lines: list[tuple[float, str]], value_lines: list[tuple[float, str]], tol: float = 5.0
    ) -> list[str]:
        """기준 칸(anchor)의 각 줄 y에 value 칸 줄을 정렬한다.

        value 줄 수가 anchor와 비슷하면(_ALIGN_ONE_TO_ONE_RATIO 이상) 대부분 1:1 대응으로 보고
        y가 일치하는 줄만 매칭하며, 대응 줄이 없는 anchor 행(예: 학부명)은 빈칸으로 둔다.
        value 줄이 현저히 적으면 세로 병합으로 보고 직전 값을 전파(Forward-fill)하며, 상단의
        value 없는 구간은 첫 value로 채운다.
        """
        if not value_lines:
            return ["" for _ in anchor_lines]
        if len(value_lines) / len(anchor_lines) >= _ALIGN_ONE_TO_ONE_RATIO:
            return [next((vt for vy, vt in value_lines if abs(vy - ay) <= tol), "") for ay, _ in anchor_lines]
        result = []
        vi = 0
        current = value_lines[0][1]
        for ay, _ in anchor_lines:
            while vi < len(value_lines) and value_lines[vi][0] <= ay + tol:
                current = value_lines[vi][1]
                vi += 1
            result.append(current)
        return result

    @staticmethod
    def _split_multiline_row(col_lines: list) -> list[list[str | None]]:
        """칸별 [(y,text)] 목록(None=병합 하단)을 항목 단위 논리 행들로 분할한다.

        최다 줄 칸을 기준으로 y정렬해 재구성하되, 기준 칸 외에 2줄 이상인 칸이 하나도 없으면
        (예: 학위가 1줄뿐인데 학과만 멀티라인) 정렬 근거가 약하고 누락된 값을 잘못 전파할 위험이
        크므로 분할하지 않고 단일 행으로 합친다(cross-page 누락 표 방어, #166 참고).
        그 외(모두 0~1줄)도 단일 행으로 합친다.
        """
        n_cols = len(col_lines)
        counts = {ci: len(col_lines[ci]) for ci in range(n_cols) if col_lines[ci] is not None}
        max_lines = max(counts.values(), default=0)
        single_row = [[(" ".join(t for _, t in c) if c is not None else None) for c in col_lines]]
        if max_lines <= 1:
            return single_row
        anchor_ci = max(counts, key=counts.get)
        if not any(c >= 2 for ci, c in counts.items() if ci != anchor_ci):
            return single_row
        anchor_lines = col_lines[anchor_ci]
        n_rows = len(anchor_lines)
        grid: list[list[str | None]] = [[None] * n_cols for _ in range(n_rows)]
        for ci in range(n_cols):
            lns = col_lines[ci]
            if ci == anchor_ci:
                col_vals: list[str | None] = [t for _, t in anchor_lines]
            elif lns is None:
                col_vals = [None] * n_rows
            else:
                col_vals = list(DoclingPDFParser._align_to_anchor(anchor_lines, lns))
            for ri in range(n_rows):
                grid[ri][ci] = col_vals[ri]
        return grid

    @staticmethod
    def _propagate_merged_cells(grid: list[list[str | None]]) -> None:
        """병합 하단(None) 칸을 위 행의 같은 열 값으로 전파한다(rowspan 복원)."""
        for ri in range(len(grid)):
            for ci in range(len(grid[ri])):
                if grid[ri][ci] is None:
                    grid[ri][ci] = grid[ri - 1][ci] if ri > 0 else ""

    @staticmethod
    def _cells_to_markdown(cells: list[list[str | None]], doc_type: str = "legal") -> str:
        """2D 셀 배열을 마크다운 표로 변환한다. 각 셀에 doc_type 인지 clean_text를 적용한다."""
        if not cells:
            return ""
        rows = []
        for i, row in enumerate(cells):
            cleaned = [clean_text(str(c or "").replace("\n", " "), doc_type) for c in row]
            rows.append("| " + " | ".join(cleaned) + " |")
            if i == 0:
                rows.append("|" + "|".join(["---"] * len(row)) + "|")
        return "\n".join(rows)

    def _get_page_count(self, doc) -> int:
        """Docling Document에서 전체 페이지 수를 추출"""
        try:
            return len(doc.pages)
        except Exception:
            return 0

    @staticmethod
    def build_table_content(
        tbl: dict,
        pdf_doc: "fitz.Document",
        page_title_cache: dict[int, str],
        doc_type: str = "legal",
    ) -> str:
        """표 마크다운 앞에 해당 페이지의 비-표 텍스트(표 제목/설명)를 붙여 반환한다."""
        page_num = tbl["page"]
        if page_num not in page_title_cache:
            page_title_cache[page_num] = DoclingPDFParser.extract_non_table_text(pdf_doc, page_num, doc_type)
        title_ctx = page_title_cache[page_num]
        md = normalize_table_markdown(normalize_pdf_text(tbl["markdown"], doc_type))
        return f"{title_ctx}\n\n{md}" if title_ctx else md

    @staticmethod
    def extract_non_table_text(pdf_doc: "fitz.Document", page_num: int, doc_type: str = "legal") -> str:
        """PDF 페이지에서 표 영역을 제외한 텍스트를 추출해 표 제목/설명 컨텍스트로 반환한다."""
        if page_num < 1 or page_num > len(pdf_doc):
            return ""
        page = pdf_doc[page_num - 1]
        table_bboxes = get_table_bboxes(page)

        lines = []
        for block in page.get_text("rawdict").get("blocks", []):
            if "lines" not in block:
                continue
            if table_bboxes and "bbox" in block and in_table(block["bbox"], table_bboxes):
                continue
            text, _ = extract_block_info(block)
            if not text or "|" in text:
                continue
            if _PAGE_NUMBER.match(text):
                continue
            if doc_type == "legal" and _META_NOISE.match(text):
                continue
            text = strip_leading_annotation(text).strip()
            if not text:
                continue
            lines.append((text, block.get("bbox")))

        header_skip = settings.PDF_CONTEXT_HEADER_LINES
        context_window = settings.PDF_CONTEXT_WINDOW_LINES
        if lines:
            lines = lines[header_skip:]

        return normalize_pdf_text(clean_text(join_pdf_blocks(lines[:context_window]), doc_type), doc_type)
