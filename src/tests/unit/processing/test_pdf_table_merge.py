"""표 마크다운 재구성(rowspan 전파·유령 열 제거) 단위 테스트. (이슈 #163)

멀티라인 셀을 논리 행으로 분할하는 휴리스틱은 제거됐다('리스트 셀'과 '셀 줄바꿈'이 기하적으로
동일해 안전히 못 가르고 타 문서 일반 표를 과분할). 파서는 PyMuPDF 행/열 구조를 그대로 따르며,
병합 하단(None) 칸을 위 값으로 전파하고 전부 빈 유령 열만 떨군다.

추가로 무테두리 표에서 셀 경계 틈에 떨어져 누락되던 글자를 인접 셀로 회수하는 행 단위 추출
(_extract_row_cells / _assign_cell)을 검증한다. (이슈 #154)
"""

from src.processing.pdf_parser import DoclingPDFParser


def _char_span(c: str, x0: float, x1: float, y: float, size: float = 10.0) -> dict:
    """단일 글자 span(rawdict 형식). 글자 중심 x = (x0 + x1) / 2."""
    return {"size": size, "chars": [{"c": c, "origin": (x0, y), "bbox": (x0, y, x1, y)}]}


def _blocks(*spans: dict) -> list:
    """span들을 단일 블록/라인으로 감싼 rawdict blocks."""
    return [{"lines": [{"spans": list(spans)}]}]


class TestPropagateMergedCells:
    def test_none_propagated_from_above(self):
        """병합 하단(None) 칸은 위 행의 같은 열 값으로 전파된다(rowspan 복원)."""
        grid = [["A", "x"], [None, "y"], [None, "z"]]
        DoclingPDFParser._propagate_merged_cells(grid)
        assert grid == [["A", "x"], ["A", "y"], ["A", "z"]]

    def test_none_at_top_becomes_empty(self):
        """맨 위 행이 None이면 전파할 위 값이 없으므로 빈칸으로 둔다."""
        grid = [[None, "x"], ["B", "y"]]
        DoclingPDFParser._propagate_merged_cells(grid)
        assert grid == [["", "x"], ["B", "y"]]


class TestDropEmptyColumns:
    def test_phantom_column_dropped(self):
        """PyMuPDF가 넓은 셀을 분할해 만든 전부 빈 유령 열(헤더 포함)은 제거된다(51p 학위표)."""
        grid = [["대학", "학과", "학위", ""], ["외국어대학", "아랍어과", "문학사", ""]]
        expected = [["대학", "학과", "학위"], ["외국어대학", "아랍어과", "문학사"]]
        assert DoclingPDFParser._drop_empty_columns(grid) == expected

    def test_partially_empty_column_kept(self):
        """일부 행만 비는 열(예: 예과 행의 상위 학년)은 다른 행에 값이 있으므로 보존된다."""
        grid = [["75", "34이상", "75이상", ""], ["160", "40이상", "80이상", "120이상"]]
        assert DoclingPDFParser._drop_empty_columns(grid) == grid

    def test_no_empty_column_unchanged(self):
        grid = [["a", "b"], ["c", "d"]]
        assert DoclingPDFParser._drop_empty_columns(grid) == grid

    def test_all_empty_grid_unchanged(self):
        """전 열이 비어도(degenerate) 그대로 둔다 — 떨굴 기준 열이 없음."""
        grid = [["", ""], ["", ""]]
        assert DoclingPDFParser._drop_empty_columns(grid) == grid


def _extract(cell_grid: list, blocks: list, table_bbox: tuple) -> list:
    """cell_grid(bbox 격자) + blocks를 전역 배정→텍스트 격자로 변환(파서 본 경로 재현)."""
    positions = [((ri, ci), c) for ri, r in enumerate(cell_grid) for ci, c in enumerate(r) if c]
    buckets = DoclingPDFParser._assign_chars_to_cells(blocks, table_bbox, positions)
    return DoclingPDFParser._buckets_to_grid(cell_grid, buckets)


class TestPickCell:
    def test_containing_cell_wins(self):
        """글자를 포함하는(정확 x·y) 셀이 우선 선택된다(괘선·세로병합 헤더 = 회귀 없음)."""
        positions = [((0, 0), (0, 0, 10, 10)), ((0, 1), (20, 0, 30, 10))]
        assert DoclingPDFParser._pick_cell(5, 5, positions) == (0, 0)
        assert DoclingPDFParser._pick_cell(25, 5, positions) == (0, 1)

    def test_orphan_recovered_to_nearest_2d(self):
        """어느 셀에도 안 든 글자는 2D 최근접 셀로 회수된다."""
        positions = [((0, 0), (0, 0, 10, 10)), ((0, 1), (20, 0, 30, 10))]
        assert DoclingPDFParser._pick_cell(13, 5, positions) == (0, 0)  # 까지 3 / 7
        assert DoclingPDFParser._pick_cell(17, 5, positions) == (0, 1)  # 까지 7 / 3


class TestAssignCharsToGrid:
    def test_gap_char_not_dropped(self):
        """셀 틈에 떨어진 글자(무테두리 표 <삭제> 류)도 인접 셀로 회수돼 누락되지 않는다."""
        cell_grid = [[(0, 0, 10, 5), (20, 0, 30, 5)]]
        blocks = _blocks(
            _char_span("A", 4, 6, 2),  # (0,0) 내부
            _char_span("G", 12, 14, 2),  # 틈(center 13) → (0,0)로 회수
            _char_span("B", 24, 26, 2),  # (0,1) 내부
        )
        result = _extract(cell_grid, blocks, (0, 0, 30, 5))
        assert "A" in result[0][0] and "G" in result[0][0]
        assert result[0][1] == "B"

    def test_margin_char_excluded(self):
        """표 bbox 밖(페이지 여백) 글자는 회수 대상이 아니다(외부 유입 차단)."""
        cell_grid = [[(0, 0, 10, 5)]]
        blocks = _blocks(_char_span("A", 4, 6, 2), _char_span("M", 44, 46, 2))  # center 45 — 표 밖
        assert _extract(cell_grid, blocks, (0, 0, 10, 5)) == [["A"]]

    def test_merged_header_no_absorption(self):
        """세로병합 헤더(키 큰 라벨/비고 칸)가 아랫행 글자를 흡수하지 않는다(credit 표 회귀 방지).

        col0가 2개 헤더행에 걸쳐도, 아랫행 thin 셀에 정확히 포함된 글자는 포함 우선 규칙으로
        제 셀에 들어간다. (행 단위 y밴드 방식이 윗행에 흡수시키던 버그를 막음)
        """
        cell_grid = [
            [(0, 0, 10, 20), (10, 0, 40, 10), None],  # row0: 키 큰 라벨 / 넓은 상단헤더 / 병합
            [None, (10, 10, 25, 20), (25, 10, 40, 20)],  # row1: thin 하위 셀
        ]
        blocks = _blocks(
            _char_span("L", 4, 6, 5),  # row0 col0 라벨
            _char_span("H", 24, 26, 5),  # row0 col1 상단헤더
            _char_span("P", 16, 18, 15),  # row1 col1 — 키 큰 col0에 흡수되면 안 됨
            _char_span("Q", 31, 33, 15),  # row1 col2
        )
        result = _extract(cell_grid, blocks, (0, 0, 40, 20))
        assert result[0] == ["L", "H", None]
        assert result[1] == [None, "P", "Q"]

    def test_none_cell_preserved(self):
        """None 셀(병합 하단)은 None으로 유지된다(rowspan 전파 호환)."""
        cell_grid = [[(0, 0, 10, 5), None, (20, 0, 30, 5)]]
        blocks = _blocks(_char_span("A", 4, 6, 2), _char_span("B", 24, 26, 2))
        assert _extract(cell_grid, blocks, (0, 0, 30, 5)) == [["A", None, "B"]]

    def test_bordered_no_op(self):
        """글자가 모두 제 셀 안인 괘선 표(틈 없음)는 셀별 추출과 동일하게 동작한다(no-op)."""
        cell_grid = [[(0, 0, 10, 5), (10, 0, 20, 5)]]
        blocks = _blocks(_char_span("A", 4, 6, 2), _char_span("B", 14, 16, 2))
        assert _extract(cell_grid, blocks, (0, 0, 20, 5)) == [["A", "B"]]


class TestCleanCell:
    def test_annotation_only_cell_preserved(self):
        """셀 전체가 법령 표기(<삭제>)면 정제가 비우므로 원문을 보존한다(소스 표기 보존)."""
        assert DoclingPDFParser._clean_cell("<삭제>", "legal") == "<삭제>"
        assert DoclingPDFParser._clean_cell("<삭제> <삭제> <삭제>", "legal") == "<삭제> <삭제> <삭제>"

    def test_annotation_with_content_still_stripped(self):
        """다른 내용이 같이 있으면 어노테이션만 정상 제거된다(보존은 어노테이션 전용 셀만)."""
        assert DoclingPDFParser._clean_cell("점수 <개정 2020.1.1>", "legal") == "점수"

    def test_empty_stays_empty(self):
        """빈/공백 셀은 그대로 빈칸(보존 가드는 비어있지 않은 셀에만 발동)."""
        assert DoclingPDFParser._clean_cell(None, "legal") == ""
        assert DoclingPDFParser._clean_cell("   ", "legal") == ""

    def test_normal_cell_unchanged(self):
        assert DoclingPDFParser._clean_cell("A+ Ao", "legal") == "A+ Ao"
