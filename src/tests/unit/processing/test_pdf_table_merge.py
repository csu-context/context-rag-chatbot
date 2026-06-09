"""표 마크다운 재구성(rowspan 전파·유령 열 제거) 단위 테스트. (이슈 #163)

멀티라인 셀을 논리 행으로 분할하는 휴리스틱은 제거됐다('리스트 셀'과 '셀 줄바꿈'이 기하적으로
동일해 안전히 못 가르고 타 문서 일반 표를 과분할). 파서는 PyMuPDF 행/열 구조를 그대로 따르며,
병합 하단(None) 칸을 위 값으로 전파하고 전부 빈 유령 열만 떨군다.
"""

from src.processing.pdf_parser import DoclingPDFParser


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
