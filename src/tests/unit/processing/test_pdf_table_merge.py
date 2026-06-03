"""행 미분할 멀티라인 표의 셀 y정렬 전파(세로병합 복원) 단위 테스트. (이슈 #163)"""

from src.processing.pdf_parser import DoclingPDFParser


class TestAlignToAnchor:
    def test_vertical_merge_single_value(self):
        """학위 1줄(병합 중앙)이 학과 여러 줄 전체에 전파된다."""
        anchor = [(10.0, "아랍어과"), (20.0, "일본어과"), (30.0, "중국어과"), (40.0, "독일어과"), (50.0, "러시아어과")]
        value = [(30.0, "문학사")]  # 병합 영역 중앙에 1회
        assert DoclingPDFParser._align_to_anchor(anchor, value) == ["문학사"] * 5

    def test_one_to_one(self):
        """학위 줄과 학과 줄이 y로 1:1 대응하면 그대로 매칭된다."""
        anchor = [(10.0, "a"), (20.0, "b"), (30.0, "c")]
        value = [(10.0, "X"), (20.0, "Y"), (30.0, "Z")]
        assert DoclingPDFParser._align_to_anchor(anchor, value) == ["X", "Y", "Z"]

    def test_partial_merge_then_individual(self):
        """앞부분 세로병합(다수) + 뒷부분 1:1 (51p 외국어대학 패턴, value/anchor 비율 낮음)."""
        anchor = [(i * 10.0, f"a{i}") for i in range(1, 11)]  # 10줄
        value = [(30.0, "문학사"), (80.0, "아시아"), (100.0, "글로벌")]  # 3/10=0.3 → 병합 전파
        out = DoclingPDFParser._align_to_anchor(anchor, value)
        assert out == ["문학사"] * 7 + ["아시아", "아시아", "글로벌"]

    def test_one_to_one_with_blank_rows(self):
        """학위가 대체로 1:1 대응 + 학위 없는 행(학부명)은 빈칸 유지 (미래사회 패턴)."""
        anchor = [(10.0, "학부A"), (20.0, "전공1"), (30.0, "전공2"), (40.0, "학부B"), (50.0, "전공3")]
        value = [(20.0, "컨설팅학사"), (30.0, "컨설팅학사"), (50.0, "공학사")]  # 3/5=0.6 → 1:1
        out = DoclingPDFParser._align_to_anchor(anchor, value)
        assert out == ["", "컨설팅학사", "컨설팅학사", "", "공학사"]

    def test_empty_value(self):
        anchor = [(10.0, "a"), (20.0, "b")]
        assert DoclingPDFParser._align_to_anchor(anchor, []) == ["", ""]


class TestSplitMultilineRow:
    def test_vertical_merge_split(self):
        """학위가 2줄 이상이면 학과 단위로 분할되고 병합 학위는 전파된다."""
        col_lines = [
            [(60.0, "외국어대학")],
            [
                (10.0, "아랍어과"),
                (20.0, "일본어과"),
                (30.0, "중국어과"),
                (40.0, "독일어과"),
                (50.0, "기획"),
                (60.0, "글로벌"),
            ],
            [(30.0, "문학사"), (60.0, "글로벌학사")],  # 2줄 → 가드 통과, 비율 0.33 → 병합 전파
        ]
        grid = DoclingPDFParser._split_multiline_row(col_lines)
        assert grid == [
            ["외국어대학", "아랍어과", "문학사"],
            ["외국어대학", "일본어과", "문학사"],
            ["외국어대학", "중국어과", "문학사"],
            ["외국어대학", "독일어과", "문학사"],
            ["외국어대학", "기획", "문학사"],
            ["외국어대학", "글로벌", "글로벌학사"],
        ]

    def test_guard_single_value_not_split(self):
        """기준 칸 외 칸이 모두 1줄 이하면(학위 누락 의심) 분할하지 않고 단일 행을 유지한다."""
        col_lines = [
            [],  # 대학 빈칸
            [(10.0, "전자공학부"), (20.0, "전자공학전공"), (30.0, "정보통신공학부")],  # 학과 3줄
            [(30.0, "정보통신공학 공학사")],  # 학위 1줄뿐 → cross-page 누락 의심
            None,
        ]
        grid = DoclingPDFParser._split_multiline_row(col_lines)
        assert grid == [["", "전자공학부 전자공학전공 정보통신공학부", "정보통신공학 공학사", None]]

    def test_single_line_row_kept(self):
        """모든 칸 0~1줄이면 단일 행으로 유지(None은 병합 표시로 보존)."""
        col_lines = [[(10.0, "군사학부")], [(10.0, "군사학부")], [(10.0, "군사학사")], None]
        assert DoclingPDFParser._split_multiline_row(col_lines) == [["군사학부", "군사학부", "군사학사", None]]


class TestPropagateMergedCells:
    def test_none_propagated_from_above(self):
        grid = [["A", "x"], [None, "y"], [None, "z"]]
        DoclingPDFParser._propagate_merged_cells(grid)
        assert grid == [["A", "x"], ["A", "y"], ["A", "z"]]

    def test_none_at_top_becomes_empty(self):
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
