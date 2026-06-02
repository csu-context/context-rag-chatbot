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
        """대학(1줄)+학과(3줄)+학위(1줄 병합) → 학과 단위 3행, 학위 전파."""
        col_lines = [
            [(30.0, "외국어대학")],
            [(10.0, "아랍어과"), (20.0, "일본어과"), (30.0, "중국어과")],
            [(20.0, "문학사")],
        ]
        grid = DoclingPDFParser._split_multiline_row(col_lines)
        assert grid == [
            ["외국어대학", "아랍어과", "문학사"],
            ["외국어대학", "일본어과", "문학사"],
            ["외국어대학", "중국어과", "문학사"],
        ]

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
