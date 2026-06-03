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

    def test_ratio_exactly_half_is_merge(self):
        """비율 정확히 0.5는 1:1이 아니라 세로병합으로 본다 (54p 체육대학: 학위 3/학과 6).

        체육학사가 4개 학과(체육학과·공연예술무용과·태권도·스포츠산업)에 세로병합인데
        1:1로 처리하면 가장 가까운 한 행에만 붙고 나머지가 빈칸이 된다. 엄격 비교(>)로
        forward-fill 분기에 보내 병합 학위를 전 구간에 복원한다.
        """
        anchor = [
            (121.7, "체육학과"),
            (132.5, "공연예술무용과"),
            (143.5, "태권도학과"),
            (154.6, "스포츠산업학과"),
            (165.6, "스포츠건강재활융합전공"),
            (176.6, "공연·예술융합전공"),
        ]
        value = [(132.5, "체육학사"), (165.6, "헬스케어학사"), (176.6, "공연·예술학사")]  # 3/6=0.5
        out = DoclingPDFParser._align_to_anchor(anchor, value)
        assert out == ["체육학사", "체육학사", "체육학사", "체육학사", "헬스케어학사", "공연·예술학사"]

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
        """앵커는 길지만(>=4) 기준 칸 외 칸이 모두 1줄이면(학위 누락 의심) 분할 보류 (cross-page 방어)."""
        col_lines = [
            [],  # 대학 빈칸
            [
                (10.0, "전자공학부"),
                (20.0, "(전자공학전공)"),
                (30.0, "(지능IoT전공)"),
                (40.0, "정보통신공학부"),
                (50.0, "(정보통신공학전공)"),
                (60.0, "(임베디드보안전공)"),
            ],  # 학과 6줄
            [(30.0, "정보통신공학 공학사")],  # 학위 1줄뿐 → cross-page 누락 의심
            None,
        ]
        grid = DoclingPDFParser._split_multiline_row(col_lines)
        joined = "전자공학부 (전자공학전공) (지능IoT전공) 정보통신공학부 (정보통신공학전공) (임베디드보안전공)"
        assert grid == [["", joined, "정보통신공학 공학사", None]]

    def test_non_degree_table_joins_wrap(self):
        """학위표가 아닌 표(allow_split=False)는 셀 줄바꿈을 분할하지 않고 단일 행으로 합친다.

        '8학기\\n이수대상자'·'64학점\\n이내'(편입학점표)는 리스트 셀(보건과학대학 5학과 1:1)과
        형상이 동일해 기하로 구분 불가하므로, 분할은 학위표로 한정하고 그 외 표는 줄바꿈을 병합한다.
        """
        col_lines = [
            [(10.0, "8학기"), (20.0, "이수대상자")],  # 행 레이블 wrap 2줄
            [(10.0, "-")],
            [(10.0, "64학점"), (20.0, "이내")],  # 값 wrap 2줄
            [(15.0, "3학년으로 편입한 약학과 학생")],
        ]
        grid = DoclingPDFParser._split_multiline_row(col_lines, allow_split=False)
        assert grid == [["8학기 이수대상자", "-", "64학점 이내", "3학년으로 편입한 약학과 학생"]]

    def test_degree_list_anchor3_splits(self):
        """학위표(allow_split=True)에서는 짧은 앵커(3줄)도 학과 리스트로 분할된다 (p47 법학과 회귀 방지).

        편입표 wrap과 형상이 같아도 학위표 도메인에서는 분할이 올바르므로 줄 수와 무관하게 분할한다.
        """
        col_lines = [
            [(18.0, "법과대학")],  # 단과대학 1줄(병합) → 전 행 전파
            [(10.0, "법학과"), (18.0, "글로벌법학과"), (30.0, "경찰행정학과")],  # 학과 3줄
            [(14.0, "법학사"), (30.0, "경찰행정학사")],  # 학위 2줄(법학사 병합)
        ]
        grid = DoclingPDFParser._split_multiline_row(col_lines, allow_split=True)
        assert grid == [
            ["법과대학", "법학과", "법학사"],
            ["법과대학", "글로벌법학과", "법학사"],
            ["법과대학", "경찰행정학과", "경찰행정학사"],
        ]

    def test_single_line_row_kept(self):
        """모든 칸 0~1줄이면 단일 행으로 유지(None은 병합 표시로 보존)."""
        col_lines = [[(10.0, "군사학부")], [(10.0, "군사학부")], [(10.0, "군사학사")], None]
        assert DoclingPDFParser._split_multiline_row(col_lines) == [["군사학부", "군사학부", "군사학사", None]]


class TestIsDegreeMappingTable:
    class _FakeTable:
        def __init__(self, header):
            self._header = header

        def extract(self):
            return [self._header]

    def test_degree_header_true(self):
        """헤더에 '학위'(공백 포함 '학 위')가 있으면 학위표로 본다 → 분할 허용."""
        t = self._FakeTable(["대 학", "학 과(부)", "학 위"])
        assert DoclingPDFParser._is_degree_mapping_table(t) is True

    def test_non_degree_header_false(self):
        """'학위' 열이 없는 표(편입학점표 등)는 학위표가 아니다 → 분할 금지(병합)."""
        t = self._FakeTable(["구 분", "120~130학점체제의 경우", "비 고"])
        assert DoclingPDFParser._is_degree_mapping_table(t) is False


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
