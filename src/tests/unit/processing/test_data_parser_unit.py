from unittest.mock import MagicMock, patch

import pytest

from src.data.parser import ManualParser


@pytest.fixture
def mock_file_setup(tmp_path):
    # 테스트용 임시 디렉토리 및 파일 구성
    with patch("src.data.parser.RAW_DATA_DIR", tmp_path):
        yield tmp_path


def test_manual_parser_initialization_markdown(mock_file_setup):
    tmp_path = mock_file_setup
    md_file = tmp_path / "test.md"
    md_file.write_text("# Title\n\nContent line\n", encoding="utf-8")

    with patch("src.data.parser.generate_file_hash", return_value="hash_md"):
        parser = ManualParser("test.md")
        assert parser.file_name == "test.md"
        assert parser.extension == "md"
        assert parser.category == "일반"
        assert parser.source_id == "hash_md"


def test_manual_parser_parse_markdown(mock_file_setup):
    tmp_path = mock_file_setup
    md_file = tmp_path / "test.md"
    content = """# 장 1
## 조 1
본문 내용입니다.
### 조 2
또 다른 본문 내용입니다.
"""
    md_file.write_text(content, encoding="utf-8")

    with patch("src.data.parser.generate_file_hash", return_value="hash_md"):
        parser = ManualParser("test.md")
        parsed = parser.parse()
        assert len(parsed) == 2
        assert parsed[0]["chapter"] == "장 1"
        assert parsed[0]["article"] == "조 1"
        assert parsed[0]["content"] == "본문 내용입니다."
        assert parsed[1]["article"] == "조 2"


def test_manual_parser_parse_pdf(mock_file_setup):
    tmp_path = mock_file_setup
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4...")  # dummy bytes

    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 1
    mock_page = MagicMock()

    # _calculate_base_font_size를 위한 dict 모킹
    mock_page.get_text.side_effect = lambda mode, **kwargs: {
        "dict": {"blocks": [{"lines": [{"spans": [{"text": "본문", "size": 10.0}]}]}]},
        "rawdict": {
            "blocks": [
                {
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "제 1 조 목적",
                                    "size": 10.0,
                                    "origin": (10.0, 20.0),
                                    "chars": [
                                        {"c": "제", "origin": (10.0, 20.0), "bbox": (10.0, 20.0, 15.0, 30.0)},
                                        {"c": "1", "origin": (15.0, 20.0), "bbox": (15.0, 20.0, 20.0, 30.0)},
                                        {"c": "조", "origin": (20.0, 20.0), "bbox": (20.0, 20.0, 25.0, 30.0)},
                                    ],
                                }
                            ]
                        }
                    ]
                },
                {"lines": [{"spans": [{"text": "본문 내용입니다.", "size": 10.0, "origin": (10.0, 40.0)}]}]},
            ]
        },
    }.get(mode, {})

    mock_doc.__getitem__.return_value = mock_page

    with (
        patch("src.data.parser.generate_file_hash", return_value="hash_pdf"),
        patch("fitz.open", return_value=mock_doc),
    ):
        parser = ManualParser("test.pdf")
        parsed = parser.parse()
        assert len(parsed) >= 1
        assert parsed[0]["article"] == "제1조"


def test_manual_parser_chars_collect_and_join():
    # 레이아웃 유틸 함수를 직접 호출하여 특이 분기 검증
    from src.processing.layout_utils import collect_chars_from_span, join_sorted_chars

    # chars가 없는 경우 (span["text"] 사용 분기)
    span_no_chars = {"size": 10.0, "origin": (10.0, 20.0), "text": "hello"}
    chars = collect_chars_from_span(span_no_chars)
    assert len(chars) == 5
    assert chars[0][2] == "h"

    # join_sorted_chars 줄바꿈/간격 검증
    all_chars = [
        (20.0, 10.0, "A", 10.0, 15.0),
        (20.0, 25.0, "B", 10.0, 30.0),  # x 간격 넓음
        (40.0, 10.0, "C", 10.0, 15.0),  # y 변경
    ]
    joined = join_sorted_chars(all_chars)
    assert "A B" in joined
    assert "C" in joined

    # 표 셀 soft break 검증: 셀 폭이 좁아 단어 중간에서 줄바꿈 발생
    # "편입한" → 셀 내부에서 "편" + 줄바꿈 + "입한"
    cell_bbox = (50.0, 10.0, 100.0, 60.0)  # 셀 좌측 50, 우측 100
    cell_chars = [
        (20.0, 80.0, "편", 10.0, 95.0),  # 첫 줄 끝 (x1=95 ≈ 셀 우측 100)
        (32.0, 52.0, "입", 10.0, 67.0),  # 둘째 줄 시작 (x0=52 ≈ 셀 좌측 50)
        (32.0, 67.0, "한", 10.0, 82.0),
    ]
    joined_cell = join_sorted_chars(cell_chars, cell_bbox=cell_bbox)
    assert joined_cell == "편입한", f"soft break에서 공백이 삽입됨: '{joined_cell}'"

    # cell_bbox 없으면 기존 동작: 줄바꿈 → 공백 삽입
    joined_no_bbox = join_sorted_chars(cell_chars)
    assert "편 입" in joined_no_bbox


def test_normalize_table_markdown():
    from src.processing.text_utils import normalize_table_markdown

    md = """
| 이 수 학 번 | 비 고 | 이 수 내 용 | 20 01학년도 |
|---|---|---|---|
| 의학과, 치의학과, 약학과로 편 입한 학생 | 이수자 중 관련과정에서 12학 점 이상을 추가 이수하면 복 수전공 인정 | 전 공 필 수 | 20 02 |
    """  # noqa: E501

    expected = """
| 이수학번 | 비고 | 이수내용 | 2001학년도 |
|---|---|---|---|
| 의학과, 치의학과, 약학과로 편입한 학생 | 이수자 중 관련과정에서 12학점 이상을 추가 이수하면 복수전공 인정 | 전공필수 | 2002 |
    """  # noqa: E501

    assert normalize_table_markdown(md.strip()) == expected.strip()


def test_normalize_table_markdown_keeps_single_char_particles():
    # _SINGLE_CHAR_KEEP(의및중후등수)는 조사/의존명사라 인접 어절과 병합하지 않아야 한다.
    # 하드코딩 예외 목록의 동작을 명시적으로 고정하는 회귀 테스트(과적합 가시화).
    from src.processing.text_utils import normalize_table_markdown

    out = normalize_table_markdown("| 이수자 중 관련 | 학생 및 교원 |\n|---|---|")
    assert "이수자 중 관련" in out  # '중' 단독 보존
    assert "학생 및 교원" in out  # '및' 단독 보존
