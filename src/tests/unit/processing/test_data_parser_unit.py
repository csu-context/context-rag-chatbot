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
    # staticmethod 직접 호출하여 특이 분기 검증
    # chars가 없는 경우 (span["text"] 사용 분기)
    span_no_chars = {"size": 10.0, "origin": (10.0, 20.0), "text": "hello"}
    chars = ManualParser._collect_chars_from_span(span_no_chars)
    assert len(chars) == 5
    assert chars[0][2] == "h"

    # join_sorted_chars 줄바꿈/간격 검증
    all_chars = [
        (20.0, 10.0, "A", 10.0, 15.0),
        (20.0, 25.0, "B", 10.0, 30.0),  # x 간격 넓음
        (40.0, 10.0, "C", 10.0, 15.0),  # y 변경
    ]
    joined = ManualParser._join_sorted_chars(all_chars)
    assert "A B" in joined
    assert "C" in joined
