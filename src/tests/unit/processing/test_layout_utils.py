from src.processing.layout_utils import (
    collect_chars_from_span,
    extract_block_info,
    get_table_bboxes,
    in_table,
    join_pdf_blocks,
    join_sorted_chars,
)


def test_extract_block_info_glues_full_width_line_wrap():
    # 줄이 우측 여백까지 꽉 찬 뒤(끝 x1=480 ≈ 우측 500) 다음 줄이 좌측 여백(시작 x0=12 ≈ 좌측 10)에서
    # 시작하면 단어 중간 줄바꿈으로 보아 공백 없이 잇는다. 이렇게 해야 '매뉴얼'이 '매뉴 얼'로 갈라져
    # 형태소 토큰이 소실되는 일을 막는다. 어절 경계가 우연히 겹쳐 '안내복학'이 되더라도
    # Kiwi가 '안내','복학'으로 동일 재분해하므로 하류 영향이 없다(과결합은 복구 가능).
    block = {
        "bbox": (10.0, 100.0, 500.0, 130.0),  # 폭 490pt 단락
        "lines": [
            {
                "spans": [
                    {
                        "size": 10.0,
                        "chars": [
                            {"c": "안", "origin": (460.0, 110.0), "bbox": (460.0, 100.0, 470.0, 120.0)},
                            {"c": "내", "origin": (470.0, 110.0), "bbox": (470.0, 100.0, 480.0, 120.0)},
                        ],
                    }
                ]
            },
            {
                "spans": [
                    {
                        "size": 10.0,
                        "chars": [
                            {"c": "복", "origin": (12.0, 122.0), "bbox": (12.0, 112.0, 22.0, 132.0)},
                            {"c": "학", "origin": (22.0, 122.0), "bbox": (22.0, 112.0, 32.0, 132.0)},
                        ],
                    }
                ]
            },
        ],
    }
    text, _ = extract_block_info(block)
    assert text == "안내복학"


def test_extract_block_info_keeps_space_on_short_line_wrap():
    # 줄 끝이 우측 여백에 닿지 않으면(끝 x1=130, 우측 500과 거리 큼 → 의도적 줄바꿈/짧은 줄)
    # 단어 중간 줄바꿈이 아니므로 공백을 유지한다. 헤딩·라벨·짧은 항목이 다음 줄에 들러붙지 않게 한다.
    block = {
        "bbox": (10.0, 100.0, 500.0, 130.0),
        "lines": [
            {
                "spans": [
                    {
                        "size": 10.0,
                        "chars": [
                            {"c": "안", "origin": (110.0, 110.0), "bbox": (110.0, 100.0, 120.0, 120.0)},
                            {"c": "내", "origin": (120.0, 110.0), "bbox": (120.0, 100.0, 130.0, 120.0)},
                        ],
                    }
                ]
            },
            {
                "spans": [
                    {
                        "size": 10.0,
                        "chars": [
                            {"c": "복", "origin": (12.0, 122.0), "bbox": (12.0, 112.0, 22.0, 132.0)},
                            {"c": "학", "origin": (22.0, 122.0), "bbox": (22.0, 112.0, 32.0, 132.0)},
                        ],
                    }
                ]
            },
        ],
    }
    text, _ = extract_block_info(block)
    assert text == "안내 복학"


def test_join_sorted_chars_inserts_space_after_punctuation_at_cell_wrap():
    # 셀 내부에서 구두점(쉼표)이 줄 끝, 다음 어절이 다음 줄 시작인 경우:
    # soft break로 붙이지 않고 공백을 삽입해야 한다 (예: '다만,' + 줄바꿈 + '교육학과는').
    cell_bbox = (50.0, 10.0, 100.0, 60.0)
    chars = [
        (20.0, 80.0, "가", 10.0, 92.0),  # 첫 줄
        (20.0, 92.0, ",", 10.0, 96.0),  # 쉼표가 셀 우측끝(100) 근처에서 줄 끝
        (32.0, 52.0, "나", 10.0, 67.0),  # 다음 줄 시작이 셀 좌측끝(50) 근처
        (32.0, 67.0, "다", 10.0, 82.0),
    ]
    assert join_sorted_chars(chars, cell_bbox=cell_bbox) == "가, 나다"


def test_join_sorted_chars_still_glues_midword_wrap():
    # 직전 글자가 비구두점이면 기존 soft break 동작(공백 없이 연결)을 유지한다.
    cell_bbox = (50.0, 10.0, 100.0, 60.0)
    chars = [
        (20.0, 80.0, "편", 10.0, 95.0),
        (32.0, 52.0, "입", 10.0, 67.0),
        (32.0, 67.0, "한", 10.0, 82.0),
    ]
    assert join_sorted_chars(chars, cell_bbox=cell_bbox) == "편입한"


def test_collect_chars_from_span():
    # 1. chars key present
    span_chars = {
        "size": 10.0,
        "chars": [
            {"c": "A", "origin": (10, 20), "bbox": (0, 0, 15, 0)},
            {"c": "\n", "origin": (10, 20)},  # should be ignored
            {"c": " ", "origin": (20, 20), "bbox": (0, 0, 25, 0)},  # space
        ],
    }
    res = collect_chars_from_span(span_chars)
    assert len(res) == 2
    assert res[0] == (20.0, 10.0, "A", 10.0, 15.0)
    assert res[1] == (20.0, 20.0, " ", 10.0, 25.0)

    # 2. chars key absent
    span_text = {"size": 10.0, "origin": (10, 20), "text": "Hello"}
    res2 = collect_chars_from_span(span_text)
    assert len(res2) == 5
    assert res2[0] == (20.0, 10.0, "H", 10.0, 15.5)


def test_join_pdf_blocks():
    # Empty block
    assert join_pdf_blocks([]) == ""

    # Blocks without valid bboxes
    blocks = [("Hello", None), ("World", None)]
    assert join_pdf_blocks(blocks) == "Hello World"

    # Blocks with soft breaks near margins
    # page_left = 10, page_right = 200 (width > 100)
    blocks_margin = [
        ("복", (10, 10, 190, 20)),  # near right (190 >= 200 - 40)
        ("학", (15, 30, 50, 40)),  # near left (15 <= 10 + 40)
    ]
    assert join_pdf_blocks(blocks_margin) == "복학"

    # Blocks with punctuation
    blocks_punct = [
        ("Sentence.", (10, 10, 190, 20)),
        ("Next", (15, 30, 50, 40)),
    ]
    assert join_pdf_blocks(blocks_punct) == "Sentence. Next"

    # Block with list item
    blocks_list = [
        ("Item 1", (10, 10, 190, 20)),
        ("1. Item 2", (15, 30, 50, 40)),
    ]
    assert join_pdf_blocks(blocks_list) == "Item 1 1. Item 2"


def test_get_table_bboxes():
    class MockTable:
        def __init__(self, bbox):
            self.bbox = bbox

    class MockFinder:
        def __init__(self, tables):
            self.tables = tables

    class MockPage:
        def find_tables(self):
            return MockFinder([MockTable((0, 0, 10, 10))])

    assert get_table_bboxes(MockPage()) == [(0, 0, 10, 10)]

    class MockPageFail:
        def find_tables(self):
            raise ValueError("error")

    assert get_table_bboxes(MockPageFail()) == []


def test_in_table():
    table_bboxes = [(10, 10, 50, 50)]
    assert in_table((20, 20, 30, 30), table_bboxes) is True
    assert in_table((0, 0, 5, 5), table_bboxes) is False
