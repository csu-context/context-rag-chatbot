import re

# 블록 경계에서 새 목록 항목 시작 패턴: 숫자+마침표(1. 2.), 원문자(①-⑳)
_LIST_ITEM_START = re.compile(r"^(\d+\.|[①-⑳])")
_ESTIMATED_CHAR_WIDTH_RATIO = 0.55

# 이 문자들 뒤에서는 어절/구 경계로 보아 공백을 강제한다(블록 경계 + 셀 줄바꿈 공용).
_SPACE_AFTER_PUNCT = ".?!>)]”’'\",;:"  # noqa: RUF001

# 블록 경계 soft break(열 줄바꿈) 판정 여백: 절대 pt 대신 열 폭 대비 비율 → 해상도/규격 독립.
# 단일컬럼 A4 본문 폭(~490pt) 기준 0.08 ≈ 기존 40pt 와 동등(실측 코퍼스 동작 보존).
_SOFT_BREAK_MARGIN_RATIO = 0.08
# 이보다 좁은 폭이면 열 줄바꿈 판정을 생략한다(degenerate 레이아웃 방어).
_MIN_COLUMN_WIDTH = 100


def collect_chars_from_span(span: dict) -> list[tuple[float, float, str, float, float]]:
    """span에서 (y, x0, char, size, x1) 튜플 목록을 수집한다."""
    size = span["size"]
    chars = []
    if "chars" in span:
        for ch in span["chars"]:
            c = ch["c"]
            if not c or c in ("\n", "\r", "\t"):
                continue
            if c.isspace():
                c = " "
            x0 = ch["origin"][0]
            y0 = ch["origin"][1]
            x1 = ch["bbox"][2] if "bbox" in ch else x0 + size * _ESTIMATED_CHAR_WIDTH_RATIO
            chars.append((round(y0, 1), round(x0, 1), c, size, round(x1, 1)))
    else:
        x, y = span["origin"]
        char_w = size * _ESTIMATED_CHAR_WIDTH_RATIO
        for i, c in enumerate(span["text"]):
            if c:
                x0 = x + i * char_w
                chars.append((round(y, 1), round(x0, 1), c, size, round(x0 + char_w, 1)))
    return chars


def join_sorted_chars(
    all_chars: list[tuple[float, float, str, float, float]],
    cell_bbox: tuple[float, float, float, float] | None = None,
) -> str:
    """(y, x0, char, size, x1) 목록을 읽기 순서대로 조합해 문자열로 반환한다."""
    if not all_chars:
        return ""

    result: list[str] = []
    prev_y, prev_x1, prev_size = all_chars[0][0], -1.0, all_chars[0][3]
    for y, x0, c, size, x1 in all_chars:
        if c == " ":
            if result and result[-1] != " ":
                result.append(" ")
            prev_x1 = x1
            continue
        if abs(y - prev_y) > size * 0.5:
            if cell_bbox is not None and prev_x1 != -1.0:
                cell_left, _, cell_right, _ = cell_bbox
                cell_w = cell_right - cell_left
                margin = max(size * 0.5, cell_w * 0.1)
                near_right = prev_x1 >= cell_right - margin
                near_left = x0 <= cell_left + margin
                # 단어 중간 줄바꿈만 공백 없이 잇는다(예: '편'+'입한'→'편입한').
                # 단, 직전 글자가 구두점이면 어절/구 경계이므로 잇지 않고 공백을 삽입한다
                # (예: 셀 안에서 '다만,'이 줄 끝, '교육학과는'이 다음 줄 시작).
                if near_right and near_left and (not result or result[-1] not in _SPACE_AFTER_PUNCT):
                    prev_y, prev_x1 = y, -1.0
                    result.append(c)
                    prev_x1, prev_size = x1, size
                    continue
            if result and result[-1] != " ":
                result.append(" ")
            prev_y, prev_x1 = y, -1.0
        elif prev_x1 != -1.0 and x0 - prev_x1 > prev_size * 0.35:
            result.append(" ")
        result.append(c)
        prev_x1, prev_size = x1, size
    return "".join(result).lstrip()


def extract_block_info(block: dict) -> tuple[str, float]:
    """블록 내 텍스트와 최대 폰트 크기를 추출."""
    if "lines" not in block:
        return "", 0.0

    all_chars: list[tuple[float, float, str, float, float]] = []
    max_size = 0.0

    for line in block["lines"]:
        for span in line["spans"]:
            if span["size"] > max_size:
                max_size = span["size"]
            all_chars.extend(collect_chars_from_span(span))

    if not all_chars:
        return "", round(max_size, 1)

    all_chars.sort(key=lambda v: (v[0], v[1]))
    # 블록 bbox를 soft-break 기준으로 넘긴다: 줄이 우측 여백까지 꽉 찬 뒤 다음 줄이 좌측 여백에서
    # 시작하면(=폭이 꽉 찬 줄의 단어 중간 줄바꿈) 공백 없이 잇는다. 줄바꿈에 무조건 공백을 넣으면
    # '매뉴얼'이 '매뉴 얼'로 갈라져 형태소 분석에서 토큰 자체가 사라진다(검색 불가).
    # 어절 경계가 같은 위치에서 줄바꿈돼 우연히 붙더라도(예: '안내복학') Kiwi가 동일하게
    # 재분해('안내','복학')하므로 하류 영향이 없다 — 과결합은 복구 가능, 과소결합(분할)은 토큰 소실로
    # 비가역. 줄 끝이 구두점이면 join_sorted_chars가 어절 경계로 보아 공백을 유지한다('다만,' 줄바꿈).
    # 표 셀은 별도 경로(_extract_cell_text)에서 셀 bbox로 처리된다.
    return join_sorted_chars(all_chars, cell_bbox=block.get("bbox")), round(max_size, 1)


def join_pdf_blocks(blocks: list[tuple[str, tuple[float, float, float, float] | None]]) -> str:
    """PDF 블록 목록 연결: bbox 기반 soft break 감지 및 trailing space 보존."""
    if not blocks:
        return ""

    result = blocks[0][0]
    prev_bbox = blocks[0][1]

    valid_bboxes = [b[1] for b in blocks if b[1]]
    if valid_bboxes:
        page_left = min(b[0] for b in valid_bboxes)
        page_right = max(b[2] for b in valid_bboxes)
    else:
        page_left, page_right = 0, 9999

    for text, bbox in blocks[1:]:
        if result and not result[-1].isspace() and text and not text[0].isspace():
            insert_space = True

            if prev_bbox and bbox and (page_right - page_left > _MIN_COLUMN_WIDTH):
                margin = (page_right - page_left) * _SOFT_BREAK_MARGIN_RATIO
                prev_x1 = prev_bbox[2]
                curr_x0 = bbox[0]
                near_right = prev_x1 >= page_right - margin
                near_left = curr_x0 <= page_left + margin
                if near_right and near_left:
                    insert_space = False

            if result[-1] in _SPACE_AFTER_PUNCT or _LIST_ITEM_START.match(text):
                insert_space = True

            if insert_space:
                result += " "
        result += text
        prev_bbox = bbox
    return result


def get_table_bboxes(page) -> list[tuple[float, float, float, float]]:
    """페이지 내 표 영역 bbox 목록 반환."""
    try:
        return [t.bbox for t in page.find_tables().tables]
    except Exception:
        return []


def in_table(
    block_bbox: tuple[float, float, float, float],
    table_bboxes: list[tuple[float, float, float, float]],
) -> bool:
    """block_bbox가 임의의 표 영역과 겹치는지 확인."""
    bx0, by0, bx1, by1 = block_bbox
    return any(bx0 < tx1 and bx1 > tx0 and by0 < ty1 and by1 > ty0 for tx0, ty0, tx1, ty1 in table_bboxes)
