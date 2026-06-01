import re

from src.common.config import settings

# 도메인 정규화 기본 모드. legal 문서에만 분류어 역순 교정·숫자단위 재결합을 적용한다.
_DEFAULT_DOC_TYPE = "legal"

_CLASSIFIER_SUBS: list[tuple[re.Pattern[str], str]] = [
    sub
    for c in ("조", "호", "항", "절", "장", "편", "관", "목")
    for sub in (
        (re.compile(rf"제\s*{c}\s*(\d+)"), rf"제\1{c}"),
        (re.compile(rf"제\s*(\d+)\s*{c}"), rf"제\1{c}"),
    )
]

# 숫자와 단위 사이의 공백 제거. 단위 목록은 settings로 외부화(코드 과적합 방지).
# 우경계 (?![가-힣]): 단위 글자 뒤에 한글이 이어지면 다른 단어의 일부일 수 있으므로 결합하지 않는다
# (예: "5 권한"→"5권한" 오결합 방지. "5 개"·"2 학기"처럼 공백/EOS 경계는 정상 결합).
_NUM_UNIT = re.compile(r"(\d+)\s+(" + "|".join(re.escape(u) for u in settings.KOREAN_NUMERIC_UNITS) + r")(?![가-힣])")

# 블록 앞머리 법령 어노테이션
_LEADING_ANNOT = re.compile(r"^<[^>]*(개정|신설|삭제)[^>]*>\s*")


def rejoin_num_unit(text: str, doc_type: str = _DEFAULT_DOC_TYPE) -> str:
    """숫자와 단위 사이에 잘못 삽입된 공백을 제거한다 (legal 도메인 한정)."""
    if doc_type != "legal":
        return text
    return _NUM_UNIT.sub(r"\1\2", text)


def normalize_pdf_text(text: str, doc_type: str = _DEFAULT_DOC_TYPE) -> str:
    """PDF 텍스트 공백 정규화: 숫자/단위 재결합 (legal 도메인 한정)."""
    return rejoin_num_unit(text, doc_type)


def strip_leading_annotation(text: str) -> str:
    """블록 앞머리 법령 어노테이션(<개정 …>, <신설 …>, <삭제 …>)을 제거한다."""
    return _LEADING_ANNOT.sub("", text)


def clean_text(text: str, doc_type: str = _DEFAULT_DOC_TYPE) -> str:
    """텍스트의 다중 공백 및 법령 어노테이션 등을 제거하여 정제한다.

    분류어 역순 교정(_CLASSIFIER_SUBS)·숫자단위 재결합은 도메인 특화 규칙이므로
    doc_type == "legal" 에서만 적용한다. 공백/빈괄호/어노테이션 정리는 범용으로 항상 수행.
    """
    text = re.sub(r"\s+", " ", text).strip()
    text = rejoin_num_unit(text, doc_type)
    if doc_type == "legal":
        for pattern, repl in _CLASSIFIER_SUBS:
            text = pattern.sub(repl, text)
    text = re.sub(r"\(\s+\)", "()", text)
    text = re.sub(r"\s*<[^>]*(개정|신설|삭제)[^>]*>", "", text).strip()
    return text


# --- 표 셀 마크다운 정규화 ---
_TABLE_NUM_SPACES = re.compile(r"(?<=\d)\s+(?=\d)")
_TABLE_SINGLE_CHAR_SEQ = re.compile(
    r"(?<![가-힣A-Za-z0-9])([가-힣A-Za-z0-9](?:\s+[가-힣A-Za-z0-9])+)(?![가-힣A-Za-z0-9])"
)
# 표 셀 띄어쓰기 교정에서 '병합 제외'할 1글자 한글 집합.
# 이들은 조사/접속/의존명사로 단독 출현이 정상이라 인접 어절과 합치면 오교정된다.
#   의(관형격 조사) · 및(접속) · 중/후(의존명사) · 등(조사/접미) · 수(의존명사)
# NOTE: 하드코딩 블랙리스트는 과적합 소지가 있다. 정식 해법은 kiwi POS(조사/의존명사) 기반
#       판정이며 후속 작업으로 분리한다. 현재는 명시 상수 + 회귀 테스트로 동작을 고정한다.
_SINGLE_CHAR_KEEP = "의및중후등수"
_TABLE_SINGLE_CHAR_LEFT = re.compile(rf"(?<![가-힣A-Za-z0-9\)])((?![{_SINGLE_CHAR_KEEP}])[가-힣])\s+([가-힣]{{2,}})")
_TABLE_SINGLE_CHAR_RIGHT = re.compile(rf"([가-힣]{{2,}})\s+((?![{_SINGLE_CHAR_KEEP}])[가-힣](?![가-힣A-Za-z0-9]))")
_TABLE_NUMBER_RIGHT = re.compile(rf"([0-9]+[가-힣])\s+((?![{_SINGLE_CHAR_KEEP}])[가-힣](?![가-힣A-Za-z0-9]))")


def _normalize_table_cell_text(text: str) -> str:
    """표 셀 내부 텍스트의 띄어쓰기 아티팩트 교정."""
    if not text:
        return text

    text = _TABLE_NUM_SPACES.sub("", text)
    text = _TABLE_SINGLE_CHAR_SEQ.sub(lambda m: m.group(1).replace(" ", ""), text)
    text = _TABLE_NUMBER_RIGHT.sub(r"\1\2", text)
    while True:
        prev_text = text
        text = _TABLE_SINGLE_CHAR_LEFT.sub(r"\1\2", text)
        text = _TABLE_SINGLE_CHAR_RIGHT.sub(r"\1\2", text)
        if text == prev_text:
            break
    return text


def normalize_table_markdown(md_text: str) -> str:
    """마크다운 표의 각 셀에 띄어쓰기 교정을 적용한다."""
    if not md_text:
        return md_text

    lines = md_text.split("\n")
    for i, line in enumerate(lines):
        if "|" in line:
            parts = line.split("|")
            for j in range(1, len(parts) - 1):
                cell = parts[j]
                if cell.strip() and not re.match(r"^\s*:?-+:?\s*$", cell):
                    leading = len(cell) - len(cell.lstrip(" "))
                    trailing = len(cell) - len(cell.rstrip(" "))
                    cleaned = _normalize_table_cell_text(cell.strip())
                    parts[j] = (" " * leading) + cleaned + (" " * trailing)
            lines[i] = "|".join(parts)

    return "\n".join(lines)
