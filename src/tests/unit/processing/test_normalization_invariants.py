"""정규화 과적합 검증 장치 (Phase 2).

튜닝 코퍼스(조선대 학사규정·학칙) 밖의 '홀드아웃' 입력과 도메인 혼합 입력에 대해
정규화 규칙이 (1) 멱등하고 (2) 출처/일반 텍스트를 훼손하지 않음을 고정한다.
규칙이 특정 문서에 과적합되면 이 불변식들이 깨진다.
"""

import pytest

from src.processing.text_utils import (
    _normalize_table_cell_text,
    clean_text,
    normalize_pdf_text,
    normalize_table_markdown,
)

# 홀드아웃(코퍼스 외) + 법령/학사/영문/숫자 혼합 입력
_SAMPLES = [
    "제  1  조  목적",
    "제호2 서식 및 별표",
    "2 학기 12 학점 이수",
    "이수자 중 관련 과정에서 복 수전공 인정",
    "Section 3 of the Agreement shall apply",  # 영문 (held-out)
    "서울대학교 학칙 제3조 (목적)",  # 타 대학 (held-out)
    "사과 3 그루와 배 5 개를 수확",  # 숫자+비단위/단위 혼합
    "복학 후 전과 신청",
]

_TABLE_SAMPLES = [
    "| 이 수 학 번 | 20 01학년도 |\n|---|---|\n| 의학과 약학과로 편 입한 학생 | 12학 점 |",
    "| 회의 중 관련 | 학생 및 교원 |\n|---|---|",
]


@pytest.mark.parametrize("doc_type", ["legal", "general"])
@pytest.mark.parametrize("text", _SAMPLES)
def test_clean_text_is_idempotent(text, doc_type):
    once = clean_text(text, doc_type)
    assert clean_text(once, doc_type) == once


@pytest.mark.parametrize("text", _SAMPLES)
def test_normalize_pdf_text_is_idempotent(text):
    once = normalize_pdf_text(text)
    assert normalize_pdf_text(once) == once


@pytest.mark.parametrize("md", _TABLE_SAMPLES)
def test_normalize_table_markdown_is_idempotent(md):
    once = normalize_table_markdown(md)
    assert normalize_table_markdown(once) == once


@pytest.mark.parametrize("cell", ["이 수 학 번", "복 수전공 인정", "회의 중 관련", "20 01학년도"])
def test_normalize_table_cell_is_idempotent(cell):
    once = _normalize_table_cell_text(cell)
    assert _normalize_table_cell_text(once) == once


def test_general_doc_type_does_not_mangle_text():
    # general 모드: 도메인 규칙(분류어/숫자단위) 미적용 → 공백 정리만 수행.
    assert clean_text("제호2 서식", "general") == "제호2 서식"
    assert clean_text("2 학기", "general") == "2 학기"
    assert clean_text("Section 3 of 5", "general") == "Section 3 of 5"
    assert clean_text("서울대학교 학칙 제3조", "general") == "서울대학교 학칙 제3조"


def test_numbers_merge_only_whitelisted_units():
    # 숫자+비단위 어절은 절대 결합되지 않는다(과잉 교정 방지). 화이트리스트 단위만 결합.
    assert clean_text("사과 3 그루") == "사과 3 그루"  # 그루: 비단위 → 유지
    assert clean_text("3 그리고 4") == "3 그리고 4"  # 접속부사 → 유지
    assert clean_text("5 개") == "5개"  # 개: 단위(대조군) → 결합


def test_num_unit_right_boundary_blocks_word_prefixes():
    # 단위 글자 뒤에 한글이 이어지면 다른 단어의 일부일 수 있어 결합하지 않는다(우경계 가드).
    assert clean_text("학칙 5 권한 위임") == "학칙 5 권한 위임"  # 권한: '권'으로 시작하는 단어
    assert clean_text("10 월급 명세") == "10 월급 명세"  # 월급
    assert clean_text("3 차이 분석") == "3 차이 분석"  # 차이
    assert clean_text("5 명령 체계") == "5 명령 체계"  # 명령
    # 단위가 공백/EOS 경계에 오는 정상 케이스는 그대로 결합
    assert clean_text("정원 5 명") == "정원 5명"
    assert clean_text("3 개월") == "3개월"


def test_held_out_terms_are_not_rewritten():
    # 코퍼스 밖 고유명사/일반어는 clean_text가 재작성하지 않는다(출처 보존).
    assert "서울대학교" in clean_text("서울대학교 입학 안내")
    assert clean_text("학칙 준수 의무") == "학칙 준수 의무"  # 학칙→학사규정 류 치환 없음


def test_table_single_char_particles_preserved_between_words():
    # 다어절 사이의 1글자 조사/의존명사(의및중후등수)는 인접 어절과 병합되지 않는다.
    out = normalize_table_markdown("| 회의 중 관련 | 학생 및 교원 |\n|---|---|")
    assert "회의 중 관련" in out  # '중' 보존
    assert "학생 및 교원" in out  # '및' 보존
