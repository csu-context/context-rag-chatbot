"""동의어 과적합 어블레이션 결론을 고정하는 회귀 테스트 (Phase 2).

Phase 1 결정을 검색 동작으로 증명한다:
  - 학칙→학사규정 매핑을 되살리면 출처 혼동(거짓 양성)이 발생한다 → 삭제가 옳다.
  - 조대 약어 동의어가 없으면 '조대' 질의가 아무 문서도 못 찾는다 → 유지가 옳다.
  - phase1 상태는 출처를 보존하면서 약어 검색을 유지한다.
"""

from src.eval.overfit_ablation import (
    HARMFUL_SYNONYMS,
    NO_SYNONYMS,
    PHASE1_SYNONYMS,
    confusion_docs,
    evaluate,
    ranked_ids,
)


def test_phase1_preserves_source_distinction():
    result = evaluate(PHASE1_SYNONYMS)
    assert result["p_at_1"] == 1.0
    assert result["mrr"] == 1.0
    # '학사규정' 질의에 학칙 문서가 끼어들지 않는다(출처 보존).
    assert confusion_docs(PHASE1_SYNONYMS, "학사규정", {"haksa"}) == []
    # '학칙' 질의는 학칙 문서만 잡는다.
    assert ranked_ids(PHASE1_SYNONYMS, "학칙") == ["hakchik"]


def test_harmful_synonym_causes_source_confusion():
    # 학칙→학사규정 매핑 복원 시 '학사규정' 질의가 학칙 문서를 거짓 양성으로 끌어온다.
    confused = confusion_docs(HARMFUL_SYNONYMS, "학사규정", {"haksa"})
    assert "hakchik" in confused


def test_jodae_synonym_adds_retrieval_value():
    # 조대 약어 동의어가 검색 가치를 더한다(없으면 '조대' 질의 결과가 비어 있음).
    assert ranked_ids(PHASE1_SYNONYMS, "조대") == ["univ"]
    assert ranked_ids(NO_SYNONYMS, "조대") == []
