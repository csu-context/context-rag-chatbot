"""동의어 과적합 어블레이션 하니스 (Phase 2 검증 장치).

소규모 라벨 코퍼스에서 동의어 사전을 토글하며 BM25 검색 지표를 측정해
Phase 1 결정을 정량 검증한다. 모델/API/임베딩 불필요 — Kiwi 토크나이저 + BM25만 사용.

세 시나리오를 비교한다:
  - phase1     : {"조대": "조선대학교"}            (현재 상태)
  - harmful    : phase1 + {"학칙": "학사규정"}     (Phase 1 이전 과적합 매핑 복원)
  - none       : {}                                (동의어 전부 제거)

결론:
  - harmful 은 '학사규정' 질의에 학칙 문서를 끌어와 출처를 혼동시킨다(거짓 양성).
  - none 은 '조대' 약어 질의에서 아무 문서도 찾지 못한다(조대 동의어의 검색 가치).
  - phase1 은 출처를 보존하면서 약어 검색을 유지한다.
"""

from __future__ import annotations

from src.vector_db.bm25_index import BM25PlusIndex
from src.vector_db.bm25_tokenizer import BM25Tokenizer

# (doc_id, text). 학칙/학사규정은 별개 출처 문서로, 토큰이 융합되면 안 된다.
# 의도한 매칭만 변별되도록 교차 어휘(조선대/규정 등)는 분리한다.
DOCS: list[tuple[str, str]] = [
    ("hakchik", "학칙 총장 임면 조항"),
    ("haksa", "학사규정 학점 이수 졸업"),
    # 조선대학교 유일 출현 → 조대 동의어 효과 변별.
    # NOTE: Kiwi는 후행어에 따라 '조선대학교'를 통째('안내' 등) 또는 분해('소개 자료' 등)한다.
    #       조대→조선대학교 동의어의 검색 효과는 이처럼 문서 토큰화에 의존적이다(어블레이션이 드러낸 한계).
    ("univ", "조선대학교 안내"),
    ("it", "인공지능 챗봇 자연어 처리"),
]

# (query, 관련 doc_id 집합)
QUERIES: list[tuple[str, set[str]]] = [
    ("학칙", {"hakchik"}),
    ("학사규정", {"haksa"}),
    ("조대", {"univ"}),
]

PHASE1_SYNONYMS: dict[str, str] = {"조대": "조선대학교"}
HARMFUL_SYNONYMS: dict[str, str] = {**PHASE1_SYNONYMS, "학칙": "학사규정"}
NO_SYNONYMS: dict[str, str] = {}

SCENARIOS: dict[str, dict[str, str]] = {
    "phase1": PHASE1_SYNONYMS,
    "harmful": HARMFUL_SYNONYMS,
    "none": NO_SYNONYMS,
}


def _build_index(synonyms: dict[str, str]) -> tuple[BM25Tokenizer, BM25PlusIndex, list[str]]:
    tokenizer = BM25Tokenizer()
    tokenizer.synonyms = dict(synonyms)  # 파일 대신 직접 주입
    doc_ids = [doc_id for doc_id, _ in DOCS]
    tokenized = [tokenizer.tokenize(text) for _, text in DOCS]
    index = BM25PlusIndex()
    index.build(tokenized)
    return tokenizer, index, doc_ids


def ranked_ids(synonyms: dict[str, str], query: str) -> list[str]:
    """질의에 대해 점수 > 0 인 문서를 점수 내림차순으로 반환한다."""
    tokenizer, index, doc_ids = _build_index(synonyms)
    q_tokens = tokenizer.tokenize(query)
    if not q_tokens:
        return []
    scores = index.get_scores(q_tokens)
    order = sorted(range(len(doc_ids)), key=lambda i: float(scores[i]), reverse=True)
    return [doc_ids[i] for i in order if float(scores[i]) > 0.0]


def evaluate(synonyms: dict[str, str]) -> dict:
    """시나리오별 P@1 / MRR / 질의별 랭킹을 계산한다."""
    per_query: dict[str, list[str]] = {}
    rr_sum = 0.0
    hits_at_1 = 0
    for query, relevant in QUERIES:
        ranked = ranked_ids(synonyms, query)
        per_query[query] = ranked
        if ranked and ranked[0] in relevant:
            hits_at_1 += 1
        for rank, doc_id in enumerate(ranked, start=1):
            if doc_id in relevant:
                rr_sum += 1.0 / rank
                break
    n = len(QUERIES)
    return {"p_at_1": hits_at_1 / n, "mrr": rr_sum / n, "per_query": per_query}


def confusion_docs(synonyms: dict[str, str], query: str, relevant: set[str]) -> list[str]:
    """질의에서 점수가 잡히지만 관련 문서가 아닌(=출처 혼동) doc_id 목록."""
    return [doc_id for doc_id in ranked_ids(synonyms, query) if doc_id not in relevant]


def main() -> None:
    print("=" * 64)
    print("  동의어 과적합 어블레이션 리포트 (BM25, 모델 불필요)")
    print("=" * 64)
    for name, syn in SCENARIOS.items():
        result = evaluate(syn)
        print(f"\n[{name}] synonyms={syn or '{}'}")
        print(f"  P@1={result['p_at_1']:.2f}  MRR={result['mrr']:.2f}")
        for query, ranked in result["per_query"].items():
            print(f"    질의 '{query}': {ranked or '(검색 결과 없음)'}")
    # 핵심 결론
    print("\n" + "-" * 64)
    harmful_conf = confusion_docs(HARMFUL_SYNONYMS, "학사규정", {"haksa"})
    print(f"harmful: '학사규정' 질의 출처 혼동 문서 = {harmful_conf}  (학칙 문서가 끼어들면 과적합 폐해)")
    print(f"none:    '조대' 질의 결과 = {ranked_ids(NO_SYNONYMS, '조대') or '(없음)'}  (조대 동의어 제거 시 검색 불가)")
    print(f"phase1:  '학사규정' 질의 = {ranked_ids(PHASE1_SYNONYMS, '학사규정')}  (출처 보존)")


if __name__ == "__main__":
    main()
