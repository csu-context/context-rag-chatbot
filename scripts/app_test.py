"""
Streamlit 앱 구동 테스트 스크립트
RAG 파이프라인 전 과정 (Retrieval -> Reranking -> Generation) 검증
"""
import asyncio
import sys
import time
from dotenv import load_dotenv

# Windows cp949 환경에서 이모지 등 유니코드 출력 처리
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

load_dotenv()

from src.core.chains import get_rag_chain
from src.core.retriever import EnsembleRetriever
from src.vector_db.chroma_manager import ChromaDBManager
from src.vector_db.bm25_manager import BM25Manager

# ── 초기화 ──────────────────────────────────────────────────
print("=" * 60)
print("  RAG 앱 구동 테스트")
print("=" * 60)

db      = ChromaDBManager(collection_name="rag_collection")
bm25    = BM25Manager()
retriever = EnsembleRetriever(chroma_manager=db, bm25_manager=bm25)
rag_chain = get_rag_chain(retriever)

print(f"  ChromaDB : {db.get_count()} 청크")
print(f"  BM25     : {len(bm25.corpus_data)} docs")
print()

# ── 테스트 케이스 ────────────────────────────────────────────
TEST_CASES = [
    {
        "id": "TC-01",
        "category": "기본 목적 조회",
        "query": "조선대학교 학칙의 목적이 무엇인가요?",
        "expect_keywords": ["목적", "교육"],
    },
    {
        "id": "TC-02",
        "category": "표 구조 활용 (수업 연한)",
        "query": "학부 수업 연한은 몇 년인가요?",
        "expect_keywords": ["수업연한", "년"],
    },
    {
        "id": "TC-03",
        "category": "표 구조 활용 (학점)",
        "query": "졸업하려면 몇 학점을 이수해야 하나요?",
        "expect_keywords": ["학점", "이수", "졸업"],
    },
    {
        "id": "TC-04",
        "category": "성적 평가 기준",
        "query": "성적 평가 방법과 기준을 알려주세요",
        "expect_keywords": ["성적", "평가"],
    },
    {
        "id": "TC-05",
        "category": "범위 외 질문 (거절 테스트)",
        "query": "오늘 날씨가 어때요?",
        "expect_keywords": ["범위", "불가"],
    },
    {
        "id": "TC-06",
        "category": "인사 처리",
        "query": "안녕하세요!",
        "expect_keywords": [],
    },
]

# ── 실행 ─────────────────────────────────────────────────────
results = []

for tc in TEST_CASES:
    print(f"[{tc['id']}] {tc['category']}")
    print(f"  Q: {tc['query']}")

    start = time.time()
    try:
        resp = asyncio.run(
            rag_chain.ainvoke({"question": tc["query"], "k": 10, "final_k": 3})
        )
        elapsed = time.time() - start
        answer  = resp["answer"]
        sources = resp["source_documents"]

        # 기대 키워드 포함 여부 확인
        keyword_hit = all(kw in answer for kw in tc["expect_keywords"]) if tc["expect_keywords"] else True
        status = "PASS" if keyword_hit else "WARN"

        print(f"  A: {answer[:120].replace(chr(10), ' ')}...")
        print(f"  출처 문서: {len(sources)}개 | 응답 시간: {elapsed:.2f}초 | 상태: [{status}]")

        results.append({
            "id": tc["id"],
            "status": status,
            "elapsed": elapsed,
            "answer_len": len(answer),
            "source_count": len(sources),
        })

    except Exception as e:
        elapsed = time.time() - start
        print(f"  [ERROR] {e}")
        results.append({"id": tc["id"], "status": "FAIL", "elapsed": elapsed, "error": str(e)})

    print()

# ── 최종 요약 ─────────────────────────────────────────────────
print("=" * 60)
print("  테스트 결과 요약")
print("=" * 60)
print(f"  {'ID':<8} {'상태':<6} {'응답시간':>8}  {'답변길이':>8}  {'참조문서':>6}")
print(f"  {'-'*8} {'-'*6} {'-'*8}  {'-'*8}  {'-'*6}")
for r in results:
    status = r.get("status", "FAIL")
    elapsed = r.get("elapsed", 0)
    alen = r.get("answer_len", 0)
    scount = r.get("source_count", 0)
    print(f"  {r['id']:<8} {status:<6} {elapsed:>7.2f}s  {alen:>8}자  {scount:>6}개")

pass_count = sum(1 for r in results if r["status"] in ("PASS", "WARN"))
fail_count = sum(1 for r in results if r["status"] == "FAIL")
avg_time   = sum(r.get("elapsed", 0) for r in results) / len(results)

print()
print(f"  통과: {pass_count}/{len(results)} | 실패: {fail_count} | 평균 응답시간: {avg_time:.2f}초")
print("=" * 60)
