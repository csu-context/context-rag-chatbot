"""하이브리드 검색 병렬화 벤치마크 (Issue #131 / DoD A).

EnsembleRetriever 는 BM25 leg 와 Vector leg 를 ThreadPoolExecutor 로 병렬 실행한다.
본 스크립트는 *동일한 두 leg* 를 직렬(sequential)과 병렬(parallel)로 각각 실행해
소요 시간을 측정하고, 직렬 대비 병렬의 단축률(%)을 산출한다.
(RRF 병합·표 보조 검색 등 공통 오버헤드는 양쪽에서 동일하므로 측정에서 제외한다.)

목표(DoD): 직렬 대비 ≥ 40% 단축.

사용법:
    python scripts/benchmark_parallel_retrieval.py                # auto (실데이터 우선, 실패 시 합성)
    python scripts/benchmark_parallel_retrieval.py --mode synthetic
    python scripts/benchmark_parallel_retrieval.py --repeats 10 --n 5
    python scripts/benchmark_parallel_retrieval.py --latency-bm25 0.05 --latency-vector 0.15
"""

import argparse
import json
import logging
import os
import statistics
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Windows 콘솔(cp949)에서 이모지·한글 출력 깨짐 방지
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.WARNING)  # 검색 내부 INFO 로그 억제 (측정 잡음 방지)
logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

DEFAULT_QUERIES = [
    "졸업에 필요한 학점은 몇 학점인가요?",
    "휴학은 최대 몇 년까지 가능한가요?",
    "장학금 수혜 자격 기준을 알려주세요.",
    "전과 신청 절차는 어떻게 되나요?",
    "학사경고는 몇 회 받으면 제적되나요?",
]


class _StubManager:
    """합성 모드용: 지정한 지연만큼 sleep 후 더미 결과를 반환하는 가짜 검색 매니저.

    time.sleep 은 GIL 을 해제하므로 실제 I/O·임베딩 연산처럼 스레드 병렬화 이득이 드러난다.
    """

    def __init__(self, name: str, latency: float):
        self.name = name
        self.latency = latency

    def retrieve(self, query: str, n: int = 5, metadata_filter: dict | None = None) -> list[dict[str, Any]]:
        time.sleep(self.latency)
        return [{"content": f"{self.name}-doc{i}", "metadata": {"chunk_id": f"{self.name}-{i}"}} for i in range(n)]


def _build_retriever(mode: str, latency_bm25: float, latency_vector: float):
    """(retriever, 실제모드 여부)를 반환. real 실패 시 synthetic 으로 폴백."""
    from src.core.retriever import EnsembleRetriever

    if mode in ("real", "auto"):
        try:
            from src.vector_db.bm25_manager import BM25Manager
            from src.vector_db.chroma_manager import ChromaDBManager

            chroma = ChromaDBManager(collection_name="rag_collection")
            bm25 = BM25Manager()
            retriever = EnsembleRetriever(chroma_manager=chroma, bm25_manager=bm25)
            # 워밍업 겸 인덱스 유효성 확인 (임베딩 모델·BM25 인덱스 로드)
            retriever._get_vector_results("워밍업 질의", 3)
            retriever._get_bm25_results("워밍업 질의", 3)
            logger.info("실데이터(real) 모드로 실행합니다.")
            return retriever, True
        except Exception as e:
            if mode == "real":
                raise
            print(f"⚠️  실데이터 로드 실패 → 합성(synthetic) 모드로 폴백: {e}")

    bm25 = _StubManager("bm25", latency_bm25)
    chroma = _StubManager("vector", latency_vector)
    retriever = EnsembleRetriever(chroma_manager=chroma, bm25_manager=bm25)
    return retriever, False


def _serial_fetch(retriever, query: str, n: int) -> None:
    """두 leg 를 순차 실행 (직렬 baseline)."""
    retriever._get_bm25_results(query, n)
    retriever._get_vector_results(query, n)


def _parallel_fetch(retriever, query: str, n: int) -> None:
    """두 leg 를 ThreadPoolExecutor 로 동시 실행 (운영 코드와 동일 구조)."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_bm25 = executor.submit(retriever._get_bm25_results, query, n)
        f_vec = executor.submit(retriever._get_vector_results, query, n)
        f_bm25.result()
        f_vec.result()


def _time_call(fn: Callable, *args) -> float:
    start = time.perf_counter()
    fn(*args)
    return time.perf_counter() - start


def run_benchmark(retriever, queries: list[str], n: int, repeats: int) -> dict[str, Any]:
    per_query = []
    serial_all: list[float] = []
    parallel_all: list[float] = []

    for q in queries:
        serial_times = [_time_call(_serial_fetch, retriever, q, n) for _ in range(repeats)]
        parallel_times = [_time_call(_parallel_fetch, retriever, q, n) for _ in range(repeats)]
        s_mean = statistics.mean(serial_times)
        p_mean = statistics.mean(parallel_times)
        reduction = (s_mean - p_mean) / s_mean * 100 if s_mean > 0 else 0.0
        per_query.append(
            {
                "query": q,
                "serial_mean_s": round(s_mean, 4),
                "parallel_mean_s": round(p_mean, 4),
                "reduction_pct": round(reduction, 2),
            }
        )
        serial_all.extend(serial_times)
        parallel_all.extend(parallel_times)

    agg_serial = statistics.mean(serial_all)
    agg_parallel = statistics.mean(parallel_all)
    agg_reduction = (agg_serial - agg_parallel) / agg_serial * 100 if agg_serial > 0 else 0.0

    return {
        "per_query": per_query,
        "aggregate": {
            "serial_mean_s": round(agg_serial, 4),
            "parallel_mean_s": round(agg_parallel, 4),
            "reduction_pct": round(agg_reduction, 2),
            "samples": len(serial_all),
        },
    }


def _print_report(result: dict[str, Any], meta: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("📊 하이브리드 검색 병렬화 벤치마크 (직렬 vs 병렬)")
    print("=" * 72)
    print(f"  모드        : {meta['mode']}")
    print(f"  질의 수     : {meta['num_queries']}  /  반복: {meta['repeats']}회  /  n={meta['n']}")
    print("-" * 72)
    print(f"  {'질의':<34} {'직렬(s)':>9} {'병렬(s)':>9} {'단축률':>8}")
    print("-" * 72)
    for row in result["per_query"]:
        q = row["query"][:32]
        print(f"  {q:<34} {row['serial_mean_s']:>9.4f} {row['parallel_mean_s']:>9.4f} {row['reduction_pct']:>7.1f}%")
    agg = result["aggregate"]
    print("-" * 72)
    print(
        f"  {'전체 평균':<34} {agg['serial_mean_s']:>9.4f} {agg['parallel_mean_s']:>9.4f} {agg['reduction_pct']:>7.1f}%"
    )
    print("=" * 72)
    target_met = agg["reduction_pct"] >= 40.0
    mark = "✅ 달성" if target_met else "❌ 미달"
    print(f"  DoD 목표(≥40% 단축): {mark}  (실측 {agg['reduction_pct']:.1f}%)")
    print("=" * 72 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="하이브리드 검색 병렬화 벤치마크")
    parser.add_argument("--mode", choices=["auto", "real", "synthetic"], default="auto")
    parser.add_argument("--repeats", type=int, default=5, help="질의당 반복 측정 횟수")
    parser.add_argument("--n", type=int, default=5, help="검색 결과 개수")
    parser.add_argument("--latency-bm25", type=float, default=0.05, help="합성 모드 BM25 leg 지연(초)")
    parser.add_argument("--latency-vector", type=float, default=0.15, help="합성 모드 Vector leg 지연(초)")
    parser.add_argument("--no-save", action="store_true", help="JSON 리포트 저장 생략")
    args = parser.parse_args()

    retriever, is_real = _build_retriever(args.mode, args.latency_bm25, args.latency_vector)
    effective_mode = "real" if is_real else "synthetic"

    result = run_benchmark(retriever, DEFAULT_QUERIES, n=args.n, repeats=args.repeats)
    meta = {
        "mode": effective_mode,
        "num_queries": len(DEFAULT_QUERIES),
        "repeats": args.repeats,
        "n": args.n,
        "latency_bm25": args.latency_bm25 if not is_real else None,
        "latency_vector": args.latency_vector if not is_real else None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    _print_report(result, meta)

    if not args.no_save:
        REPORTS_DIR.mkdir(exist_ok=True)
        out = REPORTS_DIR / f"benchmark_parallel_retrieval_{int(time.time())}.json"
        out.write_text(json.dumps({"meta": meta, "result": result}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"📁 리포트 저장: {out}")

    return 0 if result["aggregate"]["reduction_pct"] >= 40.0 else 1


if __name__ == "__main__":
    sys.exit(main())
