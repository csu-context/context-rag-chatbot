import asyncio
import contextlib
import logging
import os
import sys
import time
from collections.abc import Iterator
from typing import Any

# 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.chains import get_rag_chain
from src.pipeline import PipelineOrchestrator
from src.utils.logger import setup_global_logging
from src.vector_db.chroma_manager import ChromaDBManager

# Windows 환경에서 asyncio 관련 경고 방지
if sys.platform.startswith("win"):
    with contextlib.suppress(Exception):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 로깅 설정
setup_global_logging()
logger = logging.getLogger(__name__)


def measure_ttft_and_consume(rag_chain_iterator: Iterator[dict[str, Any]]) -> float:
    """
    RAG 체인 스트림에서 첫 번째 'generation' 토큰이 나올 때까지의 시간(TTFT)을 측정하고,
    캐시 저장 등 후속 작업을 위해 나머지 스트림을 모두 소진시킵니다.
    """
    start_time = time.perf_counter()
    first_token_time = None

    for step in rag_chain_iterator:
        if step.get("stage") == "generation" and step.get("status") == "streaming" and first_token_time is None:
            first_token_time = time.perf_counter()
            # TTFT 측정을 위해 시간을 기록하지만, break하지 않고 계속 진행하여 스트림을 모두 소비

    if first_token_time:
        return first_token_time - start_time

    # 스트리밍 청크가 하나도 없는 경우 (예: 오류)
    return -1.0


def run_ttft_benchmark():
    """
    시맨틱 캐시의 TTFT 성능을 벤치마킹합니다.
    """
    logger.info("=" * 50)
    logger.info("🚀 TTFT 성능 벤치마크 시작 🚀")
    logger.info("=" * 50)

    # 1. RAG 시스템 초기화
    try:
        db_manager = ChromaDBManager(collection_name="rag_collection")
        rag_chain = get_rag_chain(db_manager)
        orchestrator = PipelineOrchestrator()
    except Exception as e:
        logger.error(f"시스템 초기화 실패: {e}")
        return

    # 2. 테스트 시작 전 캐시 초기화
    logger.info("\n--- 🧼 1. 시맨틱 캐시 초기화 ---")
    orchestrator.cache.flush()

    # 3. 테스트 질문 정의
    test_question = "졸업 학점은 얼마나 되나요?"
    identical_question = "졸업 학점은 얼마나 되나요?"
    similar_question = "졸업 학점은 얼마인지 알려주세요."

    # 4. Cache Miss 테스트
    logger.info(f"\n--- 🧪 2. 캐시 미스(Miss) 테스트 (질문: '{test_question}') ---")
    miss_iterator = rag_chain.stream({"question": test_question})
    # 이 단계에서는 스트림을 모두 소비하여 캐시에 저장되도록 합니다.
    miss_ttft = measure_ttft_and_consume(miss_iterator)
    logger.info(f"✅ TTFT (캐시 미스): {miss_ttft:.4f} 초")

    # 5. Cache Hit 테스트 (동일 질문)
    logger.info(f"\n--- 🧪 3. 캐시 적중(Hit) 테스트 (동일 질문: '{identical_question}') ---")
    hit_iterator = rag_chain.stream({"question": identical_question})
    hit_ttft = measure_ttft_and_consume(hit_iterator)
    logger.info(f"✅ TTFT (캐시 적중 - 동일 질문): {hit_ttft:.4f} 초")

    # 6. Cache Hit 테스트 (유사 질문)
    logger.info(f"\n--- 🧪 4. 캐시 적중(Hit) 테스트 (유사 질문: '{similar_question}') ---")
    similar_hit_iterator = rag_chain.stream({"question": similar_question})
    similar_hit_ttft = measure_ttft_and_consume(similar_hit_iterator)
    logger.info(f"✅ TTFT (캐시 적중 - 유사 질문): {similar_hit_ttft:.4f} 초")

    logger.info("\n" + "=" * 50)
    logger.info("📊 벤치마크 결과 요약 📊")
    logger.info(f"  - 첫 질문 (캐시 미스):      {miss_ttft:.4f} 초")
    logger.info(f"  - 반복 질문 (캐시 적중):      {hit_ttft:.4f} 초")
    logger.info(f"  - 유사 질문 (캐시 적중):      {similar_hit_ttft:.4f} 초")
    logger.info("=" * 50)


if __name__ == "__main__":
    run_ttft_benchmark()
