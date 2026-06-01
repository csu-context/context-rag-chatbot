import asyncio
import json
import logging
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import EvaluationDataset
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.evaluation import EvaluationResult, evaluate  # type: ignore
from ragas.llms import LangchainLLMWrapper

# Metrics 임포트 (Pylance 경고 차단)
from ragas.metrics.collections import (  # type: ignore
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from ragas.run_config import RunConfig

from src.models.factory import LLMFactory
from src.utils.logger import setup_global_logging
from src.utils.paths import EVAL_DATA_DIR, EVAL_LOGS_DIR
from src.vector_db.chroma_manager import ChromaDBManager

# 평가 관련 기본 설정 상수
DEFAULT_EVAL_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MAX_WORKERS = 2
DEFAULT_EVAL_TIMEOUT = 180
DEFAULT_GOLDEN_SET_PATH = EVAL_DATA_DIR / "synthetic_dataset_50.json"

# 라이브러리 내부의 DeprecationWarning 및 런타임 경고 완전 차단
warnings.filterwarnings("ignore")

# 로깅 설정
setup_global_logging()
logger = logging.getLogger(__name__)

load_dotenv(override=True)


async def run_rag_inference(test_data: list[dict[str, Any]]) -> tuple[EvaluationDataset, str]:
    """RAG 추론 수행 및 Ragas EvaluationDataset 생성 (시스템 표준 체인 사용)"""
    from src.core.chains import get_rag_chain

    db_manager = ChromaDBManager()
    # 시스템 표준 RAG 체인 초기화
    rag_chain = get_rag_chain(db_manager)

    # 모델 정보 획득 (로깅용)
    temp_llm = LLMFactory.create_llm()
    model_name = str(temp_llm.model_name)

    samples = []
    logger.info(f"{len(test_data)}개 샘플 추론 시작 (모델: {model_name})")

    for i, row in enumerate(test_data):
        question = row["question"]
        ground_truth = row["ground_truth"]

        start_time = time.time()
        try:
            full_response = ""
            source_docs = []

            # 스트리밍 파이프라인 소비
            for step in rag_chain.stream({"question": question}):
                stage = step.get("stage")
                status = step.get("status")

                if stage == "generation" and status == "streaming":
                    full_response += step.get("output", "")
                elif stage == "citation" and status == "complete":
                    source_docs = step.get("source_documents", [])

            # Ragas 평가를 위해 검색된 문서들의 텍스트 추출 (이중 검색 제거)
            contexts = [doc.page_content for doc in source_docs]

            # 답변 본문 사용
            answer = full_response.strip()

            latency = time.time() - start_time

            logger.info(f"[{i + 1}/{len(test_data)}] 성공 ({latency:.2f}s)")

            samples.append(
                {
                    "user_input": question,
                    "response": answer,
                    "retrieved_contexts": contexts,
                    "reference": ground_truth,
                    "latency_sec": latency,
                }
            )
        except Exception as e:
            logger.error(f"오류 발생 ({question[:20]}...): {e}")

    return EvaluationDataset.from_list(samples), model_name


async def main():
    golden_path = Path(os.getenv("GOLDEN_SET_PATH", str(DEFAULT_GOLDEN_SET_PATH)))
    if not golden_path.exists():
        logger.error(f"파일 없음: {golden_path}")
        return

    with open(golden_path, encoding="utf-8") as f:
        data = json.load(f)

    # 신규 생성된 50개 샘플 전체를 평가하기 위해 기본값 수정
    max_samples = int(os.getenv("EVAL_MAX_SAMPLES", 50))
    test_data = data[:max_samples]

    # 1. RAG 추론 수행
    dataset, model_name = await run_rag_inference(test_data)

    if len(dataset) == 0:
        logger.error("유효한 추론 결과가 없습니다.")
        return

    # 2. 평가 판사 및 임베딩 설정
    eval_model_type = os.getenv("EVAL_JUDGE_TYPE", "claude")
    eval_model_name = os.getenv("EVAL_JUDGE_MODEL", DEFAULT_EVAL_MODEL)

    logger.info(f"평가 판사 설정 (Type: {eval_model_type}, Model: {eval_model_name})")

    # LLMFactory를 통해 평가 판사 인스턴스 생성
    eval_llm_inst = LLMFactory.create_llm(model_type=eval_model_type, model_name=eval_model_name, temperature=0)
    ragas_llm = LangchainLLMWrapper(eval_llm_inst.get_model())

    # 임베딩 모델 설정
    embed_model_name = os.getenv("EVAL_EMBED_MODEL", "BAAI/bge-m3")
    lc_embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)
    ragas_embeddings = LangchainEmbeddingsWrapper(lc_embeddings)

    # 지표 리스트 (클래스 인스턴스화)
    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
    ]

    logger.info(f"RAGAS 지표 계산 시작 (평가 모델: {eval_model_name})")

    try:
        # 3. 평가 실행
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=RunConfig(
                max_workers=int(os.getenv("EVAL_MAX_WORKERS", DEFAULT_MAX_WORKERS)),
                timeout=int(os.getenv("EVAL_TIMEOUT", DEFAULT_EVAL_TIMEOUT)),
            ),
        )

        # 4. 결과 집계
        eval_results = cast(EvaluationResult, results)
        df = eval_results.to_pandas()
        numeric_df = df.select_dtypes(include=["number"])

        # scores 처리: 모든 키와 값을 기본 타입으로 강제 변환
        avg_scores: dict[str, float] = {}
        raw_scores = getattr(eval_results, "scores", {})

        if isinstance(raw_scores, dict):
            for k, v in raw_scores.items():
                avg_scores[str(k)] = float(v)
        else:
            # Fallback: pandas DataFrame 평균값 사용
            mean_vals = numeric_df.mean().to_dict()
            for k, v in mean_vals.items():
                k_str = str(k)
                if k_str != "latency_sec":
                    avg_scores[k_str] = float(v)

        eval_dir = EVAL_LOGS_DIR
        eval_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        summary_data: dict[str, Any] = {
            "scores": avg_scores,
            "avg_latency_sec": float(numeric_df["latency_sec"].mean()) if "latency_sec" in numeric_df.columns else 0.0,
            "total_samples": len(df),
            "model_name": str(model_name),
            "timestamp": str(timestamp),
        }

        summary_path = eval_dir / f"eval_summary_{timestamp}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=4)

        # 상세 결과(개별 샘플 점수 및 사유) CSV 저장 추가
        details_path = eval_dir / f"eval_details_{timestamp}.csv"
        df.to_csv(details_path, index=False, encoding="utf-8-sig")

        logger.info(f"평가 완료. 요약 저장: {summary_path}")
        logger.info(f"상세 내역 저장: {details_path}")
        logger.info(f"평균 응답 속도: {summary_data['avg_latency_sec']:.2f}s")
        for m, s in avg_scores.items():
            logger.info(f"   - {m}: {s:.4f}")

    except Exception as e:
        logger.error(f"평가 실패: {e}")


if __name__ == "__main__":
    asyncio.run(main())
