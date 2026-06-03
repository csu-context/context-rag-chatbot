import asyncio
import inspect
import json
import logging
import math
import os
import platform
import subprocess
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import instructor
import pandas as pd
import psutil
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from ragas import SingleTurnSample
from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings
from ragas.llms import InstructorLLM
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from src.common.config import settings
from src.core.retriever import RetrieverFactory
from src.models.factory import LLMFactory
from src.utils.logger import setup_global_logging
from src.utils.paths import EVAL_DATA_DIR, EVAL_LOGS_DIR, PROCESSED_DATA_DIR

DEFAULT_GOLDEN_SET_PATH = EVAL_DATA_DIR / "synthetic_dataset_50.json"
SECTION_LINE = "─" * 49

# 외부 라이브러리(ragas, langchain, huggingface, anthropic 등)의 Deprecation/Future 경고만 모듈 단위로 차단.
# 전역 차단(filterwarnings("ignore"))은 자체 코드의 경고까지 가리므로 사용하지 않는다.
_NOISY_MODULES = (
    "ragas",
    "langchain",
    "langchain_huggingface",
    "huggingface_hub",
    "transformers",
    "sentence_transformers",
    "anthropic",
    "instructor",
)
for _noisy_module in _NOISY_MODULES:
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=_noisy_module)
    warnings.filterwarnings("ignore", category=FutureWarning, module=_noisy_module)

# 로깅 설정
setup_global_logging()
logger = logging.getLogger(__name__)

load_dotenv(override=False)


@dataclass
class InferenceResult:
    samples: list[SingleTurnSample]
    latencies: list[float]
    resources: list[dict[str, float]]
    model_name: str


def _get_cpu_name() -> str:
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if platform.system() == "Windows":
            out = subprocess.check_output(
                ["powershell", "-Command", "(Get-CimInstance -ClassName Win32_Processor).Name"],
                text=True,
                timeout=5,
            )
            return out.strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _get_system_info() -> dict[str, Any]:
    mem = psutil.virtual_memory()
    info: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu": _get_cpu_name(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_total_gb": round(mem.total / 1024 / 1024 / 1024, 1),
    }
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=3,
        )
        name, total = out.strip().splitlines()[0].split(",")
        info["gpu"] = name.strip()
        info["gpu_vram_total_mb"] = float(total.strip())
    except Exception:
        info["gpu"] = None
        info["gpu_vram_total_mb"] = None
    return info


def _get_indexed_documents() -> list[str]:
    try:
        paths = sorted([f for f in PROCESSED_DATA_DIR.glob("*.json") if f.name != "manifest.json"])
        docs = []
        for path in paths:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if '"relative_path"' in line:
                        value = line.split('"relative_path"', 1)[1]
                        value = value.split('"', 2)
                        if len(value) >= 3:
                            docs.append(value[1])
                        break
        return sorted(set(docs))
    except Exception:
        return []


def _get_vram_mb() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=3,
        )
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def _get_resource_snapshot() -> dict[str, float]:
    mem = psutil.virtual_memory()
    snapshot: dict[str, float] = {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_mb": round(mem.used / 1024 / 1024, 1),
    }
    vram = _get_vram_mb()
    if vram is not None:
        snapshot["vram_mb"] = vram
    return snapshot


async def run_rag_inference(test_data: list[dict[str, Any]]) -> InferenceResult:
    """RAG 추론 수행 및 SingleTurnSample 리스트 생성 (시스템 표준 체인 사용)"""
    from src.core.chains import get_rag_chain

    retriever = RetrieverFactory.create_retriever()
    logger.info(SECTION_LINE)
    logger.info("파이프라인 설정")
    logger.info(
        f"  Retriever : {settings.RETRIEVER_TYPE:<8} "
        f"(BM25 w={settings.HYBRID_WEIGHT_BM25} / Vector w={settings.HYBRID_WEIGHT_VECTOR} / k={settings.RETRIEVAL_K})"
    )
    logger.info(
        f"  Reranker  : {settings.RERANKER_TYPE:<8} "
        f"({settings.RERANKER_MODEL_NAME} / top_k={settings.RERANKER_MAX_DOCS})"
    )
    logger.info(f"  LLM       : {settings.MODEL_TYPE:<8} ({settings.MODEL_NAME})")
    logger.info(SECTION_LINE)

    temp_llm = LLMFactory.create_llm()
    model_name = str(temp_llm.model_name)

    if hasattr(temp_llm, "warmup"):
        temp_llm.warmup()

    chroma_target = getattr(retriever, "chroma", retriever)
    if hasattr(chroma_target, "embedder"):
        _ = chroma_target.embedder

    rag_chain = get_rag_chain(retriever, llm=temp_llm.get_model(), use_cache=False)

    samples: list[SingleTurnSample] = []
    latencies: list[float] = []
    resources: list[dict[str, float]] = []
    logger.info(f"[추론] {len(test_data)}개 샘플 시작 (모델: {model_name})")

    for i, row in enumerate(test_data):
        question = row["question"]
        ground_truth = row["ground_truth"]

        start_time = time.time()
        try:
            full_response = ""
            source_docs = []

            for step in rag_chain.stream({"question": question}):
                stage = step.get("stage")
                status = step.get("status")

                if stage == "generation" and status == "streaming":
                    full_response += step.get("output", "")
                elif stage == "citation" and status == "complete":
                    source_docs = step.get("source_documents", [])

            contexts = [doc.page_content for doc in source_docs]
            answer = full_response.strip()
            latency = time.time() - start_time

            logger.info(f"[{i + 1}/{len(test_data)}] 성공 ({latency:.2f}s)")

            samples.append(
                SingleTurnSample(
                    user_input=question,
                    response=answer,
                    retrieved_contexts=contexts,
                    reference=ground_truth,
                )
            )
            latencies.append(latency)
            resources.append(_get_resource_snapshot())
        except Exception as e:
            logger.error(f"오류 발생 ({question[:20]}...): {e}")

    return InferenceResult(samples=samples, latencies=latencies, resources=resources, model_name=model_name)


async def _score_and_save(result: InferenceResult, missing_docs: list[str] | None = None) -> None:
    """메트릭 초기화, 샘플별 스코어링, 결과 저장"""
    logger.info(SECTION_LINE)
    logger.info("평가 판사 설정")
    logger.info(f"  Judge LLM : {settings.EVAL_JUDGE_TYPE:<8} ({settings.EVAL_JUDGE_MODEL})")
    logger.info(f"  Embedding : {settings.EMBEDDING_MODEL_NAME}")
    logger.info(SECTION_LINE)

    instructor_client = instructor.from_anthropic(AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY))
    ragas_llm = InstructorLLM(
        client=instructor_client,
        model=settings.EVAL_JUDGE_MODEL,
        provider="anthropic",
        temperature=0,
        max_tokens=8192,
    )
    ragas_llm.model_args.pop("top_p", None)
    ragas_embeddings = RagasHFEmbeddings(model=settings.EMBEDDING_MODEL_NAME)

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    sem = asyncio.Semaphore(settings.EVAL_MAX_WORKERS)
    metric_params = {m.name: set(inspect.signature(m.ascore).parameters) for m in metrics}

    async def _ascore(metric: Any, sample: SingleTurnSample) -> float:
        all_kwargs = {
            "user_input": sample.user_input or "",
            "response": sample.response or "",
            "retrieved_contexts": sample.retrieved_contexts or [],
            "reference": sample.reference or "",
        }
        kwargs = {k: v for k, v in all_kwargs.items() if k in metric_params[metric.name]}
        async with sem:
            result_obj = await metric.ascore(**kwargs)
        return float(result_obj.value)

    logger.info(f"[스코어링] {len(result.samples)}개 샘플 시작")

    scored_rows: list[dict[str, Any]] = []
    for i, (sample, latency, resource) in enumerate(
        zip(result.samples, result.latencies, result.resources, strict=False)
    ):
        try:
            scores = await asyncio.gather(*[_ascore(m, sample) for m in metrics])
            row: dict[str, Any] = {m.name: s for m, s in zip(metrics, scores, strict=False)}
        except Exception as e:
            logger.error(f"샘플 {i + 1} 스코어링 실패: {e}")
            row = {m.name: float("nan") for m in metrics}
        row["latency_sec"] = latency
        row.update(resource)
        scored_rows.append(row)

        if (i + 1) % 10 == 0 or (i + 1) == len(result.samples):
            interim_df = pd.DataFrame(scored_rows)
            interim_scores = " / ".join(
                f"{name}: {interim_df[name].mean():.4f}"
                for name in [m.name for m in metrics]
                if name in interim_df.columns
            )
            avg_latency = interim_df["latency_sec"].mean()
            logger.info(f"[{i + 1}/{len(result.samples)}] 중간 점수 — {interim_scores} / latency: {avg_latency:.2f}s")

    df = pd.DataFrame(scored_rows)
    metric_names = [m.name for m in metrics]
    avg_scores: dict[str, float] = {
        name: float(df[name].mean()) for name in metric_names if name in df.columns and not math.isnan(df[name].mean())
    }

    eval_dir = EVAL_LOGS_DIR
    eval_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    avg_latency = float(df["latency_sec"].mean()) if "latency_sec" in df.columns else 0.0
    summary_data: dict[str, Any] = {
        "timestamp": str(timestamp),
        "data": {
            "indexed_documents": _get_indexed_documents(),
            "missing_documents": missing_docs or [],
        },
        "pipeline": {
            "llm": f"{settings.MODEL_TYPE}/{result.model_name}",
            "retriever": (
                f"{settings.RETRIEVER_TYPE} (BM25 w={settings.HYBRID_WEIGHT_BM25}"
                f" / Vector w={settings.HYBRID_WEIGHT_VECTOR} / k={settings.RETRIEVAL_K})"
            ),
            "reranker": (
                f"{settings.RERANKER_TYPE}/{settings.RERANKER_MODEL_NAME} (top_k={settings.RERANKER_MAX_DOCS})"
            ),
        },
        "judge": {
            "llm": f"{settings.EVAL_JUDGE_TYPE}/{settings.EVAL_JUDGE_MODEL}",
            "embedding": settings.EMBEDDING_MODEL_NAME,
            "max_workers": settings.EVAL_MAX_WORKERS,
        },
        "results": {
            **avg_scores,
            "avg_latency_sec": avg_latency,
            "total_samples": len(df),
        },
        "system": _get_system_info(),
    }

    summary_path = eval_dir / f"eval_summary_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=4)

    details_path = eval_dir / f"eval_details_{timestamp}.csv"
    df.to_csv(details_path, index=False, encoding="utf-8-sig")

    logger.info(SECTION_LINE)
    logger.info("평가 결과")
    for m, s in avg_scores.items():
        logger.info(f"  {m:<24}: {s:.4f}")
    logger.info(f"  {'avg_latency':<24}: {avg_latency:.2f}s")
    logger.info(SECTION_LINE)
    logger.info(f"요약 저장: {summary_path}")
    logger.info(f"상세 저장: {details_path}")


async def main():
    golden_path = Path(os.getenv("GOLDEN_SET_PATH", str(DEFAULT_GOLDEN_SET_PATH)))
    if not golden_path.exists():
        logger.error(f"파일 없음: {golden_path}")
        return

    with open(golden_path, encoding="utf-8") as f:
        data = json.load(f)

    missing_docs: list[str] = []
    dataset_source_ids = {row["source_id"] for row in data if "source_id" in row}
    if dataset_source_ids:
        indexed_ids = {f.stem for f in PROCESSED_DATA_DIR.glob("*.json") if f.name != "manifest.json"}
        missing_ids = dataset_source_ids - indexed_ids
        if missing_ids:
            missing_docs = sorted({row["source"] for row in data if row.get("source_id") in missing_ids})
            logger.warning(
                f"데이터셋-인덱스 불일치 감지. 누락된 문서: {missing_docs} "
                f"(source_id: {sorted(missing_ids)}) — 평가를 계속 진행합니다."
            )

    result = await run_rag_inference(data[: settings.EVAL_MAX_SAMPLES])

    if not result.samples:
        logger.error("유효한 추론 결과가 없습니다.")
        return

    try:
        await _score_and_save(result, missing_docs=missing_docs)
    except Exception as e:
        logger.error(f"평가 실패: {e}")


if __name__ == "__main__":
    asyncio.run(main())
