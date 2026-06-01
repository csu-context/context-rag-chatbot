#!/usr/bin/env python3
"""Issue 34: sLLM 모델 교체 Ragas 품질 검증 벤치마크.

Faithfulness·Answer Relevancy 전후 비교 실행 스크립트.
사용법:
    python scripts/run_ragas_eval.py --model eeve-korean:10.8b --samples 20
    python scripts/run_ragas_eval.py --compare  # 현재 모델 vs 이전 결과 비교
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("ALLOW_EXTERNAL_API", "true")


def run_eval(model_name: str, samples: int, output_path: Path) -> dict:
    """RAG 파이프라인 Ragas 평가 실행."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    from src.core.chains import get_rag_chain
    from src.core.retriever import RetrieverFactory
    from src.utils.logger import setup_global_logging
    from src.vector_db.chroma_manager import ChromaDBManager

    setup_global_logging()

    print(f"[Ragas Eval] 모델: {model_name}, 샘플: {samples}")

    # 리트리버/체인 초기화
    db = ChromaDBManager()
    retriever = RetrieverFactory.create_retriever(chroma_manager=db)
    chain = get_rag_chain(retriever)

    # 평가 데이터셋 로드 (eval/dataset 디렉토리 또는 자동 생성)
    dataset_path = ROOT / "data" / "eval_dataset.json"
    if not dataset_path.exists():
        print(f"평가 데이터셋이 없습니다: {dataset_path}")
        print("scripts/generate_eval_dataset.py를 먼저 실행하세요.")
        sys.exit(1)

    with open(dataset_path, encoding="utf-8") as f:
        eval_data = json.load(f)[:samples]

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in eval_data:
        q = item["question"]
        gt = item.get("ground_truth", "")
        questions.append(q)
        ground_truths.append(gt)

        result_list = []
        full_answer = ""
        for step in chain.stream({"question": q, "k": 20, "final_k": 5, "history": []}):
            if step.get("stage") == "citation" and step.get("status") == "complete":
                docs = step.get("source_documents", [])
                result_list = [d.page_content if hasattr(d, "page_content") else d.get("content", "") for d in docs]
            if step.get("stage") == "generation" and step.get("status") == "streaming":
                full_answer += step.get("output", "")

        answers.append(full_answer)
        contexts.append(result_list or [""])

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    scores = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "samples": samples,
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "context_precision": float(result["context_precision"]),
        "context_recall": float(result["context_recall"]),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    print(f"\n=== Ragas 평가 결과 ({model_name}) ===")
    for k, v in scores.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    print(f"결과 저장: {output_path}")
    return scores


def compare_results(results_dir: Path) -> None:
    """평가 결과 파일들을 비교 출력."""
    files = sorted(results_dir.glob("ragas_eval_*.json"), key=lambda p: p.stat().st_mtime)
    if len(files) < 2:
        print("비교할 결과 파일이 2개 이상 필요합니다.")
        return

    print("\n=== 모델 비교 ===")
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    print(f"{'모델':<30} " + " ".join(f"{m:<20}" for m in metrics))
    print("-" * (30 + 21 * len(metrics)))

    for f in files[-5:]:  # 최근 5개만 비교
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
        vals = " ".join(f"{data.get(m, 0):<20.4f}" for m in metrics)
        print(f"{data.get('model', f.stem):<30} {vals}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ragas RAG 평가 벤치마크")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "llama3.2:1b"), help="평가 모델명")
    parser.add_argument("--samples", type=int, default=int(os.getenv("EVAL_MAX_SAMPLES", "20")))
    parser.add_argument("--compare", action="store_true", help="기존 결과 비교 출력")
    args = parser.parse_args()

    results_dir = ROOT / "logs" / "eval"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = results_dir / f"ragas_eval_{args.model.replace(':', '_')}_{timestamp}.json"

    if args.compare:
        compare_results(results_dir)
    else:
        run_eval(args.model, args.samples, output)
