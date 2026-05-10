import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from datasets import Dataset
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ragas import RunConfig, evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.models.factory import LLMFactory
from src.utils.paths import BASE_DIR
from src.vector_db.chroma_manager import ChromaDBManager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class RagasEvaluator:
    # 모델별 단가 정의 (1M 토큰 기준 USD)
    PRICING: ClassVar[dict[str, dict[str, float]]] = {
        "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
        "claude-haiku-4-5-20251001": {"input": 0.25, "output": 1.25},
        "gemini-2.0-flash": {"input": 0.1, "output": 0.4},
    }

    def __init__(self, eval_model_type: str = "claude", eval_model_name: str = "claude-haiku-4-5-20251001"):
        # 평가를 위한 LLM 설정 (Haiku)
        self.eval_llm_inst = LLMFactory.create_llm(
            model_type=eval_model_type, model_name=eval_model_name, temperature=0
        )
        self.eval_llm = self.eval_llm_inst.get_model()
        self.eval_model_name = eval_model_name

        # RAGAS는 임베딩 모델도 필요함
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.eval_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", google_api_key=self.api_key
        )

        # RAG 파이프라인 초기화
        self.db_manager = ChromaDBManager()
        # 토큰 추적을 위한 Sonnet 인스턴스
        self.rag_llm_inst = LLMFactory.create_llm(model_type="claude", model_name="claude-sonnet-4-6")

        # 비용 추적 카운터
        self.total_usage = {
            "inference": {"input": 0, "output": 0, "model": self.rag_llm_inst.model_name},
            "evaluation": {"input": 0, "output": 0, "model": self.eval_model_name},
        }
        self.total_eval_time = 0

    def _calculate_cost(self, usage_dict):
        model = usage_dict["model"]
        rates = self.PRICING.get(model, {"input": 0, "output": 0})
        cost = (usage_dict.get("input", 0) * rates["input"] / 1000000) + (
            usage_dict.get("output", 0) * rates["output"] / 1000000
        )
        return cost

    def _handle_api_error(self, error: Exception, context: str):
        error_msg = str(error)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            print(f"\n[!] API 할당량 초과 알림 ({context})")
        else:
            logger.error(f"{context} 중 에러 발생: {error}")

    def prepare_dataset(self, golden_dataset_path: Path, max_samples: int = 20, delay: float = 1.0) -> Dataset:
        """Golden dataset을 로드하고 RAG 답변을 생성하며 비용을 추적합니다."""
        with open(golden_dataset_path, encoding="utf-8") as f:
            golden_data = json.load(f)

        if len(golden_data) > max_samples:
            golden_data = golden_data[:max_samples]

        questions, answers, contexts, ground_truths = [], [], [], []
        logger.info(f"{len(golden_data)}개의 샘플에 대해 RAG 응답을 생성합니다.")

        from langchain_core.prompts import ChatPromptTemplate

        from src.core.prompts import RAG_SYSTEM_PROMPT

        start_time = time.time()
        for i, item in enumerate(golden_data):
            question = item["question"]
            ground_truth = item["ground_truth"]
            if delay > 0:
                time.sleep(delay)

            sample_start = time.time()
            try:
                # 1. 검색
                search_results = self.db_manager.search(query_text=question, k=5)
                retrieved_contexts = [res["content"] for res in search_results]

                prompt_template = ChatPromptTemplate.from_messages(
                    [("system", RAG_SYSTEM_PROMPT), ("human", "{question}")]
                )
                context_str = "\n\n".join(
                    [
                        f"내용: {res['content']}\n출처: [{res['metadata'].get('src_name')}, "
                        f"p.{res['metadata'].get('pg_num')}]"
                        for res in search_results
                    ]
                )
                prompt_val = prompt_template.invoke({"question": question, "context": context_str})

                # [중요] invoke 결과를 직접 받아 usage 추출
                response_obj = self.rag_llm_inst.invoke(prompt_val)
                response_text = response_obj.content

                if response_obj.usage:
                    self.total_usage["inference"]["input"] += response_obj.usage.get("input_tokens", 0)
                    self.total_usage["inference"]["output"] += response_obj.usage.get("output_tokens", 0)
                else:
                    # usage가 없는 경우 (Mock 등) 추정치라도 넣음 (한글 1자당 약 0.6토큰)
                    self.total_usage["inference"]["input"] += int(len(str(prompt_val)) * 0.6)
                    self.total_usage["inference"]["output"] += int(len(str(response_text)) * 0.6)

                if "\n\n출처:" in response_text:
                    response_text = response_text.split("\n\n출처:")[0].strip()

                questions.append(question)
                answers.append(response_text)
                contexts.append(retrieved_contexts)
                ground_truths.append(ground_truth)

                sample_elapsed = time.time() - sample_start
                logger.info(f"[{i + 1}/{len(golden_data)}] 성공 ({sample_elapsed:.2f}s)")
            except Exception as e:
                self._handle_api_error(e, f"샘플 처리 실패: {question[:20]}")
                continue

        total_elapsed = time.time() - start_time
        logger.info(f"RAG 응답 생성 완료 (총 {total_elapsed:.2f}s)")

        return Dataset.from_dict(
            {"question": questions, "answer": answers, "contexts": contexts, "ground_truth": ground_truths}
        )

    def run_evaluation(self, dataset: Dataset, max_workers: int = 2):
        """RAGAS 평가를 수행합니다."""
        if len(dataset) == 0:
            return None
        logger.info(f"RAGAS 평가를 시작합니다. (모델: {self.eval_model_name})")

        start_time = time.time()
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        try:
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self.eval_llm,
                embeddings=self.eval_embeddings,
                run_config=RunConfig(max_workers=max_workers, timeout=180),
            )
            self.total_eval_time = time.time() - start_time
            logger.info(f"RAGAS 지표 계산 완료 (총 소요 시간: {self.total_eval_time:.2f}s)")

            # [평가 비용 추정] Haiku 기준 샘플당 평균 1,800 토큰 발생 (Input 80%, Output 20%)
            num_samples = len(dataset)
            num_metrics = len(metrics)
            # RAGAS는 내부적으로 지표당 2~3회 호출하므로 보수적으로 계산
            self.total_usage["evaluation"]["input"] = int(num_samples * num_metrics * 2000 * 0.8)
            self.total_usage["evaluation"]["output"] = int(num_samples * num_metrics * 2000 * 0.2)

            return result
        except Exception as e:
            self._handle_api_error(e, "RAGAS 지표 계산 단계")
            return None

    def save_results(self, result, output_dir: Path):
        """평가 결과를 저장하고 상세 비용 리포트를 출력합니다."""
        if result is None:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)

        inf_cost = self._calculate_cost(self.total_usage["inference"])
        eval_cost = self._calculate_cost(self.total_usage["evaluation"])
        total_cost = inf_cost + eval_cost

        # RAGAS 결과에서 점수 추출
        try:
            scores_dict = {k: v for k, v in result.items()}
        except Exception:
            # 직접 지표별 평균 계산 (fallback)
            df = result.to_pandas()
            metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
            scores_dict = {m: df[m].mean() for m in metrics if m in df.columns}

        summary_data = {
            "scores": scores_dict,
            "usage": self.total_usage,
            "total_eval_time_sec": self.total_eval_time,
            "estimated_cost_usd": {"inference": inf_cost, "evaluation": eval_cost, "total": total_cost},
        }

        with open(output_dir / f"eval_summary_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=4)

        result.to_pandas().to_csv(output_dir / f"eval_details_{timestamp}.csv", index=False, encoding="utf-8-sig")

        print("\n" + "=" * 50)
        print("      최종 RAG 평가 및 비용 리포트")
        print("=" * 50)
        print(f"1. 추론 단계 ({self.total_usage['inference']['model']})")
        print(f"   - 비용: ${inf_cost:.4f}")
        print(f"   - 토큰: {self.total_usage['inference']['input'] + self.total_usage['inference']['output']:,} tokens")
        print("-" * 50)
        print(f"2. 평가 단계 ({self.eval_model_name} - 추정치)")
        print(f"   - 비용: ${eval_cost:.4f}")
        print(
            f"   - 토큰: {self.total_usage['evaluation']['input'] + self.total_usage['evaluation']['output']:,} tokens"
        )
        print("-" * 50)
        print("3. 합계 (Total)")
        print(f"   - 예상 총 비용: ${total_cost:.4f}")
        print(f"   - 총 소요 시간: {self.total_eval_time:.1f}초")
        print("-" * 50)
        print("4. 지표 결과 (RAGAS Scores)")
        for m, s in scores_dict.items():
            print(f"   * {m:18}: {s:.4f}")
        print("=" * 50)


if __name__ == "__main__":
    golden_path = BASE_DIR / "data" / "eval" / "golden_dataset.json"

    if not golden_path.exists():
        logger.error(f"Golden dataset이 없습니다: {golden_path}")
    else:
        EVAL_SAMPLES = int(os.getenv("EVAL_MAX_SAMPLES", 20))
        EVAL_DELAY = float(os.getenv("EVAL_DELAY", 1.0))

        evaluator = RagasEvaluator()
        dataset = evaluator.prepare_dataset(golden_path, max_samples=EVAL_SAMPLES, delay=EVAL_DELAY)

        if dataset and len(dataset) > 0:
            result = evaluator.run_evaluation(dataset)
            if result:
                evaluator.save_results(result, BASE_DIR / "logs" / "eval")
