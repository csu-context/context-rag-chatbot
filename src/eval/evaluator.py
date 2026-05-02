import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import RunConfig, evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.core.chains import get_rag_chain
from src.utils.paths import BASE_DIR
from src.vector_db.chroma_manager import ChromaDBManager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class RagasEvaluator:
    def __init__(
        self, eval_model_name: str = "gemini-2.0-flash", embedding_model_name: str = "models/gemini-embedding-001"
    ):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY가 설정되어 있지 않습니다.")

        # 평가를 위한 LLM 및 임베딩 설정
        self.eval_llm = ChatGoogleGenerativeAI(model=eval_model_name, google_api_key=self.api_key, temperature=0)
        self.eval_embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model_name, google_api_key=self.api_key)

        # RAG 파이프라인 초기화
        self.db_manager = ChromaDBManager()
        self.rag_chain = get_rag_chain(self.db_manager)

    def _handle_api_error(self, error: Exception, context: str):
        error_msg = str(error)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            print(f"\n[!] API 할당량 초과 알림 ({context})")
            print("-" * 60)
            print(f"상세 내용: {error_msg}")
            print("\n분석:")
            if "limit: 20" in error_msg:
                print("- 일일 요청 제한(Daily Quota) 도달 가능성 높음.")
            else:
                print("- 분당 요청 제한(RPM) 도달 가능성 높음.")
            print("-" * 60)
        else:
            logger.error(f"{context} 중 예기치 않은 에러 발생: {error}")

    def prepare_dataset(self, golden_dataset_path: Path, max_samples: int = 50, delay: float = 2.0) -> Dataset:
        """
        Golden dataset을 로드하고 RAG 시스템의 실제 응답을 생성하여 평가용 데이터셋을 구축합니다.
        """
        with open(golden_dataset_path, encoding="utf-8") as f:
            golden_data = json.load(f)

        if len(golden_data) > max_samples:
            logger.info(f"데이터셋 중 상위 {max_samples}개 샘플만 사용합니다.")
            golden_data = golden_data[:max_samples]

        questions = []
        answers = []
        contexts = []
        ground_truths = []

        logger.info(f"{len(golden_data)}개의 샘플에 대해 RAG 응답을 생성합니다. (지연 시간: {delay}s)")

        for item in golden_data:
            question = item["question"]
            ground_truth = item["ground_truth"]

            if delay > 0:
                time.sleep(delay)

            try:
                # 1. 문서 검색 (k=5)
                search_results = self.db_manager.search(query_text=question, k=5)
                retrieved_contexts = [res["content"] for res in search_results]

                # 2. 실제 RAG 답변 생성
                response_text = self.rag_chain.invoke({"question": question, "k": 5})

                if "\n\n출처:" in response_text:
                    response_text = response_text.split("\n\n출처:")[0].strip()

                questions.append(question)
                answers.append(response_text)
                contexts.append(retrieved_contexts)
                ground_truths.append(ground_truth)

                logger.info(f"성공: {question[:30]}...")
            except Exception as e:
                self._handle_api_error(e, f"샘플 처리 실패: {question[:20]}...")
                continue

        data = {"question": questions, "answer": answers, "contexts": contexts, "ground_truth": ground_truths}
        return Dataset.from_dict(data)

    def run_evaluation(self, dataset: Dataset, max_workers: int = 1):
        """RAGAS 평가를 수행합니다."""
        if len(dataset) == 0:
            logger.warning("평가할 데이터가 없습니다.")
            return None

        logger.info(f"RAGAS 평가를 시작합니다. (병렬 작업 수: {max_workers})")

        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

        try:
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self.eval_llm,
                embeddings=self.eval_embeddings,
                run_config=RunConfig(max_workers=max_workers, timeout=120),
            )
            return result
        except Exception as e:
            self._handle_api_error(e, "RAGAS 지표 계산 단계")
            return None

    def save_results(self, result, output_dir: Path):
        """평가 결과를 JSON 및 CSV로 저장합니다."""
        if result is None:
            logger.error("저장할 평가 결과(result)가 없습니다.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 요약 결과 저장 (EvaluationResult를 딕셔너리로 변환)
        summary_path = output_dir / f"eval_summary_{timestamp}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(dict(result), f, ensure_ascii=False, indent=4)

        # 2. 상세 결과 저장 (CSV)
        df = result.to_pandas()
        csv_path = output_dir / f"eval_details_{timestamp}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        logger.info(f"평가 결과 저장 완료: {output_dir}")
        print("\n" + "=" * 30)
        print("   평가 요약 (Evaluation Summary)")
        print("=" * 30)
        print(result)
        print("=" * 30)


if __name__ == "__main__":
    golden_path = BASE_DIR / "data" / "eval" / "golden_dataset.json"

    if not golden_path.exists():
        logger.error(f"Golden dataset이 없습니다: {golden_path}")
        print("\n[힌트] python src/eval/dataset_generator.py를 먼저 실행하여 데이터셋을 생성하세요.")
    else:
        # 설정값 (환경 변수로 관리 권장)
        EVAL_SAMPLES = int(os.getenv("EVAL_MAX_SAMPLES", 50))
        EVAL_DELAY = float(os.getenv("EVAL_DELAY", 2.0))

        evaluator = RagasEvaluator()

        # 1. 데이터셋 준비 (RAG 응답 생성)
        dataset = evaluator.prepare_dataset(golden_path, max_samples=EVAL_SAMPLES, delay=EVAL_DELAY)

        if dataset and len(dataset) > 0:
            # 2. RAGAS 평가 수행
            result = evaluator.run_evaluation(dataset)
            # 3. 결과 저장
            if result:
                evaluator.save_results(result, BASE_DIR / "logs" / "eval")
            else:
                print("평가 지표 계산에 실패했습니다.")
        else:
            print("평가할 데이터셋이 비어있거나 생성에 실패했습니다.")
