import asyncio
import json
import logging
import os
import time
import warnings
from datetime import datetime
from typing import Any, cast, Dict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import EvaluationDataset
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.evaluation import EvaluationResult, evaluate  # type: ignore
from ragas.llms import LangchainLLMWrapper

# Metrics 임포트 (Pylance 경고 차단)
from ragas.metrics import (  # type: ignore
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from ragas.run_config import RunConfig

from src.core.prompts import RAG_SYSTEM_PROMPT
from src.models.factory import LLMFactory
from src.utils.logger import setup_global_logging
from src.utils.paths import BASE_DIR, LOGS_DIR
from src.vector_db.chroma_manager import ChromaDBManager

# 라이브러리 내부의 DeprecationWarning 및 런타임 경고 완전 차단
warnings.filterwarnings("ignore")

# 로깅 설정
setup_global_logging()
logger = logging.getLogger(__name__)

load_dotenv(override=True)

async def run_rag_inference(test_data: list[dict[str, Any]]) -> tuple[EvaluationDataset, str]:
    """RAG 추론 수행 및 Ragas EvaluationDataset 생성"""
    db_manager = ChromaDBManager()
    rag_llm_inst = LLMFactory.create_llm()
    
    samples = []
    logger.info(f"🚀 {len(test_data)}개 샘플 추론 시작 (모델: {rag_llm_inst.model_name})")
    
    for i, row in enumerate(test_data):
        question = row["question"]
        ground_truth = row["ground_truth"]
        
        start_time = time.time()
        try:
            search_results = db_manager.search(query_text=question, k=5)
            contexts = [res["content"] for res in search_results]
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", RAG_SYSTEM_PROMPT),
                ("human", "{question}")
            ])
            context_str = "\n\n".join([
                f"내용: {res['content']}\n출처: [{res['metadata'].get('src_name')}, p.{res['metadata'].get('pg_num')}]"
                for res in search_results
            ])
            prompt_val = prompt_template.invoke({"question": question, "context": context_str})
            
            response_obj = rag_llm_inst.invoke(prompt_val)
            answer = str(response_obj.content)
            
            if "\n\n출처:" in answer:
                answer = answer.split("\n\n출처:")[0].strip()
            
            latency = time.time() - start_time
            logger.info(f"[{i+1}/{len(test_data)}] 성공 ({latency:.2f}s)")
            
            samples.append({
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": ground_truth,
                "latency_sec": latency
            })
        except Exception as e:
            logger.error(f"오류 발생 ({question[:20]}...): {e}")

    return EvaluationDataset.from_list(samples), str(rag_llm_inst.model_name)

async def main():
    golden_path = BASE_DIR / "data" / "eval" / "synthetic_dataset_50.json"
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
    eval_model_name = "claude-haiku-4-5-20251001"
    
    # ChatAnthropic의 타입 체크를 완전히 우회하기 위해 Any로 캐스팅
    eval_llm = cast(Any, ChatAnthropic)(
        model=eval_model_name,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0,
        timeout=None,
        stop=None
    )  # type: ignore
    ragas_llm = LangchainLLMWrapper(eval_llm)
    
    lc_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    ragas_embeddings = LangchainEmbeddingsWrapper(lc_embeddings)
    
    # 지표 리스트 (클래스 인스턴스화)
    metrics = [
        Faithfulness(),  # type: ignore
        AnswerRelevancy(),  # type: ignore
        ContextPrecision(),  # type: ignore
        ContextRecall(),  # type: ignore
    ]
    
    logger.info(f"📊 RAGAS 지표 계산 시작 (평가 모델: {eval_model_name})")
    
    try:
        # 3. 평가 실행
        results = evaluate(  # type: ignore
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=RunConfig(max_workers=2, timeout=180)
        )
        
        # 4. 결과 집계
        eval_results = cast(EvaluationResult, results)
        df = eval_results.to_pandas()
        numeric_df = df.select_dtypes(include=['number'])
        
        # scores 처리: 모든 키와 값을 기본 타입으로 강제 변환
        avg_scores: Dict[str, float] = {}
        raw_scores = getattr(eval_results, "scores", {})
        
        if isinstance(raw_scores, dict):
            for k, v in raw_scores.items():
                avg_scores[str(k)] = float(v)
        else:
            mean_vals = numeric_df.mean().to_dict()
            for k, v in mean_vals.items():
                k_str = str(k)
                if k_str != "latency_sec":
                    avg_scores[k_str] = float(v)
            
        eval_dir = LOGS_DIR / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        summary_data: Dict[str, Any] = {
            "scores": avg_scores,
            "avg_latency_sec": float(numeric_df["latency_sec"].mean()) if "latency_sec" in numeric_df.columns else 0.0,
            "total_samples": int(len(df)),
            "model_name": str(model_name),
            "timestamp": str(timestamp)
        }
        
        summary_path = eval_dir / f"eval_summary_{timestamp}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=4)
            
        logger.info(f"✅ 완료! 요약: {summary_path}")
        logger.info(f"📊 속도: {summary_data['avg_latency_sec']:.2f}s")
        for m, s in avg_scores.items():
            logger.info(f"   - {m}: {s:.4f}")
        
    except Exception as e:
        logger.error(f"평가 실패: {e}")

if __name__ == "__main__":
    asyncio.run(main())
