import json
import logging
import random
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

from src.common.config import settings
from src.models.factory import LLMFactory
from src.utils.logger import setup_global_logging
from src.utils.paths import EVAL_DATA_DIR, PROCESSED_DATA_DIR

setup_global_logging()
logger = logging.getLogger(__name__)

load_dotenv(override=False)

GENERATION_PROMPT = """
당신은 RAG(Retrieval-Augmented Generation) 시스템의 평가를 위한 정답 데이터셋(Golden Dataset) 제작 전문가입니다.
제공된 [문서 내용]을 바탕으로, 이 내용을 완벽하게 설명하거나 질문할 수 있는
한국어 질문과 그에 대한 정확한 정답을 작성하세요.

[문서 내용]
{context}

[작성 규칙]
1. 질문(question): 문서의 핵심 정보를 묻는 구체적인 질문이어야 합니다.
2. 정답(ground_truth): 문서 내용을 바탕으로 작성된 정확하고 상세한 답변이어야 합니다.
3. 컨텍스트(context): 제공된 문서 내용을 그대로 유지하세요.
4. 출력 형식: 반드시 아래의 JSON 형식을 지켜주세요. 다른 설명은 하지 마세요.

{{
    "question": "질문 내용",
    "ground_truth": "정확한 정답 내용",
    "context": "문서 내용"
}}
"""


class GoldenDatasetGenerator:
    def __init__(self, model_type: str = "claude", model_name: str = "claude-haiku-4-5"):
        self.llm_instance = LLMFactory.create_llm(model_type=model_type, model_name=model_name, temperature=0.3)
        if hasattr(self.llm_instance, "warmup"):
            self.llm_instance.warmup()
        self.prompt_template = ChatPromptTemplate.from_template(GENERATION_PROMPT)

    @staticmethod
    def load_processed_data(file_path: Path) -> list[dict[str, Any]]:
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _extract_json(content: Any) -> dict[str, str] | None:
        """LLM 응답에서 JSON 파싱."""
        try:
            if isinstance(content, list):
                content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
            else:
                content = str(content)

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            return json.loads(content)
        except Exception:
            return None

    def run(self, input_files: list[Path], output_file: Path, num_samples: int = 50):
        chunk_pool: list[tuple[str, str, str]] = []
        for input_file in input_files:
            data = self.load_processed_data(input_file)
            chunk_pool += [
                (
                    child["text"],
                    parent.get("metadata", {}).get("relative_path", "unknown"),
                    parent.get("metadata", {}).get("source_id", "unknown"),
                )
                for parent in data
                for child in parent.get("children", [])
                if len(child["text"]) > 150
            ]
        chunk_pool = [
            (text, src, sid)
            for text, src, sid in chunk_pool
            if sum(1 for ch in text if "가" <= ch <= "힣") / len(text) > 0.3
        ]
        chunk_pool = list({text: (text, src, sid) for text, src, sid in chunk_pool}.values())

        if not chunk_pool:
            logger.error("유효한 청크가 없습니다. 필터 조건을 확인하세요.")
            return

        num_samples = min(num_samples, len(chunk_pool))
        samples = random.sample(chunk_pool, num_samples)

        golden_dataset = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0

        logger.info(f"{num_samples}개의 평가 데이터 생성을 시작합니다.")

        for context, source, source_id in tqdm(samples):
            try:
                prompt_val = self.prompt_template.invoke({"context": context})
                response_obj = self.llm_instance.invoke(prompt_val)
                pair = self._extract_json(response_obj.content)

                if pair:
                    pair["source"] = source
                    pair["source_id"] = source_id
                    golden_dataset.append(pair)
                    total_cost += response_obj.cost
                    if response_obj.usage:
                        total_input_tokens += response_obj.usage.get("input_tokens", 0)
                        total_output_tokens += response_obj.usage.get("output_tokens", 0)
            except Exception as e:
                logger.error(f"데이터 생성 중 오류 발생: {e}")

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(golden_dataset, f, ensure_ascii=False, indent=4)

        logger.info(f"데이터 저장 완료: {len(golden_dataset)}개 → {output_file}")
        logger.info(f"모델: {self.llm_instance.model_name}")
        logger.info(f"입력 토큰: {total_input_tokens:,} / 출력 토큰: {total_output_tokens:,}")
        logger.info(f"예상 비용: ${total_cost:.4f}")


if __name__ == "__main__":
    json_files = [f for f in PROCESSED_DATA_DIR.glob("*.json") if f.name != "manifest.json"]

    if not json_files:
        logger.error("전처리된 JSON 파일을 찾을 수 없습니다.")
    else:
        logger.info(f"대상 파일 {len(json_files)}개: {[f.name for f in json_files]}")
        generator = GoldenDatasetGenerator(
            model_type=settings.EVAL_DATA_GEN_TYPE,
            model_name=settings.EVAL_DATA_GEN_MODEL,
        )
        generator.run(
            input_files=json_files,
            output_file=EVAL_DATA_DIR / "synthetic_dataset_50.json",
            num_samples=settings.EVAL_DATA_GEN_SAMPLES,
        )
