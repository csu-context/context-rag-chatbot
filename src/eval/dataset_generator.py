import json
import logging
import os
import random
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

from src.models.factory import LLMFactory
from src.utils.paths import EVAL_DATA_DIR, PROCESSED_DATA_DIR

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

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
        # 비용 효율을 위해 Haiku 모델 사용 (기존: claude-sonnet-4-6)
        self.llm_instance = LLMFactory().get_model(model_type=model_type, model_name=model_name, temperature=0.3)
        self.prompt = ChatPromptTemplate.from_template(GENERATION_PROMPT)
        # LLMFactory 인스턴스에서 LangChain 모델 객체 추출
        self.llm = self.llm_instance.get_model()
        self.chain = self.prompt | self.llm

    def load_processed_data(self, file_path: Path) -> list[dict[str, Any]]:
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)

    def generate_pair(self, context: str) -> dict[str, str] | None:
        try:
            response = self.chain.invoke({"context": context})
            content = response.content

            # content가 리스트인 경우 처리
            if isinstance(content, list):
                text_parts = [part.get("text", "") if isinstance(part, dict) else str(part) for part in content]
                content = "".join(text_parts)
            else:
                content = str(content)

            # JSON 파싱 시도
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            return json.loads(content)
        except Exception as e:
            logger.error(f"데이터 생성 중 오류 발생: {e}")
            return None

    def run(self, input_file: Path, output_file: Path, num_samples: int = 20):
        data = self.load_processed_data(input_file)

        # ... (기존 필터링 로직)
        all_chunks = []
        for parent in data:
            for child in parent.get("children", []):
                all_chunks.append(child["text"])

        all_chunks = list(set([c for c in all_chunks if len(c) > 150]))
        filtered_chunks = [c for c in all_chunks if len([char for char in c if "가" <= char <= "힣"]) / len(c) > 0.3]
        all_chunks = filtered_chunks

        if len(all_chunks) < num_samples:
            num_samples = len(all_chunks)

        samples = random.sample(all_chunks, num_samples)

        golden_dataset = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0

        logger.info(f"{num_samples}개의 평가 데이터 생성을 시작합니다.")

        for context in tqdm(samples):
            try:
                # LLMFactory 인스턴스의 invoke를 직접 호출하여 토큰 정보를 가져오기 위해 로직 약간 수정
                prompt_template = ChatPromptTemplate.from_template(GENERATION_PROMPT)
                prompt_val = prompt_template.invoke({"context": context})

                response_obj = self.llm_instance.invoke(prompt_val)
                pair = self.generate_pair_from_response(response_obj)

                if pair:
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

        logger.info(f"성공적으로 {len(golden_dataset)}개의 데이터를 저장했습니다.")
        print("\n" + "=" * 40)
        print("   데이터 생성 비용 요약 (Estimated Cost)")
        print("-" * 40)
        print(f"- 사용 모델: {self.llm_instance.model_name}")
        print(f"- 총 입력 토큰: {total_input_tokens:,}")
        print(f"- 총 출력 토큰: {total_output_tokens:,}")
        print(f"- 예상 합계 비용: ${total_cost:.4f}")
        print("=" * 40)

    def generate_pair_from_response(self, response_obj) -> dict[str, str] | None:
        """LLMResponse 객체에서 정답 쌍을 파싱합니다."""
        try:
            content = response_obj.content
            if isinstance(content, list):
                content = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in content])

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            return json.loads(content)
        except Exception:
            return None


if __name__ == "__main__":
    # 가장 최근의 전처리 파일 찾기
    processed_dir = PROCESSED_DATA_DIR
    json_files = sorted(processed_dir.glob("preprocessed_*.json"), key=os.path.getmtime, reverse=True)

    if not json_files:
        print("전처리된 JSON 파일을 찾을 수 없습니다.")
    else:
        latest_file = json_files[0]
        # 최신 파싱 결과를 반영한 신규 합성 데이터셋
        output_path = EVAL_DATA_DIR / "synthetic_dataset_50.json"

        # 데이터셋 생성용 모델 설정 (기본값: Claude)
        gen_type = os.getenv("EVAL_DATA_GEN_TYPE", "claude")
        gen_model = os.getenv("EVAL_DATA_GEN_MODEL", "claude-haiku-4-5")

        generator = GoldenDatasetGenerator(model_type=gen_type, model_name=gen_model)
        # 50개 샘플 생성
        generator.run(latest_file, output_path, num_samples=50)
