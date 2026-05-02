import json
import logging
import os
import random
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from tqdm import tqdm

from src.utils.paths import BASE_DIR

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
    def __init__(self, model_name: str = "gemini-flash-latest"):
        api_key = os.getenv("GOOGLE_API_KEY")
        self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.2)
        self.prompt = ChatPromptTemplate.from_template(GENERATION_PROMPT)
        self.chain = self.prompt | self.llm

    def load_processed_data(self, file_path: Path) -> list[dict[str, Any]]:
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)

    def generate_pair(self, context: str) -> dict[str, str] | None:
        try:
            response = self.chain.invoke({"context": context})
            # content가 리스트인 경우 (Gemini 멀티모달 응답 등) 처리
            if isinstance(response.content, list):
                text_parts = [
                    part.get("text", "") if isinstance(part, dict) else str(part) for part in response.content
                ]
                content = "".join(text_parts)
            else:
                content = str(response.content)

            # JSON 파싱 시도
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            return json.loads(content)
        except Exception as e:
            logger.error(f"데이터 생성 중 오류 발생: {e}")
            return None

    def run(self, input_file: Path, output_file: Path, num_samples: int = 50):
        data = self.load_processed_data(input_file)

        # Child chunks만 평평하게 추출
        all_chunks = []
        for parent in data:
            for child in parent.get("children", []):
                all_chunks.append(child["text"])

        # 중복 제거 및 너무 짧은 텍스트 제외
        all_chunks = list(set([c for c in all_chunks if len(c) > 100]))

        if len(all_chunks) < num_samples:
            logger.warning(f"사용 가능한 청크({len(all_chunks)})가 요청된 샘플 수({num_samples})보다 적습니다.")
            num_samples = len(all_chunks)

        samples = random.sample(all_chunks, num_samples)

        golden_dataset = []
        logger.info(f"{num_samples}개의 평가 데이터 생성을 시작합니다.")

        for context in tqdm(samples):
            pair = self.generate_pair(context)
            if pair:
                golden_dataset.append(pair)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(golden_dataset, f, ensure_ascii=False, indent=4)

        logger.info(f"성공적으로 {len(golden_dataset)}개의 데이터를 {output_file}에 저장했습니다.")


if __name__ == "__main__":
    # 가장 최근의 전처리 파일 찾기
    processed_dir = BASE_DIR / "data" / "processed"
    json_files = sorted(processed_dir.glob("preprocessed_v1_*.json"), key=os.path.getmtime, reverse=True)

    if not json_files:
        print("전처리된 JSON 파일을 찾을 수 없습니다.")
    else:
        latest_file = json_files[0]
        output_path = BASE_DIR / "data" / "eval" / "golden_dataset.json"

        generator = GoldenDatasetGenerator()
        # 실제 목표인 50개 생성
        generator.run(latest_file, output_path, num_samples=50)
