#!/usr/bin/env python3
"""Issue 34: run_ragas_eval.py 실행에 필요한 평가 데이터셋 자동 생성.

data/processed/ 의 모든 source JSON 파일에서 청크를 수집하고
GoldenDatasetGenerator(LLM-as-judge)를 통해 Q&A 쌍을 생성합니다.

사용법:
    python scripts/generate_eval_dataset.py
    python scripts/generate_eval_dataset.py --samples 50 --model claude-haiku-4-5
    python scripts/generate_eval_dataset.py --append  # 기존 데이터셋에 추가
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("ALLOW_EXTERNAL_API", "true")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def collect_chunks(processed_dir: Path, min_len: int = 150, min_korean_ratio: float = 0.3) -> list[str]:
    """data/processed/ 하위 모든 source JSON에서 텍스트 청크 수집."""
    chunks: list[str] = []
    json_files = [f for f in processed_dir.glob("*.json") if f.name != "manifest.json"]

    if not json_files:
        logger.error(f"processed 디렉토리에 JSON 파일이 없습니다: {processed_dir}")
        return chunks

    logger.info(f"{len(json_files)}개 source JSON 파일 스캔 중...")

    for jf in json_files:
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            for parent in data:
                for child in parent.get("children", []):
                    text = child.get("text", "")
                    if len(text) < min_len:
                        continue
                    # 한글 비율 필터
                    korean_chars = sum(1 for c in text if "가" <= c <= "힣")
                    if len(text) > 0 and korean_chars / len(text) >= min_korean_ratio:
                        chunks.append(text)
        except Exception as e:
            logger.warning(f"파일 읽기 실패 ({jf.name}): {e}")

    unique_chunks = list(set(chunks))
    logger.info(f"총 {len(unique_chunks)}개 고유 청크 수집 완료")
    return unique_chunks


def generate_dataset(
    samples: int,
    model_type: str,
    model_name: str,
    output_path: Path,
    append: bool = False,
) -> None:
    from src.eval.dataset_generator import GoldenDatasetGenerator
    from src.utils.paths import PROCESSED_DATA_DIR

    chunks = collect_chunks(PROCESSED_DATA_DIR)
    if not chunks:
        logger.error("생성 가능한 청크가 없습니다. 먼저 문서를 인제스트하세요.")
        sys.exit(1)

    samples = min(samples, len(chunks))
    selected = random.sample(chunks, samples)
    logger.info(f"샘플 {samples}개 선택 완료. LLM으로 Q&A 생성 시작...")

    generator = GoldenDatasetGenerator(model_type=model_type, model_name=model_name)

    existing: list[dict] = []
    if append and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)
        logger.info(f"기존 데이터셋 {len(existing)}개 항목 로드 완료 (추가 모드)")

    golden: list[dict] = []
    total_cost = 0.0
    skipped = 0

    for i, context in enumerate(selected, 1):
        logger.info(f"[{i}/{samples}] Q&A 생성 중...")
        try:
            from langchain_core.prompts import ChatPromptTemplate

            from src.eval.dataset_generator import GENERATION_PROMPT

            prompt_template = ChatPromptTemplate.from_template(GENERATION_PROMPT)
            prompt_val = prompt_template.invoke({"context": context})
            response_obj = generator.llm_instance.invoke(prompt_val)
            total_cost += response_obj.cost
            pair = generator.generate_pair_from_response(response_obj)
            if pair:
                golden.append(pair)
            else:
                skipped += 1
        except Exception as e:
            logger.warning(f"생성 실패 (스킵): {e}")
            skipped += 1

    final_dataset = existing + golden
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 50}")
    print("  평가 데이터셋 생성 완료")
    print(f"  - 신규 생성: {len(golden)}개 / 스킵: {skipped}개")
    print(f"  - 전체 데이터셋: {len(final_dataset)}개")
    print(f"  - 예상 비용: ${total_cost:.4f} USD")
    print(f"  - 저장 경로: {output_path}")
    print(f"{'=' * 50}\n")
    print("이제 다음 명령으로 Ragas 평가를 실행할 수 있습니다:")
    print(f"  python scripts/run_ragas_eval.py --samples {len(final_dataset)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ragas 평가용 데이터셋 자동 생성")
    parser.add_argument(
        "--samples",
        type=int,
        default=int(os.getenv("EVAL_MAX_SAMPLES", "20")),
        help="생성할 Q&A 쌍 수 (기본: EVAL_MAX_SAMPLES 환경변수)",
    )
    parser.add_argument(
        "--model", default=os.getenv("EVAL_DATA_GEN_MODEL", "claude-haiku-4-5"), help="데이터 생성에 사용할 LLM 모델명"
    )
    parser.add_argument(
        "--model-type", default=os.getenv("EVAL_DATA_GEN_TYPE", "claude"), help="모델 타입 (claude/gemini/ollama)"
    )
    parser.add_argument("--output", default=str(ROOT / "data" / "eval_dataset.json"), help="출력 JSON 경로")
    parser.add_argument("--append", action="store_true", help="기존 데이터셋에 추가 (덮어쓰기 금지)")
    args = parser.parse_args()

    generate_dataset(
        samples=args.samples,
        model_type=args.model_type,
        model_name=args.model,
        output_path=Path(args.output),
        append=args.append,
    )
