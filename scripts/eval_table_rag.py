"""표 골든셋 qa로 RAG end-to-end Answer Correctness를 채점한다.

셀 채점(eval_table_golden.py)이 파싱 출력만 본다면, 이 스크립트는 실제 검색→생성 경로를
본다(파싱→인덱싱→검색→reranking→생성의 합성 결과). 정답이 단어 단위라 생성 답변에 정답
문자열이 포함되는지로 느슨하게 채점한다(LLM judge 없이 빠르게).

전제(중요):
- 대상 문서(조선대학교_학칙/학사규정)가 이미 인덱싱돼 있어야 한다(파이프라인 동기화 완료).
  인덱싱이 없으면 검색이 비어 전부 오답으로 나온다.
- 생성 LLM 접근이 필요하다(MODEL_TYPE/모델 설정 및 키).

실행: PYTHONIOENCODING=utf-8 python scripts/eval_table_rag.py
"""

import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

from src.core.chains import get_rag_chain
from src.core.retriever import RetrieverFactory

GOLDEN_DIR = "data/eval/golden"


def _norm(s: str) -> str:
    return re.sub(r"[\s()\-·‧,]", "", s or "")


def rag_answer(chain, question: str) -> str:
    """RAG 체인을 스트리밍 실행해 생성 단계 출력만 모은다."""
    out = ""
    for step in chain.stream({"question": question}):
        if step.get("stage") == "generation" and step.get("status") == "streaming":
            out += step.get("output", "")
    return out


def main() -> None:
    retriever = RetrieverFactory.create_retriever()
    chain = get_rag_chain(retriever, use_cache=False)  # 평가 격리: 시맨틱 캐시 우회
    correct = total = 0
    misses = []
    for path in sorted(glob.glob(f"{GOLDEN_DIR}/*.json")):
        gold = json.loads(Path(path).read_text(encoding="utf-8"))
        for qa in gold["qa"]:
            total += 1
            answer = rag_answer(chain, qa["question"])
            if _norm(qa["answer"]) in _norm(answer):
                correct += 1
            else:
                misses.append((qa["question"], qa["answer"], answer[:70]))
    print("=== RAG end-to-end Answer Correctness (느슨/substring) ===")
    print(f"  {correct}/{total} = {correct / total * 100:.1f}%")
    for q, a, got in misses[:12]:
        print(f"  MISS Q: {q[:42]} | 정답: {a} | 답변: {got}")


if __name__ == "__main__":
    main()
