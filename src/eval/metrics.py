from langchain_core.prompts import PromptTemplate
from ragas.metrics.collections import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

# 기본 평가 지표 설정
METRICS = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]


# ── [파서 벤치마크 평가 로직 보존] ──────────────────────────────────────────

PARSER_BENCHMARK_PROMPT = PromptTemplate(
    input_variables=["ground_truth", "parsed_result"],
    template="""당신은 문서 파싱 파이프라인의 성능을 평가하는 전문가(LLM-as-a-Judge)입니다.
원본(Ground Truth) 데이터와 파서가 추출한 결과(Parsed Result)를 비교하여,
특히 '표(Table) 구조의 보존 여부'와 '정보의 누락'에 초점을 맞추어 평가해주세요.

[원본 구조/텍스트]
{ground_truth}

[파싱된 결과]
{parsed_result}

다음 항목에 대해 1~5점으로 점수를 매기고, 짧은 이유를 설명해주세요:
1. 표 구조 보존 (Markdown 형태 등으로 행/열이 논리적으로 분리되어 있는가?)
2. 데이터 누락 (원문에 있는 수치나 핵심 텍스트가 누락되지 않았는가?)
3. 가독성 및 노이즈 (불필요한 줄바꿈이나 깨진 문자가 없는가?)
""",
)


# ── Issue 38: 표 데이터 보존율 지표 ──────────────────────────────────────────


def calculate_table_preservation_rate(ground_truth_chunks: list[str], retrieved_chunks: list[str]) -> float:
    """표(IS_TABLE) 청크가 검색 결과에 얼마나 보존되었는지 측정합니다.

    ground_truth_chunks: 정답 표 청크 텍스트 목록
    retrieved_chunks: 검색/리랭킹 후 반환된 청크 텍스트 목록
    반환값: 0.0~1.0 (1.0 = 모든 표 청크 포함)
    """
    if not ground_truth_chunks:
        return 1.0

    retrieved_set = set(c.strip() for c in retrieved_chunks)
    preserved = sum(1 for chunk in ground_truth_chunks if chunk.strip() in retrieved_set)
    return preserved / len(ground_truth_chunks)


def evaluate_table_preservation(
    ground_truth_table_chunks: list[list[str]],
    retrieved_chunk_lists: list[list[str]],
) -> dict[str, float]:
    """여러 쿼리에 대한 표 보존율 평균을 계산합니다."""
    if not ground_truth_table_chunks:
        return {"table_preservation_rate": 1.0, "count": 0}

    rates = [
        calculate_table_preservation_rate(gt, ret)
        for gt, ret in zip(ground_truth_table_chunks, retrieved_chunk_lists, strict=False)
    ]
    return {
        "table_preservation_rate": sum(rates) / len(rates),
        "count": len(rates),
        "min": min(rates),
        "max": max(rates),
    }
