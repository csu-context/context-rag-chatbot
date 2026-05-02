from ragas.metrics import (
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
