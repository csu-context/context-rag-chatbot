import os
import sys

# 프로젝트 루트를 PYTHONPATH에 추가 (실행을 위해)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from langchain_core.documents import Document
from tabulate import tabulate

from src.core.reranker import RerankerFactory


def main():
    # 샘플 데이터 준비 (1차 검색 결과 10개 가정)
    query = "딥러닝 모델의 과적합(Overfitting)을 방지하는 방법은 무엇인가요?"

    mock_docs = [
        Document(
            page_content=(
                "드롭아웃(Dropout)은 학습 중 신경망의 일부 뉴런을 무작위로 비활성화하여 과적합을 방지합니다."
            ),
            metadata={"id": "doc1"},
        ),
        Document(
            page_content=("정형 데이터 분석에는 주로 랜덤 포레스트나 XGBoost 같은 앙상블 기법이 활용됩니다."),
            metadata={"id": "doc2"},
        ),
        Document(
            page_content=(
                "데이터 증강(Data Augmentation)은 훈련 데이터의 다양성을 높여 모델이 일반화될 수 있도록 도와줍니다."
            ),
            metadata={"id": "doc3"},
        ),
        Document(
            page_content=("파이썬은 데이터 과학 및 인공지능 분야에서 가장 널리 사용되는 프로그래밍 언어입니다."),
            metadata={"id": "doc4"},
        ),
        Document(
            page_content=(
                "L1/L2 정규화(Regularization)는 손실 함수에 패널티 항을 "
                "추가하여 가중치가 너무 커지는 것을 막아 과적합을 줄입니다."
            ),
            metadata={"id": "doc5"},
        ),
        Document(
            page_content=("합성곱 신경망(CNN)은 주로 이미지 인식 및 분류 작업에 탁월한 성능을 발휘합니다."),
            metadata={"id": "doc6"},
        ),
        Document(
            page_content=(
                "조기 종료(Early Stopping)는 검증 데이터의 손실이 더 이상 "
                "감소하지 않을 때 학습을 중단시키는 기법입니다."
            ),
            metadata={"id": "doc7"},
        ),
        Document(
            page_content=("자연어 처리(NLP)에서는 트랜스포머(Transformer) 아키텍처가 혁신적인 발전을 이끌어냈습니다."),
            metadata={"id": "doc8"},
        ),
        Document(
            page_content=(
                "교차 검증(Cross-Validation)은 제한된 데이터를 여러 "
                "폴드로 나누어 모델의 성능을 객관적으로 평가하는 방법입니다."
            ),
            metadata={"id": "doc9"},
        ),
        Document(
            page_content=(
                "배치 정규화(Batch Normalization)는 학습 과정을 안정화하고 "
                "속도를 높이는 데 기여하지만 과적합 방지 효과도 일부 있습니다."
            ),
            metadata={"id": "doc10"},
        ),
    ]

    print("=== [1차 검색 결과 (원본 순서)] ===")
    for i, doc in enumerate(mock_docs):
        print(f"Rank {i + 1}: {doc.page_content[:50]}...")
    print("\n")

    # 리랭커 테스트 (환경 변수를 임시로 변경하여 테스트)
    reranker_types = ["local", "cohere", "jina"]

    for r_type in reranker_types:
        print(f"\n=== [2차 리랭킹 결과: {r_type.upper()} Reranker] ===")
        os.environ["RERANKER_TYPE"] = r_type

        # API 키가 없으면 스킵 처리
        if r_type == "cohere" and not os.getenv("COHERE_API_KEY"):
            print("⚠️ COHERE_API_KEY가 설정되지 않아 건너뜁니다.")
            continue
        if r_type == "jina" and not os.getenv("JINA_API_KEY"):
            print("⚠️ JINA_API_KEY가 설정되지 않아 건너뜁니다.")
            continue

        try:
            # 1차에서 10개를 받고, 2차에서 최종 5개를 추출
            reranker = RerankerFactory.create(top_k=5)
            result = reranker.rerank_with_timeout(query, mock_docs, max_k=10)

            table_data = []
            for i, (doc, score) in enumerate(zip(result.documents, result.scores, strict=True)):
                # 원본 순위 찾기
                orig_rank = next(
                    (idx + 1 for idx, d in enumerate(mock_docs) if d.metadata["id"] == doc.metadata["id"]),
                    "-",
                )
                table_data.append([i + 1, f"Rank {orig_rank}", f"{score:.4f}", doc.page_content[:40] + "..."])

            headers = ["New Rank", "Old Rank", "Relevance Score", "Content Preview"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            print(f"Elapsed Time: {result.elapsed_time_sec:.2f} sec | Filtered Docs: {result.filtered_count}")

        except Exception as e:
            print(f"Error testing {r_type}: {e}")


if __name__ == "__main__":
    main()
