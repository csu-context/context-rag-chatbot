from src.core.retriever import EnsembleRetriever


def test_ensemble_retriever():
    retriever = EnsembleRetriever()

    test_queries = ["휴학 신청 기간", "복학 신청 방법", "성적 장학금"]

    for q in test_queries:
        retriever.compare_retrievers(q, n=3)


if __name__ == "__main__":
    test_ensemble_retriever()