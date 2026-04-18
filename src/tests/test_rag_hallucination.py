from dotenv import load_dotenv
from langchain_core.documents import Document

from src.core.chains import get_rag_chain

# 1. 환경 변수 및 설정 로드
load_dotenv()


class MockRetriever:
    def __init__(self, docs):
        self.docs = docs

    def invoke(self, query):
        return self.docs

    def __or__(self, other):
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(self.invoke) | other


def run_test_scenario(name, question, docs):
    print(f"\n{'=' * 20} {name} {'=' * 20}")
    retriever = MockRetriever(docs)
    chain = get_rag_chain(retriever)

    response = chain.invoke(question)
    print(f"질문: {question}")
    print(f"답변:\n{response.content}")


if __name__ == "__main__":
    # 시나리오 A: 문서 내 정보가 있는 경우
    docs_a = [
        Document(
            page_content="2026년 신입 사원의 연봉은 5,000만 원입니다.",
            metadata={"source": "연봉규정_2026.pdf", "page": "5"},
        )
    ]
    run_test_scenario(
        "정상 답변 및 출처 테스트", "올해 신입 사원 연봉이 얼마야?", docs_a
    )

    # 시나리오 B: 문서 내 정보가 없는 경우
    docs_b = [
        Document(
            page_content="회사의 점심 시간은 12시부터 1시까지입니다.",
            metadata={"source": "복지안내.pdf", "page": "2"},
        )
    ]
    run_test_scenario("환각 방지 테스트", "회사에서 법인 차량을 빌릴 수 있어?", docs_b)
