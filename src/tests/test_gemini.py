import os

from dotenv import load_dotenv
from google.api_core import exceptions
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. 환경 변수 로드
load_dotenv()


def test_gemini_rag_foundation():
    print("--- Gemini RAG 환경 검증 시작 ---")

    # API 키 로드 확인
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("에러: .env 파일에서 GOOGLE_API_KEY를 찾을 수 없습니다.")
        return

    try:
        # 모델명을 안정적인 버전으로 수정합니다.
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",  # 또는 "gemini-3.1-flash-lite-preview"
            temperature=1.0,
            google_api_key=api_key,
        )

        # 3. RAG 시스템 프롬프트 설정 (기초 환경 검증용)
        messages = [
            (
                "system",
                "당신은 로컬 데이터(BGE-M3 임베딩)를 기반으로 답변하는 RAG 시스템의 비서입니다.",
            ),
            (
                "human",
                f"현재 설정된 데이터 경로({os.getenv('DATA_PATH')})를 인지하고 있나요? 연결 상태를 확인해줘.",
            ),
        ]

        # 4. 응답 수신 테스트
        print("모델 호출 중")
        response = llm.invoke(messages)

        print("\n[정상 응답 확인]")
        print(f"시스템 답변: {response.content}")
        print("-" * 30)
        print("결과: Gemini API 연결 및 RAG 기초 환경 검증 완료.")

    # 5. 세부 예외 처리 로직 (사용자 요청 사항 반영)
    except exceptions.InvalidArgument as e:
        print(
            f"\n[연결 실패] 에러 발생: API 키가 올바르지 않거나 모델 설정이 잘못되었습니다.\n상세내용: {e}"
        )
    except exceptions.DeadlineExceeded:
        print(
            "\n[연결 실패] 에러 발생: Google 서버 응답 시간이 초과되었습니다 (Timeout)."
        )
    except exceptions.ResourceExhausted:
        print(
            "\n[연결 실패] 에러 발생: 무료 티어 할당량(Quota)을 초과했습니다. 잠시 후 다시 시도하세요."
        )
    except Exception as e:
        print(f"\n[연결 실패] 기타 에러 발생: {e}")


if __name__ == "__main__":
    test_gemini_rag_foundation()
