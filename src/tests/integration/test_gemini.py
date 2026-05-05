import os

import pytest
from dotenv import load_dotenv
from google.api_core import exceptions
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY가 설정되지 않았습니다.")
def test_gemini_connection():
    """Gemini API 연결 및 기본적인 응답 생성 테스트"""
    api_key = os.getenv("GOOGLE_API_KEY")

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            google_api_key=api_key,
        )

        messages = [
            ("system", "당신은 RAG 시스템의 비서입니다."),
            ("human", "연결 상태를 확인해줘."),
        ]

        response = llm.invoke(messages)
        assert response.content is not None
        assert len(response.content) > 0

    except (exceptions.InvalidArgument, exceptions.DeadlineExceeded, exceptions.ResourceExhausted) as e:
        pytest.fail(f"Gemini API 호출 실패: {e}")
