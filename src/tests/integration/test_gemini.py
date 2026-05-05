import os

import google.generativeai as genai
import pytest
from dotenv import load_dotenv
from google.api_core import exceptions
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY가 설정되지 않았습니다.")
def test_gemini_connection():
    """Gemini API 연결 확인 - 사용 가능한 모델을 동적으로 찾아 테스트"""
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)

    # 1. 사용 가능한 모델 탐색 (gemini-1.5-flash 우선)
    available_models = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
    if not available_models:
        pytest.fail("사용 가능한 Gemini 모델을 찾을 수 없습니다.")

    # gemini-1.5-flash가 있으면 우선 사용, 없으면 첫 번째 모델 사용
    target_model = next((m for m in available_models if "gemini-1.5-flash" in m), available_models[0])
    # 'models/' 프리픽스 제거 (langchain용)
    target_model_clean = target_model.replace("models/", "")

    try:
        llm = ChatGoogleGenerativeAI(
            model=target_model_clean,
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
        pytest.fail(f"Gemini API 호출 실패 ({target_model_clean}): {e}")
