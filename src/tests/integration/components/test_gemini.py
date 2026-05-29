import os
from unittest.mock import patch

import google.generativeai as genai
import pytest

from src.models.factory import LLMFactory
from src.models.llm_gemini import GeminiModel


@pytest.mark.integration
def test_gemini_integration():
    """Gemini API 실제 연동 테스트"""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY 또는 GOOGLE_API_KEY가 설정되지 않아 통합 테스트를 건너뜁니다.")
    if os.getenv("CI") == "true":
        pytest.skip("CI 환경에서는 실제 Gemini API 연동 테스트를 실행하지 않습니다.")

    # 1. 사용 가능한 모델 탐색 (gemini-2.0-flash 우선)
    genai.configure(api_key=api_key)
    models = genai.list_models()
    available_models = [
        m.name.replace("models/", "") for m in models if "generateContent" in m.supported_generation_methods
    ]

    # gemini-2.0-flash가 있으면 우선 사용, 없으면 첫 번째 모델 사용
    target_model = next((m for m in available_models if "gemini-2.0-flash" in m), available_models[0])

    # 2. 모델 인스턴스 생성 및 호출
    with patch("os.getenv") as mock_getenv:
        mock_getenv.side_effect = lambda k, default=None: (
            "gemini"
            if k == "MODEL_TYPE"
            else target_model
            if k == "MODEL_NAME"
            else api_key
            if k == "GEMINI_API_KEY"
            else default
        )

        llm_instance = LLMFactory.create_llm(model_type="gemini", model_name=target_model)
        assert isinstance(llm_instance, GeminiModel)
        assert llm_instance.model_name == target_model

        model = llm_instance.get_model()

        # 간단한 프롬프트로 실제 호출 (invoke 사용)
        prompt = "안녕? 짧게 대답해줘."
        try:
            response = model.invoke(prompt)
        except Exception as e:
            # API 호출 한도 초과(429 RESOURCE_EXHAUSTED) 또는 기타 할당량 관련 예외 발생 시 테스트를 스킵합니다.
            err_msg = str(e).lower()
            if "resource_exhausted" in err_msg or "429" in err_msg or "quota" in err_msg:
                pytest.skip(f"Gemini API 호출 한도 초과로 테스트를 건너뜁니다: {e}")
            raise

        assert response is not None
        assert isinstance(response.content, str)
        assert len(response.content) > 0
        print(f"\n[Gemini API 응답 ({target_model})]: {response.content}")
