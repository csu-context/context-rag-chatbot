import os
from unittest.mock import patch

import google.generativeai as genai
import pytest

from src.models.factory import LLMFactory
from src.models.llm_gemini import GeminiModel


@pytest.mark.integration
def test_gemini_integration():
    """Gemini API 실제 연동 및 응답 생성 통합 테스트"""

    # 1. API 키 검증 및 실행 환경 확인
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY 또는 GOOGLE_API_KEY가 설정되지 않아 통합 테스트를 건너뜁니다.")
    if os.getenv("CI") == "true":
        pytest.skip("CI 환경에서는 불필요한 과금 방지를 위해 실제 Gemini API 연동 테스트를 실행하지 않습니다.")

    # 2. 사용 가능한 최적 모델 탐색 (gemini-2.0-flash 우선)
    genai.configure(api_key=api_key)
    models = genai.list_models()
    available_models = [
        m.name.replace("models/", "") for m in models if "generateContent" in m.supported_generation_methods
    ]

    # gemini-2.0-flash 지원 시 우선 할당, 미지원 시 사용 가능한 첫 번째 모델로 Fallback
    target_model = next((m for m in available_models if "gemini-2.0-flash" in m), available_models[0])

    # 3. 환경 변수 모킹 및 모델 인스턴스 생성 검증
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

        # 팩토리 패턴을 통한 GeminiModel 인스턴스화 검증
        llm_instance = LLMFactory.get_model(model_type="gemini", model_name=target_model)
        assert isinstance(llm_instance, GeminiModel)
        assert llm_instance.model_name == target_model

        model = llm_instance.get_model()

        # 4. 프롬프트 질의 및 API 응답 안정성 테스트
        prompt = "안녕? 짧게 대답해줘."
        try:
            response = model.invoke(prompt)
        except Exception as e:
            # 무료 티어 API 호출 한도 초과(429 RESOURCE_EXHAUSTED) 발생 시 테스트를 유연하게 스킵
            err_msg = str(e).lower()
            if "resource_exhausted" in err_msg or "429" in err_msg or "quota" in err_msg:
                pytest.skip(f"Gemini API 호출 할당량 초과(Quota Exceeded)로 인해 테스트를 건너뜁니다: {e}")
            raise

        # 5. 결과 객체 무결성 검증
        assert response is not None
        assert isinstance(response.content, str)
        assert len(response.content) > 0
        print(f"\n[Gemini API 응답 ({target_model})]: {response.content}")
