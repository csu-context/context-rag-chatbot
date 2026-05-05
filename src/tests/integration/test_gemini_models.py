import os
import pytest
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY가 설정되지 않았습니다.")
def test_list_gemini_models():
    """Gemini API를 통해 사용 가능한 모델 목록 조회 테스트"""
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    
    models = list(genai.list_models())
    assert len(models) > 0
    
    # generateContent를 지원하는 모델이 하나 이상 있는지 확인
    generation_models = [m for m in models if "generateContent" in m.supported_generation_methods]
    assert len(generation_models) > 0
