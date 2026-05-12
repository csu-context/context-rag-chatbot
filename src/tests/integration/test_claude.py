import os

import pytest
from dotenv import load_dotenv

from src.models.factory import LLMFactory
from src.models.llm_claude import ClaudeModel

# 테스트 실행 전 환경 변수 로드
load_dotenv()


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY가 설정되지 않았습니다.")
def test_claude_factory_creation():
    """LLMFactory를 통한 Claude 모델 생성 테스트"""
    llm = LLMFactory.create_llm(model_type="claude", model_name="claude-sonnet-4-6")
    assert isinstance(llm, ClaudeModel)
    assert llm.model_name == "claude-sonnet-4-6"


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY가 설정되지 않았습니다.")
def test_claude_invoke():
    """Claude 모델 실제 호출 테스트 (API 키 필요)"""
    llm = LLMFactory.create_llm(model_type="claude")
    response = llm.invoke("안녕하세요, 간단하게 자기소개 부탁드립니다.")

    assert response.content is not None
    assert len(response.content) > 0
    assert "latency" in response.model_dump()
    assert response.model_name is not None
    assert "claude" in response.model_name.lower()
    print(f"\nClaude 응답: {response.content[:50]}...")
