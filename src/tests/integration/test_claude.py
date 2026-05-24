import os
from unittest.mock import MagicMock, patch

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
    """Claude 모델 호출 테스트 (Mock)"""
    # ChatAnthropic 클래스 자체를 패치하여 인스턴스 생성 시 Mock이 반환되도록 함
    with patch("src.models.llm_claude.ChatAnthropic") as mock_class:
        mock_inst = MagicMock()
        mock_res = MagicMock()
        mock_res.content = "안녕하세요! 저는 Claude입니다."
        mock_res.usage_metadata = {"input_token_count": 10, "output_token_count": 20}
        mock_res.response_metadata = {}  # Pydantic 검증 통과를 위해 dict 할당

        mock_inst.invoke.return_value = mock_res
        mock_class.return_value = mock_inst

        llm = LLMFactory.create_llm(model_type="claude", model_name="claude-3-5-sonnet-20240620")
        response = llm.invoke("안녕하세요, 간단하게 자기소개 부탁드립니다.")

        assert response.content is not None
        assert "Claude" in response.content
        assert "latency" in response.model_dump()
        assert response.model_name == "claude-3-5-sonnet-20240620"
