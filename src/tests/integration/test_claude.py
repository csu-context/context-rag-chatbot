import os
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv

# 실제 구현체인 AnthropicModel을 불러오되, ClaudeModel이라는 이름으로 별칭(alias)을 붙여줌
from src.models.llm_anthropic import AnthropicModel as ClaudeModel

from src.models.factory import LLMFactory

# 테스트 실행 전 환경 변수 로드
load_dotenv()


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY가 설정되지 않았습니다.")
def test_claude_factory_creation():
    """LLMFactory를 통한 Claude 모델 생성 테스트"""
    factory = LLMFactory()
    llm = factory.get_model(model_type="anthropic", model_name="claude-sonnet-4-6")

    # ClaudeModel(실제로는 AnthropicModel)로 확인
    assert isinstance(llm, ClaudeModel)
    assert llm.model_name == "claude-sonnet-4-6"


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY가 설정되지 않았습니다.")
def test_claude_invoke():
    """Claude 모델 호출 테스트 (Mock)"""
    with patch("src.models.llm_anthropic.ChatAnthropic") as mock_class:
        mock_inst = MagicMock()
        mock_res = MagicMock()
        mock_res.content = "안녕하세요! 저는 Claude입니다."
        mock_res.usage_metadata = {"input_token_count": 10, "output_token_count": 20}
        mock_res.response_metadata = {}

        mock_inst.invoke.return_value = mock_res
        mock_class.return_value = mock_inst

        factory = LLMFactory()
        llm = factory.get_model(model_type="anthropic", model_name="claude-3-5-sonnet-20240620")
        response = llm.invoke("안녕하세요, 간단하게 자기소개 부탁드립니다.")

        assert response.content is not None
        assert "Claude" in response.content
        assert "latency" in response.model_dump()
        assert response.model_name == "claude-3-5-sonnet-20240620"
