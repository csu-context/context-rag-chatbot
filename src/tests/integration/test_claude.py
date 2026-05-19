import os
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv

from src.models.factory import LLMFactory
# 실재하는 파일 구조와 클래스명(AnthropicModel)으로 매핑을 올바르게 수정함
from src.models.llm_anthropic import AnthropicModel

# 테스트 실행 전 환경 변수 로드
load_dotenv()


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY가 설정되지 않았습니다.")
def test_claude_factory_creation():
    """LLMFactory를 통한 Claude 모델 생성 테스트"""
    # 실제 구현된 메서드인 get_model을 사용하도록 수정함
    factory = LLMFactory()
    llm = factory.get_model(model_type="anthropic", model_name="claude-sonnet-4-6")
    assert isinstance(llm, AnthropicModel)
    assert llm.model_name == "claude-sonnet-4-6"


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY가 설정되지 않았습니다.")
def test_claude_invoke():
    """Claude 모델 호출 테스트 (Mock)"""
    # ChatAnthropic 클래스 경로와 중첩된 with 문을 ruff 규칙에 맞게 한 줄로 정렬함
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