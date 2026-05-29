from unittest.mock import MagicMock, patch

from src.models.base import LLMResponse
from src.models.factory import LLMFactory
from src.models.llm_gemini import GeminiModel


def test_factory_creation():
    """팩토리가 Gemini 모델을 정상적으로 생성하는지 확인"""
    from src.models.factory import settings

    with (
        patch.object(settings, "MODEL_TYPE", "gemini"),
        patch.object(settings, "MODEL_NAME", "gemini-pro"),
        patch.object(settings, "GOOGLE_API_KEY", "dummy-key"),
    ):
        llm = LLMFactory.create_llm()
        assert isinstance(llm, GeminiModel)
        assert llm.model_name == "gemini-pro"


def test_llm_response_structure():
    """공통 응답 객체(LLMResponse)가 올바른 구조를 가지는지 확인"""
    response = LLMResponse(content="테스트 답변", usage={"total_tokens": 10}, latency=1.5, model_name="test-model")
    assert response.content == "테스트 답변"
    assert response.usage["total_tokens"] == 10


@patch("langchain_google_genai.ChatGoogleGenerativeAI.invoke")
def test_gemini_model_invoke(mock_invoke):
    """Gemini 모델 호출 시 LLMResponse로 변환되는지 확인 (Mock)"""
    with patch.dict("os.environ", {"GOOGLE_API_KEY": "dummy-key"}):
        # Mock 설정
        mock_res = MagicMock()
        mock_res.content = "가짜 답변"
        mock_res.usage_metadata = {"total_token_count": 5}
        mock_res.response_metadata = {"finish_reason": "stop"}
        mock_invoke.return_value = mock_res

        model = GeminiModel(model_name="test", api_key="dummy-key")
        result = model.invoke("안녕")

        assert isinstance(result, LLMResponse)
        assert result.content == "가짜 답변"
        assert result.usage["total_tokens"] == 5
