import unittest
from unittest.mock import MagicMock, patch

from src.models.base import LLMResponse
from src.models.factory import LLMFactory
from src.models.llm_gemini import GeminiModel


class TestModelAbstraction(unittest.TestCase):
    def test_factory_creation(self):
        """팩토리가 Gemini 모델을 정상적으로 생성하는지 확인"""
        with patch.dict("os.environ", {"MODEL_TYPE": "gemini", "MODEL_NAME": "gemini-pro"}):
            llm = LLMFactory.create_llm()
            self.assertIsInstance(llm, GeminiModel)
            self.assertEqual(llm.model_name, "gemini-pro")

    def test_llm_response_structure(self):
        """공통 응답 객체(LLMResponse)가 올바른 구조를 가지는지 확인"""
        response = LLMResponse(content="테스트 답변", usage={"total_tokens": 10}, latency=1.5, model_name="test-model")
        self.assertEqual(response.content, "테스트 답변")
        self.assertEqual(response.usage["total_tokens"], 10)

    @patch("langchain_google_genai.ChatGoogleGenerativeAI.invoke")
    def test_gemini_model_invoke(self, mock_invoke):
        """Gemini 모델 호출 시 LLMResponse로 변환되는지 확인 (Mock)"""
        # Mock 설정
        mock_res = MagicMock()
        mock_res.content = "가짜 답변"
        mock_res.usage_metadata = {"total_token_count": 5}
        mock_res.response_metadata = {"finish_reason": "stop"}
        mock_invoke.return_value = mock_res

        model = GeminiModel(model_name="test")
        result = model.invoke("안녕")

        self.assertIsInstance(result, LLMResponse)
        self.assertEqual(result.content, "가짜 답변")
        self.assertEqual(result.usage["total_tokens"], 5)


if __name__ == "__main__":
    unittest.main()
