import os
from typing import Any

from langchain_anthropic import ChatAnthropic

from src.models.base import BaseLLM, LLMResponse


class AnthropicModel(BaseLLM):
    """Anthropic Claude 모델을 프로젝트 표준 인터페이스에 맞춰 구현한 클래스입니다."""

    def __init__(self, model_name: str, **kwargs: Any):
        super().__init__(model_name=model_name)
        self._model_name = model_name

        api_key = kwargs.pop("api_key", os.getenv("ANTHROPIC_API_KEY"))
        if not api_key:
            raise ValueError("환경 변수 'ANTHROPIC_API_KEY'가 설정되지 않았습니다.")

        self.llm = ChatAnthropic(model=model_name, anthropic_api_key=api_key, **kwargs)

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """BaseLLM의 규격에 맞게 LLMResponse 객체를 반환하도록 수정함."""
        response = self.llm.invoke(prompt, **kwargs)
        return LLMResponse(content=response.content, model_name=self.model_name)

    def get_model(self) -> Any:
        """BaseLLM의 추상 메서드 구현: 내부 LangChain 모델 인스턴스 반환."""
        return self.llm

    @property
    def model_name(self) -> str:
        return self._model_name

    @model_name.setter
    def model_name(self, value: str):
        self._model_name = value
