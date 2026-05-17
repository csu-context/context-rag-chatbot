import os
from typing import Any

from langchain_anthropic import ChatAnthropic

from src.models.base import BaseLLM


class AnthropicModel(BaseLLM):
    """
    Anthropic Claude 모델을 프로젝트 표준 인터페이스에 맞춰 구현한 클래스입니다.
    BaseLLM 추상 클래스를 상속받아 일관된 모델 호출 인터페이스를 제공합니다.
    """

    def __init__(self, model_name: str, **kwargs: Any):
        # 상위 클래스(BaseLLM) 생성자 호출 시 필요한 파라미터를 전달합니다.
        super().__init__(model_name=model_name)
        self._model_name = model_name

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("환경 변수 'ANTHROPIC_API_KEY'가 설정되지 않았습니다.")

        # langchain_anthropic의 ChatAnthropic은 'model' 파라미터를 사용합니다.
        self.llm = ChatAnthropic(
            model=model_name,
            anthropic_api_key=api_key,
            **kwargs
        )

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """동기 방식으로 언어 모델의 응답을 생성합니다."""
        return self.llm.invoke(prompt, **kwargs).content

    async def ainvoke(self, prompt: str, **kwargs: Any) -> str:
        """비동기 방식으로 언어 모델의 응답을 생성합니다."""
        response = await self.llm.ainvoke(prompt, **kwargs)
        return response.content

    @property
    def model_name(self) -> str:
        """추상 클래스의 model_name 속성을 구현합니다."""
        return self._model_name

    @property
    def model_type(self) -> str:
        """현재 인스턴스의 모델 타입을 반환합니다."""
        return "anthropic"
