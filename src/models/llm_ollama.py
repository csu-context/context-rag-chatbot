import os
from typing import Any

from langchain_ollama import ChatOllama

from src.models.base import BaseLLM


class OllamaModel(BaseLLM):
    """
    Ollama를 통해 로컬 환경에서 구동되는 sLLM 모델을 제어하는 클래스입니다.
    BaseLLM 인터페이스를 상속받아 프로젝트 내부의 일관된 규격을 유지합니다.
    """

    def __init__(self, model_name: str, **kwargs: Any):
        # 상위 클래스(BaseLLM) 생성자 호출 시 model_name 파라미터를 전달합니다.
        super().__init__(model_name=model_name)
        self._model_name = model_name

        # Ollama는 API 키가 아니라 접속할 로컬 호스트 주소(URL)가 필요합니다.
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        if not base_url:
            raise ValueError("환경 변수 'OLLAMA_BASE_URL' 설정이 올바르지 않습니다.")

        self.llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            **kwargs
        )

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """동기 방식으로 Ollama 모델에 입력을 전달하고 생성된 텍스트를 반환합니다."""
        return self.llm.invoke(prompt, **kwargs).content

    async def ainvoke(self, prompt: str, **kwargs: Any) -> str:
        """비동기 방식으로 Ollama 모델에 입력을 전달하고 생성된 텍스트를 반환합니다."""
        response = await self.llm.ainvoke(prompt, **kwargs)
        return response.content

    @property
    def model_name(self) -> str:
        """추상 클래스의 model_name 속성을 구현합니다."""
        return self._model_name

    @property
    def model_type(self) -> str:
        """현재 인스턴스의 모델 제공자 타입을 반환합니다."""
        return "ollama"
