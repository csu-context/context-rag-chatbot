from models.llm_anthropic import AnthropicModel
from models.llm_ollama import OllamaModel


class LLMFactory:
    """LLM 인스턴스를 생성하는 팩토리 클래스"""

    def __init__(self):
        pass

    @staticmethod
    def get_model(model_type: str, model_name: str = None):
        """지정된 model_type에 따라 적절한 LLM 객체를 반환"""
        model_type = model_type.lower()

        if model_type == "anthropic":
            return AnthropicModel(model_name=model_name)
        elif model_type == "ollama":
            return OllamaModel(model_name=model_name)
        else:
            raise ValueError("지원하지 않는 모델 타입입니다.")
