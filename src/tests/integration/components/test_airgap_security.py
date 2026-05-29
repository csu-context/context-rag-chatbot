from unittest.mock import patch

from src.common.config import settings
from src.core.reranker import CohereReranker, RerankerFactory
from src.models.factory import LLMFactory
from src.models.llm_gemini import GeminiModel
from src.models.llm_ollama import OllamaModel


def test_llm_factory_airgap():
    # 케이스 1: ALLOW_EXTERNAL_API = False (외부 호출 불허), 모델 타입 = gemini
    with (
        patch.object(settings, "ALLOW_EXTERNAL_API", False),
        patch.object(settings, "MODEL_TYPE", "gemini"),
        patch.object(settings, "MODEL_NAME", "gemini-1.5-flash"),
    ):
        llm = LLMFactory.create_llm()
        assert isinstance(llm, OllamaModel)
        assert llm.model_name == "gemma4:e4b"

    # 케이스 2: ALLOW_EXTERNAL_API = True (외부 호출 허용), 모델 타입 = gemini
    with (
        patch.object(settings, "ALLOW_EXTERNAL_API", True),
        patch.object(settings, "MODEL_TYPE", "gemini"),
        patch.object(settings, "MODEL_NAME", "gemini-1.5-flash"),
        patch.object(settings, "GEMINI_API_KEY", "dummy-key"),
    ):
        llm = LLMFactory.create_llm()
        assert isinstance(llm, GeminiModel)
        assert llm.model_name == "gemini-1.5-flash"


@patch("src.models.embedder.SentenceTransformer")
def test_embedder_airgap(mock_sentence_transformer):
    from src.models.embedder import BGEEmbedder

    # 케이스 1: ALLOW_EXTERNAL_API = False (외부 호출 불허), 비로컬 모델명 로드 시도
    with patch.object(settings, "ALLOW_EXTERNAL_API", False):
        # 기본값이 아닌 임베딩 모델명으로 BGEEmbedder 생성 시도
        _ = BGEEmbedder(model_name="some-external-model-name")
        # 로컬 기본 모델인 BAAI/bge-m3로 전환하고 local_files_only=True 인자를 전달해야 함
        args, kwargs = mock_sentence_transformer.call_args
        assert args[0] == "BAAI/bge-m3"
        assert kwargs.get("local_files_only") is True

    # 케이스 2: ALLOW_EXTERNAL_API = True (외부 호출 허용), 요청된 모델을 그대로 로드
    mock_sentence_transformer.reset_mock()
    with patch.object(settings, "ALLOW_EXTERNAL_API", True):
        _ = BGEEmbedder(model_name="custom-model")
        args, kwargs = mock_sentence_transformer.call_args
        assert args[0] == "custom-model"
        assert kwargs.get("local_files_only") is False


def test_reranker_factory_airgap():
    # 케이스 1: ALLOW_EXTERNAL_API = False (외부 호출 불허), 리랭커 타입 = cohere
    with (
        patch.object(settings, "ALLOW_EXTERNAL_API", False),
        patch.object(settings, "RERANKER_TYPE", "cohere"),
        patch.object(settings, "ALLOW_EXTERNAL_RERANKER", True),
    ):
        # ALLOW_EXTERNAL_RERANKER가 True이더라도 ALLOW_EXTERNAL_API가 False이므로,
        # 반드시 로컬 리랭커(CrossEncoderReranker)로 Fallback 해야 함
        reranker = RerankerFactory.create()
        assert not isinstance(reranker, CohereReranker)
        assert reranker.name == "Local CrossEncoder"

    # 케이스 2: ALLOW_EXTERNAL_API = True 및 ALLOW_EXTERNAL_RERANKER = True, 리랭커 타입 = cohere
    with (
        patch.object(settings, "ALLOW_EXTERNAL_API", True),
        patch.object(settings, "RERANKER_TYPE", "cohere"),
        patch.object(settings, "ALLOW_EXTERNAL_RERANKER", True),
        patch.dict("os.environ", {"COHERE_API_KEY": "dummy-key"}),
    ):
        reranker = RerankerFactory.create()
        assert isinstance(reranker, CohereReranker)
        assert reranker.name == "Cohere"
