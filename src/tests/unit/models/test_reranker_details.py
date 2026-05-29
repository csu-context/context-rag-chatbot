import pytest
from unittest.mock import MagicMock, patch
import requests
from langchain_core.documents import Document

from src.core.reranker import (
    BaseReranker,
    CrossEncoderReranker,
    CohereReranker,
    JinaReranker,
    RerankerFactory,
    RerankResult,
)


class DummyReranker(BaseReranker):
    def rerank(self, query, documents, top_k=None, threshold=None):
        raise RuntimeError("Rerank error")


def test_base_reranker_fallback():
    reranker = DummyReranker(name="dummy", top_k=2)
    docs = [Document(page_content="doc1"), Document(page_content="doc2"), Document(page_content="doc3")]
    
    # Reranking fails, fallback returns first top_k docs
    res = reranker.rerank_with_timeout("query", docs)
    assert len(res.documents) == 2
    assert res.documents[0].page_content == "doc1"
    assert res.scores == [0.0, 0.0]


def test_cross_encoder_reranker_singleton():
    CrossEncoderReranker.reset_instance()
    inst1 = CrossEncoderReranker.get_instance(model_name="test-model", top_k=3, threshold=0.1)
    inst2 = CrossEncoderReranker.get_instance(model_name="test-model", top_k=5, threshold=0.2)
    
    assert inst1 is inst2
    assert inst1.top_k == 5
    assert inst1.threshold == 0.2


def test_cross_encoder_reranker_prediction():
    CrossEncoderReranker.reset_instance()
    reranker = CrossEncoderReranker.get_instance(model_name="test-model", device="cpu")
    
    mock_model = MagicMock()
    # Mock return logits
    mock_model.predict.return_value = MagicMock(tolist=lambda: [1.0, -1.0])
    
    reranker._model = mock_model
    docs = [Document(page_content="doc1"), Document(page_content="doc2")]
    
    res = reranker.rerank("query", docs, threshold=0.1)
    assert len(res.documents) == 2
    # Sigmoid(1.0) is approx 0.73, Sigmoid(-1.0) is approx 0.26
    assert res.scores[0] > 0.7
    assert res.scores[1] > 0.25


def test_api_base_reranker_missing_key():
    reranker = CohereReranker(api_key="")
    docs = [Document(page_content="doc1")]
    with pytest.raises(ValueError, match="API Key missing"):
        reranker.rerank("query", docs)


def test_api_base_reranker_request_error():
    reranker = CohereReranker(api_key="fake_key")
    docs = [Document(page_content="doc1")]
    
    with patch.object(reranker._session, "post", side_effect=requests.exceptions.ConnectionError("conn error")):
        with pytest.raises(requests.exceptions.RequestException):
            reranker.rerank("query", docs)


def test_cohere_jina_payloads():
    docs = [Document(page_content="hello")]
    
    cohere = CohereReranker(api_key="fake")
    cohere_payload = cohere._build_payload("query", docs, top_k=3)
    assert cohere_payload["return_documents"] is False
    assert cohere_payload["top_n"] == 3

    jina = JinaReranker(api_key="fake")
    jina_payload = jina._build_payload("query", docs, top_k=2)
    assert "return_documents" not in jina_payload
    assert jina_payload["top_n"] == 2


def test_reranker_factory():
    # Case 1: External allowed and cohere selected
    with (
        patch("src.common.config.settings.RERANKER_TYPE", "cohere"),
        patch("src.common.config.settings.ALLOW_EXTERNAL_RERANKER", True),
        patch("src.common.config.settings.ALLOW_EXTERNAL_API", True)
    ):
        ret = RerankerFactory.create(top_k=2)
        assert isinstance(ret, CohereReranker)

    # Case 2: External disallowed but cohere selected -> fallbacks to local
    with (
        patch("src.common.config.settings.RERANKER_TYPE", "cohere"),
        patch("src.common.config.settings.ALLOW_EXTERNAL_RERANKER", False),
        patch("src.common.config.settings.ALLOW_EXTERNAL_API", True)
    ):
        ret = RerankerFactory.create(top_k=2)
        assert isinstance(ret, CrossEncoderReranker)
