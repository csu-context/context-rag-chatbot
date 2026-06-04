from unittest.mock import MagicMock, patch

import pytest
from ragas import SingleTurnSample

from src.eval.evaluator import InferenceResult, _get_inference_settings, run_rag_inference


@pytest.fixture
def mock_retriever_factory():
    with patch("src.eval.evaluator.RetrieverFactory") as mock:
        mock.create_retriever.return_value = MagicMock()
        yield mock


@pytest.fixture
def mock_rag_chain():
    with patch("src.core.chains.get_rag_chain") as mock:
        chain_inst = MagicMock()
        chain_inst.stream.return_value = [
            {"stage": "generation", "status": "streaming", "output": "Test answer"},
            {
                "stage": "citation",
                "status": "complete",
                "source_documents": [MagicMock(page_content="A info", metadata={"src_name": "doc1", "pg_num": 1})],
            },
        ]
        mock.return_value = chain_inst
        yield mock


@pytest.fixture
def mock_llm_factory():
    with patch("src.eval.evaluator.LLMFactory") as mock:
        mock_inst = MagicMock()
        mock_inst.model_name = "test-model"
        mock.create_llm.return_value = mock_inst
        yield mock


@pytest.fixture
def mock_resource_snapshot():
    with patch("src.eval.evaluator._get_resource_snapshot") as mock:
        mock.return_value = {"cpu_percent": 10.0, "ram_mb": 1024.0}
        yield mock


@pytest.mark.asyncio
async def test_run_rag_inference(mock_retriever_factory, mock_rag_chain, mock_llm_factory, mock_resource_snapshot):
    test_data = [
        {"question": "What is A?", "ground_truth": "A is alpha"},
    ]

    result = await run_rag_inference(test_data)

    assert isinstance(result, InferenceResult)
    assert len(result.samples) == 1
    assert result.model_name == "test-model"

    sample = result.samples[0]
    assert isinstance(sample, SingleTurnSample)
    assert sample.user_input == "What is A?"
    assert sample.response == "Test answer"
    assert "A info" in sample.retrieved_contexts
    assert sample.reference == "A is alpha"

    assert len(result.latencies) == 1
    assert len(result.resources) == 1
    assert "cpu_percent" in result.resources[0]


@pytest.mark.asyncio
async def test_run_rag_inference_error_handling(
    mock_retriever_factory, mock_rag_chain, mock_llm_factory, mock_resource_snapshot
):
    def mock_stream_error(*args, **kwargs):
        raise Exception("Chain error")
        yield {}

    mock_rag_chain.return_value.stream.side_effect = mock_stream_error

    test_data = [{"question": "Error?", "ground_truth": "None"}]

    result = await run_rag_inference(test_data)

    assert len(result.samples) == 0


def test_inference_settings_ollama_includes_ollama_params(monkeypatch):
    """ollama 백엔드면 공통(temperature) + ollama 전용 추론 파라미터를 모두 기록한다."""
    monkeypatch.setattr("src.eval.evaluator.settings.MODEL_TYPE", "ollama")
    monkeypatch.setattr("src.eval.evaluator.settings.OLLAMA_NUM_PREDICT", 2048)
    monkeypatch.setattr("src.eval.evaluator.settings.OLLAMA_REPEAT_PENALTY", 1.1)

    info = _get_inference_settings()

    assert "temperature" in info
    assert info["num_predict"] == 2048
    assert info["repeat_penalty"] == 1.1
    assert {"num_ctx", "keep_alive", "think"} <= info.keys()


def test_inference_settings_non_ollama_only_common(monkeypatch):
    """비-ollama(gemini/claude) 백엔드면 ollama 전용 키 없이 공통 항목만 기록한다."""
    monkeypatch.setattr("src.eval.evaluator.settings.MODEL_TYPE", "claude")

    info = _get_inference_settings()

    assert "temperature" in info
    assert "num_predict" not in info
    assert all(not k.startswith(("num_", "repeat_", "keep_", "think")) for k in info)
