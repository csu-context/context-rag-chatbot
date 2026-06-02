from unittest.mock import MagicMock, patch

import pytest
from ragas import SingleTurnSample

from src.eval.evaluator import InferenceResult, run_rag_inference


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
