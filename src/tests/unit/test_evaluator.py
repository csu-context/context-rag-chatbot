from unittest.mock import MagicMock, patch

import pytest
from ragas import EvaluationDataset

from src.eval.evaluator import run_rag_inference


@pytest.fixture
def mock_db_manager():
    with patch("src.eval.evaluator.ChromaDBManager") as mock:
        manager_inst = mock.return_value
        manager_inst.search.return_value = [
            {"content": "A info", "metadata": {"src_name": "doc1", "pg_num": 1}},
        ]
        yield manager_inst


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
    # LLMFactory 인스턴스를 직접 생성하지 않고 메서드 자체를 패치
    with patch("src.eval.evaluator.LLMFactory") as mock_factory_cls:
        mock_instance = MagicMock()
        mock_model = MagicMock()
        mock_model.model_name = "test-model"

        # 팩토리 인스턴스가 get_model을 호출하면 mock_model을 반환하도록 설정
        mock_instance.get_model.return_value = mock_model
        mock_factory_cls.return_value = mock_instance

        yield mock_instance


@pytest.mark.asyncio
async def test_run_rag_inference(mock_db_manager, mock_rag_chain, mock_llm_factory):
    test_data = [
        {"question": "What is A?", "ground_truth": "A is alpha"},
    ]

    dataset, model_name = await run_rag_inference(test_data)

    assert isinstance(dataset, EvaluationDataset)
    assert len(dataset) == 1
    assert model_name == "test-model"

    sample = dataset[0]
    assert sample.user_input == "What is A?"
    assert sample.response == "Test answer"
    assert "A info" in sample.retrieved_contexts
    assert sample.reference == "A is alpha"


@pytest.mark.asyncio
async def test_run_rag_inference_error_handling(mock_db_manager, mock_rag_chain, mock_llm_factory):
    def mock_stream_error(*args, **kwargs):
        raise Exception("Chain error")
        yield {}

    mock_rag_chain.return_value.stream.side_effect = mock_stream_error

    test_data = [{"question": "Error?", "ground_truth": "None"}]

    dataset, _ = await run_rag_inference(test_data)

    assert len(dataset) == 0
