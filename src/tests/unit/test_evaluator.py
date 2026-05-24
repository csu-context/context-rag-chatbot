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
    # run_rag_inference 내부에서 import되는 get_rag_chain을 모킹
    with patch("src.core.chains.get_rag_chain") as mock:
        chain_inst = MagicMock()
        # 체인은 이제 generator를 반환하는 stream 메서드를 사용함
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


@pytest.mark.asyncio
async def test_run_rag_inference(mock_db_manager, mock_rag_chain, mock_llm_factory):
    test_data = [
        {"question": "What is A?", "ground_truth": "A is alpha"},
    ]

    dataset, model_name = await run_rag_inference(test_data)

    assert isinstance(dataset, EvaluationDataset)
    assert len(dataset) == 1
    assert model_name == "test-model"

    # Ragas v0.4.x EvaluationDataset은 samples 리스트를 가짐
    sample = dataset[0]
    assert sample.user_input == "What is A?"
    assert sample.response == "Test answer"  # 출처 제거 확인
    assert "A info" in sample.retrieved_contexts
    assert sample.reference == "A is alpha"


@pytest.mark.asyncio
async def test_run_rag_inference_error_handling(mock_db_manager, mock_rag_chain, mock_llm_factory):
    # 체인 실행 시 에러 발생 시뮬레이션
    def mock_stream_error(*args, **kwargs):
        raise Exception("Chain error")
        yield {}

    mock_rag_chain.return_value.stream.side_effect = mock_stream_error

    test_data = [{"question": "Error?", "ground_truth": "None"}]

    dataset, _ = await run_rag_inference(test_data)

    # 에러 발생 시 해당 샘플은 제외되어야 함
    assert len(dataset) == 0
