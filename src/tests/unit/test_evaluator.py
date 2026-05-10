from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from src.eval.evaluator import RagasEvaluator


@pytest.fixture
def mock_db_manager():
    with patch("src.eval.evaluator.ChromaDBManager") as mock:
        yield mock.return_value


@pytest.fixture
def mock_llm_factory():
    with patch("src.eval.evaluator.LLMFactory") as mock:
        # Create a mock for the LLM instance
        mock_inst = MagicMock()
        mock_inst.model_name = "test-model"

        # Mock the invoke return value to be an LLMResponse-like object
        mock_response = MagicMock()
        mock_response.content = "Test answer"
        mock_response.usage = {"input_tokens": 10, "output_tokens": 20}
        mock_inst.invoke.return_value = mock_response

        # Mock get_model to return a mock LangChain model
        mock_inst.get_model.return_value = MagicMock()

        mock.create_llm.return_value = mock_inst
        yield mock


def test_evaluator_initialization(mock_db_manager, mock_llm_factory):
    evaluator = RagasEvaluator(eval_model_type="claude", eval_model_name="claude-haiku")
    assert evaluator.eval_model_name == "claude-haiku"
    assert evaluator.total_usage["inference"]["model"] == "test-model"


def test_evaluator_prepare_dataset(mock_db_manager, mock_llm_factory, tmp_path):
    # Setup golden dataset
    golden_data = [
        {"question": "What is A?", "ground_truth": "A is alpha"},
        {"question": "What is B?", "ground_truth": "B is beta"},
    ]
    golden_path = tmp_path / "golden.json"
    import json

    with open(golden_path, "w", encoding="utf-8") as f:
        json.dump(golden_data, f)

    # Mock DB search
    mock_db_manager.search.return_value = [
        {"content": "A info", "metadata": {"src_name": "doc1", "pg_num": 1}},
        {"content": "B info", "metadata": {"src_name": "doc2", "pg_num": 2}},
    ]

    evaluator = RagasEvaluator()
    dataset = evaluator.prepare_dataset(golden_path, max_samples=2, delay=0)

    assert isinstance(dataset, Dataset)
    assert len(dataset) == 2
    assert dataset[0]["question"] == "What is A?"
    assert dataset[0]["answer"] == "Test answer"
    assert "A info" in dataset[0]["contexts"]


def test_calculate_cost():
    evaluator = RagasEvaluator()
    # Mock pricing for test-model
    evaluator.PRICING["test-model"] = {"input": 10.0, "output": 20.0}

    usage = {"model": "test-model", "input": 1000000, "output": 1000000}
    cost = evaluator._calculate_cost(usage)
    assert cost == 10.0 + 20.0  # 1M tokens each
