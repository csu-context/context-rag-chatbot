from unittest.mock import MagicMock, patch

import pytest

from src.core.cache import SemanticCache


@pytest.fixture(autouse=True)
def mock_semantic_cache():
    """Do not override SemanticCache.get in this file."""
    pass


def test_semantic_cache_initialization():
    mock_db = MagicMock()
    with patch("src.core.cache.ChromaDBManager", return_value=mock_db):
        cache = SemanticCache()
        assert cache.db_manager == mock_db


def test_semantic_cache_get_hit_and_miss():
    mock_db = MagicMock()

    with (
        patch("src.core.cache.ChromaDBManager", return_value=mock_db),
        patch(
            "src.core.cache.logger.error",
            side_effect=lambda msg, *args, **kwargs: pytest.fail(f"logger.error called with: {msg}"),
        ),
    ):
        cache = SemanticCache()

        # Test Case 1: Cache Miss (no results)
        mock_db.search.return_value = []
        assert cache.get("hello") is None

        # Test Case 2: Cache Miss (score below threshold)
        mock_db.search.return_value = [{"score": 0.2, "metadata": {"answer": "yes", "sources": "[]"}}]
        assert cache.get("hello") is None

        # Test Case 3: Cache Hit (score above threshold)
        cache.threshold = 0.9
        mock_db.search.return_value = [
            {"score": 0.99, "metadata": {"answer": "cached answer", "sources": '["source1"]'}}
        ]
        res = cache.get("hello")
        assert res is not None
        assert res["answer"] == "cached answer"
        assert res["sources"] == ["source1"]


def test_semantic_cache_add_and_flush():
    mock_db = MagicMock()

    with patch("src.core.cache.ChromaDBManager", return_value=mock_db):
        cache = SemanticCache()

        # Add
        cache.add("query", "answer", ["src"])
        mock_db.upsert_documents.assert_called_once()

        # Flush
        cache.flush()
        mock_db.reset_collection.assert_called_once()
