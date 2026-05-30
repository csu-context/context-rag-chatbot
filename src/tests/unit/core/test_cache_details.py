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
        assert cache.collection == mock_db.collection


def test_semantic_cache_get_hit_and_miss():
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.collection = mock_collection
    mock_db.embed_query.return_value = [0.1] * 1024

    with (
        patch("src.core.cache.ChromaDBManager", return_value=mock_db),
        patch(
            "src.core.cache.logger.error",
            side_effect=lambda msg, *args, **kwargs: pytest.fail(f"logger.error called with: {msg}"),
        ),
    ):
        cache = SemanticCache()

        # Test Case 1: Cache Miss (no results)
        mock_collection.query.return_value = {}
        assert cache.get("hello") is None

        # Test Case 2: Cache Miss (score below threshold)
        mock_collection.query.return_value = {
            "distances": [[0.8]],  # score = 1.0 - 0.8 = 0.2 < threshold
            "metadatas": [[{"answer": "yes", "sources": "[]"}]],
        }
        assert cache.get("hello") is None

        # Test Case 3: Cache Hit (score above threshold)
        cache.threshold = 0.9
        mock_collection.query.return_value = {
            "distances": [[0.01]],  # score = 1.0 - 0.01 = 0.99 >= threshold
            "metadatas": [[{"answer": "cached answer", "sources": '["source1"]'}]],
        }
        res = cache.get("hello")
        assert res is not None
        assert res["answer"] == "cached answer"
        assert res["sources"] == ["source1"]


def test_semantic_cache_add_and_flush():
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.collection = mock_collection
    mock_db.embed_query.return_value = [0.1] * 1024

    with patch("src.core.cache.ChromaDBManager", return_value=mock_db):
        cache = SemanticCache()

        # Add
        cache.add("query", "answer", ["src"])
        mock_collection.add.assert_called_once()

        # Flush
        cache.flush()
        mock_db.client.delete_collection.assert_called_once()
        mock_db.client.get_or_create_collection.assert_called_once()


def test_semantic_cache_get_valid_collection_fallback():
    mock_db = MagicMock()
    mock_collection = MagicMock()
    # count() raises Exception to simulate stale collection
    mock_collection.count.side_effect = Exception("does not exist")
    mock_db.collection = mock_collection

    with patch("src.core.cache.ChromaDBManager", return_value=mock_db):
        cache = SemanticCache()
        # _get_valid_collection should recreate the collection
        res = cache._get_valid_collection()
        mock_db.client.get_or_create_collection.assert_called_once()
        assert res == mock_db.client.get_or_create_collection.return_value
