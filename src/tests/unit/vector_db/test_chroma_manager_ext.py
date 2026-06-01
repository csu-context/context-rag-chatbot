from unittest.mock import MagicMock, patch

import pytest

from src.vector_db.chroma_manager import BGEChromaEmbeddingFunction, ChromaDBManager


@pytest.fixture
def mock_chroma_manager():
    with patch("src.vector_db.chroma_manager.ChromaDBManager._initialize_client_with_retry"):
        manager = ChromaDBManager()
        manager.collection = MagicMock()
        manager.client = MagicMock()
        manager.embedding_fn = MagicMock()
        manager.embedding_fn.model_name = "test_model"
        yield manager


class TestChromaManagerExt:
    def test_bge_chroma_embedding_function(self):
        func = BGEChromaEmbeddingFunction("test_model")
        with patch("src.vector_db.chroma_manager.BGEEmbedder") as mock_embedder:
            mock_embeddings = MagicMock()
            mock_embeddings.tolist.return_value = [[1.0, 2.0]]

            mock_inst = MagicMock()
            mock_inst.encode.return_value = mock_embeddings
            mock_embedder.get_instance.return_value = mock_inst

            res = func(["test doc"])
            assert len(res) == 1
            assert res[0][0] == 1.0
            mock_embedder.get_instance.assert_called_once_with(model_name="test_model")

    def test_check_config_and_auto_reset(self, mock_chroma_manager):
        mock_chroma_manager.collection.metadata = {
            "embedding_model": "old_model",
            "parent_chunk_size": 1000,
            "child_chunk_size": 200,
        }

        with (
            patch.object(mock_chroma_manager, "reset_collection") as mock_reset,
            patch.object(mock_chroma_manager, "_clear_processed_and_cache_files") as mock_clear,
            patch("src.processing.chunking._PARENT_CHUNK_SIZE", 1000),
            patch("src.processing.chunking._CHILD_CHUNK_SIZE", 200),
        ):
            # Different model name should trigger reset
            mock_chroma_manager._check_config_and_auto_reset()
            mock_reset.assert_called_once()
            mock_clear.assert_called_once()

            # Verify modify is called with new metadata
            mock_chroma_manager.collection.modify.assert_called_once()

    def test_upsert_documents(self, mock_chroma_manager):
        mock_chroma_manager._get_valid_collection = MagicMock(return_value=mock_chroma_manager.collection)

        # Test empty
        mock_chroma_manager.upsert_documents([], [])
        mock_chroma_manager.collection.upsert.assert_not_called()

        # Test normal
        mock_chroma_manager.upsert_documents(["id1"], ["doc1"], [{"meta": "data", "null_val": None}])
        mock_chroma_manager.collection.upsert.assert_called_once()
        _args, kwargs = mock_chroma_manager.collection.upsert.call_args
        assert kwargs["ids"] == ["id1"]
        assert kwargs["documents"] == ["doc1"]
        assert kwargs["metadatas"] == [{"meta": "data"}]

    def test_search_results(self, mock_chroma_manager):
        mock_chroma_manager._get_valid_collection = MagicMock(return_value=mock_chroma_manager.collection)

        mock_chroma_manager.collection.query.return_value = {
            "documents": [["doc1"]],
            "metadatas": [[{"meta": 1}]],
            "distances": [[0.1]],
        }

        res = mock_chroma_manager.search("query")
        assert len(res) == 1
        assert res[0]["content"] == "doc1"
        assert res[0]["metadata"] == {"meta": 1}
        assert res[0]["score"] == 0.9

    def test_get_source_count_and_chunks(self, mock_chroma_manager):
        mock_chroma_manager._get_valid_collection = MagicMock(return_value=mock_chroma_manager.collection)

        mock_chroma_manager.collection.get.return_value = {
            "ids": ["id1", "id2"],
            "documents": ["doc1", "doc2"],
            "metadatas": [{"m": 1}, {"m": 2}],
        }

        count = mock_chroma_manager.get_source_count("test.pdf")
        assert count == 2

        chunks = mock_chroma_manager.get_source_chunks("test.pdf")
        assert len(chunks) == 2
        assert chunks[0]["id"] == "id1"
        assert chunks[0]["content"] == "doc1"

    def test_delete_documents(self, mock_chroma_manager):
        mock_chroma_manager._get_valid_collection = MagicMock(return_value=mock_chroma_manager.collection)

        mock_chroma_manager.collection.get.return_value = {"ids": ["id1"]}

        mock_chroma_manager.delete_documents({"source": "test.pdf"})
        mock_chroma_manager.collection.delete.assert_called_once_with(ids=["id1"])

    def test_reset_collection(self, mock_chroma_manager):
        mock_chroma_manager._get_valid_collection = MagicMock(return_value=mock_chroma_manager.collection)

        mock_chroma_manager.collection.get.return_value = {"ids": ["id1", "id2"]}

        mock_chroma_manager.reset_collection()
        mock_chroma_manager.collection.delete.assert_called_once_with(ids=["id1", "id2"])

    def test_clear_processed_and_cache_files(self, mock_chroma_manager):
        with (
            patch("src.vector_db.chroma_manager.PROCESSED_DATA_DIR") as mock_processed,
            patch("src.vector_db.chroma_manager.CACHE_DIR") as mock_cache,
            patch.object(mock_chroma_manager, "_safe_unlink") as mock_unlink,
            patch.object(mock_chroma_manager, "_clear_bm25_cache_directory") as mock_bm25,
        ):
            mock_processed.exists.return_value = True
            mock_processed.glob.return_value = ["a.json"]

            mock_cache.exists.return_value = True
            mock_cache.glob.return_value = ["a.pkl"]

            mock_chroma_manager._clear_processed_and_cache_files()

            assert mock_unlink.call_count == 3  # a.json, manifest.json, a.pkl
            mock_bm25.assert_called_once()
