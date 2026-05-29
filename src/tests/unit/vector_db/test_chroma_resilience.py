from unittest.mock import MagicMock, patch

import pytest

from src.data.parser import ManualParser
from src.vector_db.chroma_manager import ChromaConnectionMixin


class MockChromaDBManager(ChromaConnectionMixin):
    def __init__(self):
        self.collection_name = "test_resilience_collection"
        self.embedding_fn = MagicMock()


def test_chroma_connection_retry_failures():
    manager = MockChromaDBManager()

    def mock_getenv(key, default=None):
        if key == "CHROMA_SERVER_HOST":
            return "localhost"
        if key == "CHROMA_SERVER_PORT":
            return "8000"
        return default

    # HttpClient 및 PersistentClient가 항상 예외를 유발하도록 패치
    with (
        patch("chromadb.HttpClient", side_effect=ConnectionError("Chroma server down")),
        patch("chromadb.PersistentClient", side_effect=Exception("Local path issue")),
        patch("os.getenv", side_effect=mock_getenv),
    ):
        with pytest.raises(RuntimeError, match="ChromaDB initialization failed."):
            manager._initialize_client_with_retry(max_retries=3, retry_delay=0.1)


def test_manual_parser_file_not_found():
    # 존재하지 않는 파일 초기화 시 FileNotFoundError 검증
    with pytest.raises(FileNotFoundError):
        ManualParser("non_existent_file.pdf")


def test_manual_parser_invalid_extension():
    # 유효한 임시 경로 패치하여 초기화 성공 유도 후 지원하지 않는 확장자 변경
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("src.data.parser.generate_file_hash", return_value="hash_123"),
    ):
        parser = ManualParser("sample_doc.csv")
        # csv 확장자에 대해 parse() 수행 시 ValueError 검증
        with pytest.raises(ValueError, match="지원하지 않는 파일 형식입니다"):
            parser.parse()


def test_manual_parser_clean_text():
    # 텍스트 청소 규칙 검증
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("src.data.parser.generate_file_hash", return_value="hash_123"),
    ):
        parser = ManualParser("test.md")
        assert parser._clean_text("제  1  조  목적") == "제1조 목적"
        assert parser._clean_text("제 2 조") == "제2조"
        assert parser._clean_text("문장 내 (  ) 빈 괄호 정제") == "문장 내 () 빈 괄호 정제"
        assert parser._clean_text("중복    공백    제거") == "중복 공백 제거"


def test_chroma_db_manager_operations():
    from src.vector_db.chroma_manager import ChromaDBManager, ChromaConnectionMixin

    ChromaDBManager._shared_client = None
    ChromaConnectionMixin._shared_client = None

    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    # metadata 모킹
    mock_collection.metadata = {
        "embedding_model": "BAAI/bge-m3",
        "parent_chunk_size": 500,
        "child_chunk_size": 200,
    }
    mock_collection.count.return_value = 5

    # mock embedder
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = MagicMock(tolist=lambda: [[0.1] * 1024])

    with (
        patch("chromadb.PersistentClient", return_value=mock_client),
        patch("src.models.embedder.BGEEmbedder.get_instance", return_value=mock_embedder),
        patch("src.processing.chunking._PARENT_CHUNK_SIZE", 500),
        patch("src.processing.chunking._CHILD_CHUNK_SIZE", 200),
    ):
        manager = ChromaDBManager(collection_name="test_collection")

        # embed_query 테스트
        q_vec = manager.embed_query("test query")
        assert len(q_vec) == 1024

        # count 테스트
        assert manager.get_count() == 5

        # retrieve 테스트
        mock_collection.query.return_value = {
            "documents": [["doc1"]],
            "metadatas": [[{"src_name": "test.pdf"}]],
            "distances": [[0.1]],
        }
        res = manager.retrieve("query", n=1)
        assert len(res) == 1
        assert res[0]["content"] == "doc1"
        assert res[0]["score"] == 0.9

        # upsert_documents 테스트
        manager.upsert_documents(ids=["id1"], documents=["doc1"], metadatas=[{"src_name": "test.pdf"}])
        mock_collection.upsert.assert_called_once()

        # get_source_count & get_source_chunks 테스트
        mock_collection.get.return_value = {"ids": ["id1"], "documents": ["doc1"], "metadatas": [{"src_name": "test.pdf"}]}
        assert manager.get_source_count("test.pdf") == 1
        chunks = manager.get_source_chunks("test.pdf")
        assert len(chunks) == 1
        assert chunks[0]["id"] == "id1"

        # delete_documents 테스트
        manager.delete_documents(where={"src_name": "test.pdf"})
        assert mock_collection.delete.call_count == 1

        # reset_collection 테스트
        mock_collection.get.return_value = {"ids": ["id1"]}
        manager.reset_collection()
        assert mock_collection.delete.call_count == 2


