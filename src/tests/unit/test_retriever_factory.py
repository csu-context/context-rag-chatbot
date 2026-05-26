from unittest.mock import MagicMock

from src.core.base_retriever import BaseRetriever
from src.core.retriever import EnsembleRetriever, RetrieverFactory
from src.vector_db.bm25_manager import BM25Manager
from src.vector_db.chroma_manager import ChromaDBManager


def test_retriever_factory_vector():
    chroma_mock = MagicMock(spec=ChromaDBManager)
    bm25_mock = MagicMock(spec=BM25Manager)

    retriever = RetrieverFactory.create_retriever(
        retriever_type="vector",
        chroma_manager=chroma_mock,
        bm25_manager=bm25_mock,
    )

    assert isinstance(retriever, BaseRetriever)
    assert retriever == chroma_mock


def test_retriever_factory_bm25():
    chroma_mock = MagicMock(spec=ChromaDBManager)
    bm25_mock = MagicMock(spec=BM25Manager)

    retriever = RetrieverFactory.create_retriever(
        retriever_type="bm25",
        chroma_manager=chroma_mock,
        bm25_manager=bm25_mock,
    )

    assert isinstance(retriever, BaseRetriever)
    assert retriever == bm25_mock


def test_retriever_factory_hybrid():
    chroma_mock = MagicMock(spec=ChromaDBManager)
    bm25_mock = MagicMock(spec=BM25Manager)

    retriever = RetrieverFactory.create_retriever(
        retriever_type="hybrid",
        chroma_manager=chroma_mock,
        bm25_manager=bm25_mock,
    )

    assert isinstance(retriever, BaseRetriever)
    assert isinstance(retriever, EnsembleRetriever)
    assert retriever.chroma == chroma_mock
    assert retriever.bm25 == bm25_mock
