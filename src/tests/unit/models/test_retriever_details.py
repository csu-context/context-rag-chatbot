from unittest.mock import MagicMock

from src.common.constants import MetadataFields
from src.core.retriever import EnsembleRetriever, RetrieverFactory


def test_ensemble_retriever_only_bm25():
    mock_bm25 = MagicMock()
    mock_bm25.retrieve.return_value = [{"content": "bm25_doc"}]

    retriever = EnsembleRetriever(chroma_manager=None, bm25_manager=mock_bm25)
    res = retriever.retrieve("query", n=2)
    assert len(res) == 1
    assert res[0]["content"] == "bm25_doc"


def test_ensemble_retriever_only_vector():
    mock_chroma = MagicMock()
    mock_chroma.retrieve.return_value = [{"content": "vector_doc"}]

    # bm25_manager가 전달되지 않으면 기본적으로 BM25Manager()를 호출하므로, mock하여 None으로 세팅
    retriever = EnsembleRetriever(chroma_manager=mock_chroma, bm25_manager=None)
    retriever.bm25 = None  # force None to test safe_retrieve logic

    res = retriever.retrieve("query", n=2)
    assert len(res) == 1
    assert res[0]["content"] == "vector_doc"


def test_ensemble_retriever_both_engines():
    mock_bm25 = MagicMock()
    mock_bm25.retrieve.return_value = [
        {"content": "common_doc", "metadata": {"chunk_id": "c1"}},
        {"content": "bm25_only", "metadata": {"chunk_id": "b1"}},
    ]

    mock_chroma = MagicMock()
    mock_chroma.retrieve.return_value = [
        {"content": "common_doc", "metadata": {"chunk_id": "c1"}},
        {"content": "vector_only", "metadata": {"chunk_id": "v1"}},
    ]

    retriever = EnsembleRetriever(chroma_manager=mock_chroma, bm25_manager=mock_bm25)
    res = retriever.retrieve("query", n=3)

    # RRF fusion should prioritize common_doc
    assert len(res) == 3
    assert res[0]["content"] == "common_doc"
    assert "_rrf_score" in res[0]


def test_ensemble_retriever_get_doc_id_variants():
    retriever = EnsembleRetriever(chroma_manager=MagicMock(), bm25_manager=MagicMock())

    # Variant 1: chunk_id at root
    assert retriever._get_doc_id({"chunk_id": "root_id"}) == "root_id"

    # Variant 2: chunk_id in metadata
    assert retriever._get_doc_id({"metadata": {"chunk_id": "meta_id"}}) == "meta_id"

    # Variant 3: src_name and page fallback
    doc_fallback = {"metadata": {"src_name": "test.pdf", MetadataFields.PG_NUM: 5}}
    assert retriever._get_doc_id(doc_fallback) == "test.pdf::p5"


def test_ensemble_retriever_compare_retrievers(capsys):
    mock_bm25 = MagicMock()
    mock_bm25.retrieve.return_value = [{"content": "bm25"}]
    mock_chroma = MagicMock()
    mock_chroma.retrieve.return_value = [{"content": "vector"}]

    retriever = EnsembleRetriever(chroma_manager=mock_chroma, bm25_manager=mock_bm25)
    retriever.compare_retrievers("test query", n=1)

    captured = capsys.readouterr()
    assert "질의: 'test query'" in captured.out
    assert "[BM25] 단독" in captured.out or "[BM25 단독]" in captured.out
    assert "[Vector 단독]" in captured.out
    assert "[하이브리드 RRF]" in captured.out


def test_retriever_factory():
    mock_chroma = MagicMock()
    mock_bm25 = MagicMock()

    # Case 1: Vector
    ret = RetrieverFactory.create_retriever(retriever_type="vector", chroma_manager=mock_chroma, bm25_manager=mock_bm25)
    assert ret == mock_chroma

    # Case 2: BM25
    ret = RetrieverFactory.create_retriever(retriever_type="bm25", chroma_manager=mock_chroma, bm25_manager=mock_bm25)
    assert ret == mock_bm25

    # Case 3: Hybrid/Ensemble
    ret = RetrieverFactory.create_retriever(retriever_type="hybrid", chroma_manager=mock_chroma, bm25_manager=mock_bm25)
    assert isinstance(ret, EnsembleRetriever)

    # Case 4: Unknown fallback
    ret = RetrieverFactory.create_retriever(
        retriever_type="invalid_type", chroma_manager=mock_chroma, bm25_manager=mock_bm25
    )
    assert ret == mock_chroma
