import pytest

from src.core.retriever import EnsembleRetriever
from src.common.constants import MetadataFields


@pytest.fixture
def retriever():
    return EnsembleRetriever()


def test_bm25_only_returns_results(retriever):
    """BM25 단독 검색 결과 반환 확인."""
    results = retriever._get_bm25_results("휴학 신청 기간", n=3)
    assert isinstance(results, list)


def test_vector_missing_returns_empty(retriever):
    """ChromaManager 미연결 시 빈 리스트 반환 확인."""
    results = retriever._get_vector_results("휴학 신청 기간", n=3)
    assert results == []


def test_get_relevant_documents_no_error(retriever):
    """get_relevant_documents 호출 시 에러 없이 동작 확인."""
    results = retriever.get_relevant_documents("휴학 신청 기간", n=3)
    assert isinstance(results, list)


def test_get_relevant_documents_empty_query(retriever):
    """빈 쿼리 입력 시 빈 리스트 반환 확인."""
    results = retriever.get_relevant_documents("", n=3)
    assert results == []


def test_chunk_id_deduplication(retriever):
    """chunk_id 기반 중복 제거 확인 (최상위 및 메타데이터 내부 모두 대응)."""
    doc1 = {MetadataFields.CHUNK_ID: "001", "content": "테스트1", "metadata": {}}
    doc2 = {"content": "테스트1", "metadata": {MetadataFields.CHUNK_ID: "001"}}
    doc3 = {MetadataFields.CHUNK_ID: "002", "content": "테스트2", "metadata": {}}

    assert retriever._get_doc_id(doc1) == retriever._get_doc_id(doc2)
    assert retriever._get_doc_id(doc1) != retriever._get_doc_id(doc3)