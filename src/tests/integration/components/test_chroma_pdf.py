import os

import pytest
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.paths import RAW_DATA_DIR
from src.vector_db.chroma_manager import ChromaDBManager


@pytest.fixture
def test_pdf_path():
    """테스트용 PDF 파일 경로 반환"""
    # RAW_DATA_DIR 하위에서 PDF 파일을 찾음
    pdf_files = list(RAW_DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        pytest.skip("테스트를 위한 PDF 파일이 data/raw에 없습니다.")
    return pdf_files[0]


@pytest.mark.skipif(os.getenv("CI") == "true", reason="CI 환경에서는 로컬 모델 기반 DB 업서트 테스트를 스킵합니다.")
def test_pdf_to_chroma(test_pdf_path):
    """PDF 로드, 청킹, ChromaDB 저장 및 검색 E2E 테스트"""
    # 1. PDF 로드 및 청킹
    loader = PyPDFLoader(str(test_pdf_path))
    raw_docs = loader.load()
    assert len(raw_docs) > 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(raw_docs)
    assert len(chunks) > 0

    # 2. 데이터 준비
    ids = [f"test_chunk_{i}" for i in range(len(chunks[:10]))]  # 10개만 테스트
    documents = [chunk.page_content for chunk in chunks[:10]]
    metadatas = [chunk.metadata for chunk in chunks[:10]]

    # 3. ChromaDB 연동 (테스트용 임시 컬렉션)
    db_manager = ChromaDBManager(collection_name="test_temp_collection")
    db_manager.upsert_documents(ids=ids, documents=documents, metadatas=metadatas)

    # 4. 검색 테스트
    query = "테스트 검색 쿼리"
    results = db_manager.search(query_text=query, k=2)
    assert len(results) > 0
    assert "content" in results[0]
    assert "score" in results[0]

    # 정리: 컬렉션 삭제 (필요 시 ChromaDBManager에 delete_collection 추가 필요)
    # 현재는 구현되어 있지 않으므로 생략하거나 수동 정리
