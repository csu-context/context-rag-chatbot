import pytest
from pathlib import Path
from src.utils.file_utils import generate_file_hash
from src.processing.chunking import HierarchicalChunker
from langchain_core.documents import Document
from src.core.chains import RAGPipeline
from src.vector_db.bm25_manager import BM25Manager

def test_generate_file_hash_content_based(tmp_path):
    # 테스트용 파일 생성
    file1 = tmp_path / "test.txt"
    file1.write_text("Hello World Content", encoding="utf-8")
    
    # 해시 생성
    hash1 = generate_file_hash(file1, "manual")
    
    # 수정 시각만 변경하고 내용이 동일할 때 해시가 같은지 검증
    # st_mtime이 변경되어도 콘텐츠가 같으면 해시가 동일해야 함
    import time
    time.sleep(0.1)
    file1.touch()
    
    hash2 = generate_file_hash(file1, "manual")
    assert hash1 == hash2, "Content unchanged but hash changed on mtime modification!"

    # 내용 변경 시 해시가 변경되는지 검증
    file1.write_text("Hello World Content Modified", encoding="utf-8")
    hash3 = generate_file_hash(file1, "manual")
    assert hash1 != hash3, "Content changed but hash remained identical!"

def test_korean_chunker_separators():
    chunker = HierarchicalChunker(parent_chunk_size=100, parent_chunk_overlap=10, child_chunk_size=30, child_chunk_overlap=5)
    
    # 문장 기호가 포함된 한글 텍스트
    text = "조선대학교 학칙입니다. 휴학 신청은 언제나 가능합니다! 하지만 전과는 어렵습니다?"
    
    # 자식 청크들로 분산될 때 separators가 정상 적용되는지 점검
    child_chunks = chunker.split_into_children(text, "parent_001", {"source_id": "test_src"})
    
    # 텍스트가 정상 분할되었고, 비어있지 않은지 검증
    assert len(child_chunks) > 0
    for chunk in child_chunks:
        assert len(chunk["text"]) > 0

def test_parent_document_mapping_in_pipeline(tmp_path, monkeypatch):
    # 임시 PROCESSED_DATA_DIR 모킹
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Mocking PROCESSED_DATA_DIR inside chains
    import src.core.chains as chains
    monkeypatch.setattr(chains, "PROCESSED_DATA_DIR", processed_dir)
    
    # 테스트용 부모-자식 가공 데이터 작성 및 저장
    source_id = "test_source"
    parent_id = "parent_123"
    child_id = "parent_123_c0"
    
    mock_processed_data = [
        {
            "parent_id": parent_id,
            "parent_text": "이것은 온전하고 길며 풍부한 문맥을 가진 부모 청크의 텍스트 본문 전체입니다.",
            "metadata": {
                "source_id": source_id,
                "parent_id": None,
                "chunk_id": parent_id
            },
            "children": [
                {
                    "chunk_id": child_id,
                    "metadata": {
                        "source_id": source_id,
                        "parent_id": parent_id,
                        "chunk_id": child_id
                    },
                    "text": "자식 청크 본문"
                }
            ]
        }
    ]
    
    import json
    with open(processed_dir / f"{source_id}.json", "w", encoding="utf-8") as f:
        json.dump(mock_processed_data, f)
        
    # RAGPipeline 인스턴스 생성 (Retriever 등 모킹)
    pipeline = RAGPipeline(retriever_or_db=None, llm=None, reranker=None)
    
    # 입력 자식 문서 객체 구성
    child_doc = Document(
        page_content="자식 청크 본문",
        metadata={
            "source_id": source_id,
            "parent_id": parent_id,
            "chunk_id": child_id
        }
    )
    
    # Parent mapping 적용
    resolved = pipeline._resolve_parent_documents([child_doc])
    
    assert len(resolved) == 1
    # 자식 문서의 내용이 부모의 온전한 텍스트 본문으로 전환되었는지 검증
    assert resolved[0].page_content == "이것은 온전하고 길며 풍부한 문맥을 가진 부모 청크의 텍스트 본문 전체입니다."
