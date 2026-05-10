import os

import pytest
from dotenv import load_dotenv

from src.common.constants import MetadataFields
from src.core.chains import get_rag_chain
from src.processing.chunking import create_parent_child_chunks
from src.vector_db.chroma_manager import ChromaDBManager

# 테스트 실행 전 환경 변수 로드
load_dotenv()


@pytest.mark.skipif(
    os.getenv("CI") == "true" or (os.getenv("GOOGLE_API_KEY", "") in ["", "None"] and os.getenv("ANTHROPIC_API_KEY", "") in ["", "None"]),
    reason="CI 환경에서는 기 설정된 모델 접근 권한 문제로 스킵하거나 API 키가 없습니다.",
)
def test_full_rag_pipeline():
    """데이터 전처리부터 RAG 답변 생성까지의 전체 파이프라인 테스트"""
    # 1. 테스트 데이터 준비
    sample_markdown = """
# 제1장 총칙
## 제1조 목적
이 규정은 조선대학교의 학사 운영에 관한 사항을 규정함을 목적으로 한다.

## 제2조 정의
이 규정에서 사용하는 용어의 뜻은 다음과 같다.

| 구분 | 설명 | 비고 |
|---|---|---|
| 전공 | 학생이 소속된 주전공 | - |
| 복수전공 | 주전공 외에 추가로 이수하는 전공 | 필수 아님 |
"""
    base_metadata = {
        MetadataFields.SOURCE_ID: "TEST_RULES_001",
        MetadataFields.SRC_NAME: "조선대학교_학칙_샘플.md",
        MetadataFields.DOC_TYPE: "markdown",
        MetadataFields.PG_NUM: 1,
    }

    # 2. 계층적 청킹
    chunks = create_parent_child_chunks(sample_markdown, base_metadata)

    flat_ids, flat_texts, flat_metadatas = [], [], []
    for parent in chunks:
        for child in parent["children"]:
            flat_ids.append(child["chunk_id"])
            flat_texts.append(child["text"])
            flat_metadatas.append(child["metadata"])

    # 3. ChromaDB 저장
    db_manager = ChromaDBManager(collection_name="test_e2e_collection")
    db_manager.upsert_documents(ids=flat_ids, documents=flat_texts, metadatas=flat_metadatas)

    # 4. RAG 체인 호출 및 검증
    test_query = "복수전공의 정의가 뭐야?"
    rag_chain = get_rag_chain(db_manager)

    response = rag_chain.invoke({"question": test_query, "k": 1})

    assert response is not None
    assert "복수전공" in response
    # 출처 인용 포함 여부 확인 (chains.py에서 결합됨)
    assert "조선대학교_학칙_샘플.md" in response
