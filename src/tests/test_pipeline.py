import os
import sys
from pathlib import Path

# KMP 에러 방지 (윈도우 환경)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# 프로젝트 루트 경로를 sys.path에 추가 (tests 폴더 안에서도 src를 찾을 수 있게 함)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.common.constants import MetadataFields
from src.processing.chunking import create_parent_child_chunks
from src.vector_db.chroma_manager import ChromaDBManager


def run_integration_test():
    print("🚀 [1단계] 테스트 데이터를 준비합니다...")

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

아주 짧은 문단입니다. (병합이 잘 되는지 확인)
"""

    # [리뷰 반영] 문자열 하드코딩 대신 상수 사용
    base_metadata = {
        MetadataFields.SOURCE_ID: "TEST_RULES_001",
        MetadataFields.SRC_NAME: "조선대학교_학칙_샘플.md",
        MetadataFields.DOC_TYPE: "markdown",
        MetadataFields.PG_NUM: 1,
    }

    print("\n✅ [2단계] 계층적 청킹(Hierarchical Chunking)을 시작합니다...")
    chunks = create_parent_child_chunks(sample_markdown, base_metadata)

    # 자식 청크 평탄화
    flat_ids = []
    flat_texts = []
    flat_metadatas = []

    for parent in chunks:
        for child in parent["children"]:
            flat_ids.append(child["chunk_id"])
            flat_texts.append(child["text"])
            flat_metadatas.append(child["metadata"])

    print(f"✅ 총 {len(flat_texts)}개의 자식 청크가 생성되었습니다.")

    print("\n🗄️ [3단계] ChromaDB에 데이터를 저장(Upsert)합니다...")
    db_manager = ChromaDBManager(collection_name="test_hierarchical_collection")
    db_manager.upsert_documents(ids=flat_ids, documents=flat_texts, metadatas=flat_metadatas)
    print("✅ DB 저장 완료!")

    print("\n================== 🔍 [4단계] 유사도 검색 결과 ==================")
    test_query = "복수전공의 정의가 뭐야?"
    print(f"[질문]: {test_query}")

    results = db_manager.search(query_text=test_query, k=1)

    if results:
        res = results[0]
        print(f"  -> Score: {res['score']:.4f}")
        # [리뷰 반영] 상수 적용
        print(f"  -> 계층 경로(Header Path): {res['metadata'].get(MetadataFields.HEADER_PATH, '없음')}")
        print(f"  -> 내용:\n{res['content']}")
    else:
        print("  ⚠️ 검색 결과가 없습니다.")
    print("=================================================================")


if __name__ == "__main__":
    run_integration_test()