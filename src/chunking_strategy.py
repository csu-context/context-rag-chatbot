import json
import uuid
from typing import Optional, TypedDict

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.utils.paths import PROCESSED_DATA_DIR, ensure_directories


# 청크 단위 메타데이터 스키마 정의
class ChunkMetadata(TypedDict):
    doc_id: str
    src_name: str
    src_type: Optional[str]
    pg_num: int
    sec_title: str
    chunk_id: str
    parent_id: Optional[str]


def create_parent_child_chunks(markdown_text: str, base_metadata: dict) -> list:
    """
    마크다운 텍스트를 계층적(Parent-Child)으로 분할하고 메타데이터를 매핑합니다.
    """

    # 1. 부모 청크: 마크다운 헤더 기준 분할
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    parent_docs = markdown_splitter.split_text(markdown_text)

    # 2. 자식 청크: 글자 수 기준 세부 분할
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )

    hierarchical_data = []

    for doc in parent_docs:
        # [FIX] MarkdownHeaderTextSplitter는 헤더와 다음 헤더 사이에 텍스트가 없는 경우,
        # page_content가 비어있는 Document를 생성할 수 있습니다.
        # 이러한 빈 문서는 건너뛰어 불필요한 데이터 생성을 방지합니다.
        if not doc.page_content.strip():
            continue

        parent_id = str(uuid.uuid4())

        # 가장 구체적인 헤더부터 제목을 찾습니다 (H3 -> H2 -> H1).
        sec_title = (
            doc.metadata.get("Header 3") or doc.metadata.get("Header 2") or doc.metadata.get("Header 1") or "기본 섹션"
        )

        child_docs = child_splitter.split_text(doc.page_content)

        children_list = []
        for child_text in child_docs:
            child_id = str(uuid.uuid4())

            # 자식 청크에 기본 메타데이터 및 계층 ID(parent_id, chunk_id) 병합
            child_metadata: ChunkMetadata = {
                "doc_id": base_metadata.get("doc_id", "UNKNOWN"),
                "src_name": base_metadata.get("src_name", "UNKNOWN_FILE"),
                "src_type": base_metadata.get("src_type"),
                "pg_num": base_metadata.get("pg_num", 1),
                "sec_title": sec_title,
                "chunk_id": child_id,
                "parent_id": parent_id,
            }

            children_list.append({"child_id": child_id, "metadata": child_metadata, "text": child_text})

        # 부모 데이터 구조 완성
        hierarchical_data.append({"parent_id": parent_id, "parent_text": doc.page_content, "children": children_list})

    return hierarchical_data


if __name__ == "__main__":
    # 필수 디렉토리 확인 및 생성
    ensure_directories()

    # 더미 데이터 세팅 (파서 연동 전 단독 테스트용)
    sample_text = """
    # 신규 입사자 가이드

    ## 1. 출퇴근 규정
    정규 출근 시간은 오전 9시이며, 퇴근 시간은 오후 6시입니다.
    """

    dummy_metadata = {"doc_id": "HR_001", "src_name": "인사규정_2026.pdf", "src_type": "pdf", "pg_num": 12}

    # 청킹 파이프라인 실행
    chunking_result = create_parent_child_chunks(sample_text, dummy_metadata)

    # 결과물 JSON 저장
    output_path = PROCESSED_DATA_DIR / "chunked_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunking_result, f, ensure_ascii=False, indent=4)

    print(f"✅ 메타데이터가 적용된 청킹 완료! 경로: {output_path}")
