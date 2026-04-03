import json
import uuid
from typing import TypedDict, Optional
# 팀원이 만든 경로 유틸리티 모듈 임포트
from src.utils.paths import PROCESSED_DATA_DIR, ensure_directories
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


# 1. 우리가 설계한 메타데이터의 뼈대(타입)를 코드에 정의합니다.
class ChunkMetadata(TypedDict):
    doc_id: str
    src_name: str
    src_type: Optional[str]
    pg_num: int
    sec_title: str
    chunk_id: str
    parent_id: Optional[str]


# 2. 파서(Parser)로부터 넘겨받을 '기본 문서 정보(base_metadata)' 변수를 하나 더 추가로 받습니다.
def create_parent_child_chunks(markdown_text: str, base_metadata: dict) -> list:
    """마크다운 텍스트를 입력받아 부모-자식 청크 관계와 메타데이터가 명시된 리스트를 반환합니다."""

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    parent_docs = markdown_splitter.split_text(markdown_text)

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    hierarchical_data = []

    for doc in parent_docs:
        parent_id = str(uuid.uuid4())

        # 마크다운 헤더(섹션 제목)를 가져옵니다. (없으면 기본값 사용)
        sec_title = doc.metadata.get("Header 2", doc.metadata.get("Header 1", "기본 섹션"))

        child_docs = child_splitter.split_text(doc.page_content)

        children_list = []
        for child_text in child_docs:
            child_id = str(uuid.uuid4())

            # 3. 여기서 자식 청크 하나하나마다 완벽한 메타데이터 이름표를 붙여줍니다!
            child_metadata: ChunkMetadata = {
                "doc_id": base_metadata.get("doc_id", "UNKNOWN"),
                "src_name": base_metadata.get("src_name", "UNKNOWN_FILE"),
                "src_type": base_metadata.get("src_type"),
                "pg_num": base_metadata.get("pg_num", 1),
                "sec_title": sec_title,
                "chunk_id": child_id,
                "parent_id": parent_id
            }

            children_list.append({
                "child_id": child_id,
                "metadata": child_metadata,  # 완성된 이름표 부착
                "text": child_text
            })

        hierarchical_data.append({
            "parent_id": parent_id,
            "parent_text": doc.page_content,
            "children": children_list
        })

    return hierarchical_data


if __name__ == "__main__":
    ensure_directories()

    # 임시 테스트용 마크다운 데이터
    sample_text = """
    # 신규 입사자 가이드

    ## 1. 출퇴근 규정
    정규 출근 시간은 오전 9시이며, 퇴근 시간은 오후 6시입니다.
    """

    # 테스트를 위해 가짜 메타데이터를 하나 만들어 줍니다. (나중에는 파서가 이 정보를 넘겨줍니다)
    dummy_metadata = {
        "doc_id": "HR_001",
        "src_name": "인사규정_2026.pdf",
        "src_type": "pdf",
        "pg_num": 12
    }

    # 텍스트와 가짜 메타데이터를 같이 집어넣고 실행!
    chunking_result = create_parent_child_chunks(sample_text, dummy_metadata)

    output_path = PROCESSED_DATA_DIR / "chunked_result.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunking_result, f, ensure_ascii=False, indent=4)

    print(f"✅ 메타데이터가 적용된 청킹 완료! 경로: {output_path}")