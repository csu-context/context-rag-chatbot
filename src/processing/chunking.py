import json
import uuid
from typing import Any, TypedDict, cast

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.common.constants import MetadataFields
from src.utils.paths import ensure_directories


# 청크 단위 메타데이터 스키마 정의 (표준 규격 준수)
class ChunkMetadata(TypedDict):
    source_id: str
    src_name: str
    doc_type: str | None
    pg_num: int
    sec_title: str
    chunk_id: str
    parent_id: str | None


# 자식 청크 분할을 위한 공통 스플리터 설정
def get_child_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])


def split_into_children(parent_text: str, parent_id: str, base_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """
    부모 텍스트를 자식 청크들로 분할하고 표준 메타데이터를 입힙니다.
    """
    child_splitter = get_child_splitter()
    child_docs = child_splitter.split_text(parent_text)

    children_list = []
    for idx, child_text in enumerate(child_docs):
        child_id = f"{parent_id}_c{idx}"

        # [타입 보정] 엄격한 타입 체킹을 위해 object를 거쳐 ChunkMetadata로 캐스팅
        child_metadata_dict = {
            "source_id": base_metadata.get(MetadataFields.SOURCE_ID, "UNKNOWN"),
            "src_name": base_metadata.get(MetadataFields.SRC_NAME, "UNKNOWN_FILE"),
            "doc_type": base_metadata.get(MetadataFields.DOC_TYPE, "markdown"),
            "pg_num": base_metadata.get(MetadataFields.PG_NUM, 1),
            "sec_title": base_metadata.get(MetadataFields.SEC_TITLE, "기본 섹션"),
            "chunk_id": child_id,
            "parent_id": parent_id,
        }
        child_metadata = cast(ChunkMetadata, cast(object, child_metadata_dict))

        children_list.append(
            {
                MetadataFields.CHUNK_ID: child_id,
                "metadata": child_metadata,
                "text": child_text,
            }
        )
    return children_list


def create_parent_child_chunks(markdown_text: str, base_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """
    마크다운 텍스트를 계층적(Parent-Child)으로 분할합니다.
    """
    # 1. 부모 청크: 마크다운 헤더 기준 분할
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    parent_docs = markdown_splitter.split_text(markdown_text)

    hierarchical_data = []

    for doc in parent_docs:
        if not doc.page_content.strip():
            continue

        parent_id = str(uuid.uuid4())
        # [수정] 너무 긴 라인 분할
        sec_title = (
            doc.metadata.get("Header 3") or doc.metadata.get("Header 2") or doc.metadata.get("Header 1") or "기본 섹션"
        )

        # 공통 자식 분할 로직 호출
        meta_for_children = base_metadata.copy()
        meta_for_children[MetadataFields.SEC_TITLE] = sec_title

        children_list = split_into_children(doc.page_content, parent_id, meta_for_children)

        # 부모 데이터 구조 완성
        hierarchical_data.append(
            {
                MetadataFields.PARENT_ID: parent_id,
                "parent_text": doc.page_content,
                "metadata": {
                    MetadataFields.SOURCE_ID: base_metadata.get(MetadataFields.SOURCE_ID, "UNKNOWN"),
                    MetadataFields.SRC_NAME: base_metadata.get(MetadataFields.SRC_NAME, "UNKNOWN_FILE"),
                    MetadataFields.DOC_TYPE: base_metadata.get(MetadataFields.DOC_TYPE, "markdown"),
                    MetadataFields.PG_NUM: base_metadata.get(MetadataFields.PG_NUM, 1),
                    MetadataFields.SEC_TITLE: sec_title,
                },
                "children": children_list,
            }
        )

    return hierarchical_data


if __name__ == "__main__":
    # 필수 디렉토리 확인 및 생성
    ensure_directories()

    # 테스트용 더미 메타데이터
    dummy_metadata = {
        MetadataFields.SOURCE_ID: "TEST_001",
        MetadataFields.SRC_NAME: "test_manual.md",
        MetadataFields.DOC_TYPE: "markdown",
        MetadataFields.PG_NUM: 1,
    }

    sample_text = "# 테스트\n## 섹션 1\n내용입니다."
    chunking_result = create_parent_child_chunks(sample_text, dummy_metadata)
    print(json.dumps(chunking_result, ensure_ascii=False, indent=2))
