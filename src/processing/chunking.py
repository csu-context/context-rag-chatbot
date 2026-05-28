import json
import re
import uuid
from typing import Any, cast

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.common.constants import MetadataFields
from src.common.schema import ChildChunk, ChunkMetadata, ParentChunk
from src.utils.paths import ensure_directories

_PARENT_CHUNK_SIZE: int = 1500
_CHILD_CHUNK_SIZE: int = 400


class MarkdownTableProtector:
    """마크다운 문서 내의 표(Table) 데이터를 식별, 보호 및 복원하는 유틸리티"""

    @staticmethod
    def split_markdown_table(table_text: str, max_size: int) -> list[str]:
        """마크다운 표가 max_size를 초과할 경우, 헤더(컬럼)를 유지하며 여러 표로 분할합니다."""
        lines = table_text.strip().split("\n")
        if len(lines) < 3:
            return [table_text]

        header = lines[0]
        separator = lines[1]
        data_rows = lines[2:]

        base_size = len(header) + len(separator) + 2

        chunks = []
        current_rows = []
        current_size = base_size

        for row in data_rows:
            row_size = len(row) + 1
            if current_size + row_size > max_size and current_rows:
                chunks.append("\n".join([header, separator, *current_rows]))
                current_rows = [row]
                current_size = base_size + row_size
            else:
                current_rows.append(row)
                current_size += row_size

        if current_rows:
            chunks.append("\n".join([header, separator, *current_rows]))

        return chunks

    @staticmethod
    def _register_token(text: str, registry: dict[str, str]) -> str:
        token_base = f"@@TABLE_{uuid.uuid4().hex}@@"
        padded_token = token_base + "_" * max(0, len(text) - len(token_base))
        registry[padded_token] = text
        return padded_token

    @staticmethod
    def protect_tables(text: str, max_chunk_size: int) -> tuple[str, dict[str, str]]:
        """표(Table) 데이터가 청킹 도중 잘리지 않도록 특수 토큰으로 일시 치환"""
        tables = {}

        table_pattern = re.compile(
            r"^[ \t]*\|?.*\|.*\n"
            r"^[ \t]*\|?[ \t]*[-:]+[ \t]*\|[ \t]*[-:]+.*(?:\n|$)"
            r"(?:^[ \t]*\|?.*\|.*(?:\n|$))*",
            re.MULTILINE,
        )

        def replace_with_token(match):
            table_text = match.group(0).strip()
            if len(table_text) > max_chunk_size:
                split_tables = MarkdownTableProtector.split_markdown_table(table_text, max_chunk_size)
                tokens = [MarkdownTableProtector._register_token(st, tables) for st in split_tables]
                return "".join(f"\n\n{t}\n\n" for t in tokens)
            return f"\n\n{MarkdownTableProtector._register_token(table_text, tables)}\n\n"

        protected_text = table_pattern.sub(replace_with_token, text)
        return protected_text, tables

    @staticmethod
    def restore_tables(text: str, tables: dict[str, str]) -> str:
        """패딩된 특수 토큰을 다시 원래 표 데이터로 복원"""
        for token in sorted(tables.keys(), key=len, reverse=True):
            text = text.replace(token, tables[token])
        return text


class HierarchicalChunker:
    def __init__(
        self,
        parent_chunk_size: int = _PARENT_CHUNK_SIZE,
        parent_chunk_overlap: int = 150,
        child_chunk_size: int = _CHILD_CHUNK_SIZE,
        child_chunk_overlap: int = 50,
        min_chunk_size: int = 50,
    ):
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        self.min_chunk_size = min_chunk_size

        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        self.md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on)

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.child_chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        )

    def _get_header_path(self, metadata: dict[str, str]) -> str:
        """마크다운 메타데이터에서 헤더 경로 생성 (예: 제1장 > 제1조 > 정의)"""
        path_parts = []
        for i in range(1, 5):
            header_val = metadata.get(f"Header {i}")
            if header_val:
                path_parts.append(header_val)
        return " > ".join(path_parts) if path_parts else "기본 섹션"

    def split_into_children(self, parent_text: str, parent_id: str, base_metadata: dict[str, Any]) -> list[ChildChunk]:
        """부모 텍스트를 자식 청크들로 분할하고 표준 메타데이터 상속"""
        protected_text, tables = MarkdownTableProtector.protect_tables(parent_text, self.child_chunk_size)
        child_docs = self.child_splitter.split_text(protected_text)

        merged_docs = []
        for doc_text in child_docs:
            doc_text = doc_text.strip()
            if not doc_text:
                continue

            # 짧은 문단 병합 (Minimum chunk size 적용)
            if merged_docs and len(doc_text) < self.min_chunk_size:
                merged_docs[-1] += "\n" + doc_text
            else:
                merged_docs.append(doc_text)

        children_list: list[ChildChunk] = []
        for idx, child_text in enumerate(merged_docs):
            # 복원 전 텍스트에 표 토큰이 포함되어 있다면 해당 청크는 표 데이터를 포함함을 의미함
            has_table = "@@TABLE_" in child_text

            restored_text = MarkdownTableProtector.restore_tables(child_text, tables).strip()
            if not restored_text:
                continue

            child_id = f"{parent_id}_c{idx}"

            child_metadata_dict: ChunkMetadata = {
                MetadataFields.SOURCE_ID: base_metadata.get(MetadataFields.SOURCE_ID, "UNKNOWN"),
                MetadataFields.SRC_NAME: base_metadata.get(MetadataFields.SRC_NAME, "UNKNOWN_FILE"),
                MetadataFields.DOC_TYPE: base_metadata.get(MetadataFields.DOC_TYPE, "markdown"),
                MetadataFields.PG_NUM: base_metadata.get(MetadataFields.PG_NUM, 1),
                MetadataFields.SEC_TITLE: base_metadata.get(MetadataFields.SEC_TITLE, "기본 섹션"),
                MetadataFields.CHUNK_ID: child_id,
                MetadataFields.PARENT_ID: parent_id,
                MetadataFields.HEADER_PATH: base_metadata.get(MetadataFields.HEADER_PATH, "기본 섹션"),
                MetadataFields.IS_TABLE: has_table,
                MetadataFields.PARSER: base_metadata.get(MetadataFields.PARSER, "manual"),
                MetadataFields.RELATIVE_PATH: base_metadata.get(MetadataFields.RELATIVE_PATH, "UNKNOWN"),
            }

            children_list.append(
                {
                    "chunk_id": child_id,
                    "metadata": child_metadata_dict,
                    "text": restored_text,
                }
            )

        return children_list

    def chunk(self, markdown_text: str, base_metadata: dict[str, Any]) -> list[ParentChunk]:
        """마크다운 텍스트를 계층적(Parent-Child)으로 분할합니다."""
        header_docs = self.md_splitter.split_text(markdown_text)
        hierarchical_data: list[ParentChunk] = []

        for doc in header_docs:
            if not doc.page_content.strip():
                continue

            header_path = self._get_header_path(doc.metadata)
            sec_title = (
                doc.metadata.get("Header 3")
                or doc.metadata.get("Header 2")
                or doc.metadata.get("Header 1")
                or "기본 섹션"
            )

            # 부모(Parent) 단위로 한 번 더 분할 (너무 긴 문맥 단위 처리)
            parent_splits = self.parent_splitter.split_text(doc.page_content)

            for p_text in parent_splits:
                parent_id = str(uuid.uuid4())

                meta_for_children = base_metadata.copy()
                meta_for_children[MetadataFields.SEC_TITLE] = sec_title
                meta_for_children[MetadataFields.HEADER_PATH] = header_path

                children_list = self.split_into_children(p_text, parent_id, meta_for_children)

                if not children_list:
                    continue

                parent_metadata: ChunkMetadata = {
                    MetadataFields.SOURCE_ID: base_metadata.get(MetadataFields.SOURCE_ID, "UNKNOWN"),
                    MetadataFields.SRC_NAME: base_metadata.get(MetadataFields.SRC_NAME, "UNKNOWN_FILE"),
                    MetadataFields.DOC_TYPE: base_metadata.get(MetadataFields.DOC_TYPE, "markdown"),
                    MetadataFields.PG_NUM: base_metadata.get(MetadataFields.PG_NUM, 1),
                    MetadataFields.SEC_TITLE: sec_title,
                    MetadataFields.CHUNK_ID: parent_id,
                    MetadataFields.PARENT_ID: None,
                    MetadataFields.HEADER_PATH: header_path,
                    MetadataFields.PARSER: base_metadata.get(MetadataFields.PARSER, "manual"),
                    MetadataFields.RELATIVE_PATH: base_metadata.get(MetadataFields.RELATIVE_PATH, "UNKNOWN"),
                }

                hierarchical_data.append(
                    {
                        "parent_id": parent_id,
                        "parent_text": p_text,
                        "metadata": parent_metadata,
                        "children": children_list,
                    }
                )

        return hierarchical_data


# 기존 코드 하위 호환성 래핑 함수
def create_parent_child_chunks(markdown_text: str, base_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    chunker = HierarchicalChunker()
    return cast(list[dict[str, Any]], chunker.chunk(markdown_text, base_metadata))


if __name__ == "__main__":
    ensure_directories()

    dummy_metadata = {
        MetadataFields.SOURCE_ID: "TEST_001",
        MetadataFields.SRC_NAME: "test_manual.md",
        MetadataFields.DOC_TYPE: "markdown",
        MetadataFields.PG_NUM: 1,
    }

    sample_text = """# 제1장
## 제1조 정의
이것은 테스트 문서입니다.

표 테스트:
| 항목 | 내용 |
|---|---|
| 1 | 테스트 1 |
| 2 | 테스트 2 |

매우 짧은 문단"""
    chunking_result = create_parent_child_chunks(sample_text, dummy_metadata)
    print(json.dumps(chunking_result, ensure_ascii=False, indent=2))
