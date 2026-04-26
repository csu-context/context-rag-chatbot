from typing import TypedDict


class ChunkMetadata(TypedDict, total=False):
    source_id: str
    src_name: str
    doc_type: str | None
    pg_num: int
    sec_title: str
    chunk_id: str
    parent_id: str | None
    header_path: str


class ChildChunk(TypedDict):
    chunk_id: str
    metadata: ChunkMetadata
    text: str


class ParentChunk(TypedDict):
    parent_id: str
    parent_text: str
    metadata: ChunkMetadata
    children: list[ChildChunk]
