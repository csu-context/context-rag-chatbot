from typing import Final

class MetadataFields:
    """
    metadata_schema.json 규격에 정의된 표준 메타데이터 필드명 상수
    """
    SOURCE_ID: Final[str] = "source_id"
    SRC_NAME: Final[str] = "src_name"
    DOC_TYPE: Final[str] = "doc_type"
    PG_NUM: Final[str] = "pg_num"
    CATEGORY: Final[str] = "category"
    
    # 계층적 청킹 관련
    PARENT_ID: Final[str] = "parent_id"
    CHUNK_ID: Final[str] = "chunk_id"
    SEC_TITLE: Final[str] = "sec_title"
    CONTENT_PREVIEW: Final[str] = "content_preview"
