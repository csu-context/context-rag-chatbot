from typing import Final


class MetadataFields:
    """
    metadata_schema.json 규격에 정의된 표준 메타데이터 필드명 상수
    """

    SOURCE_ID: Final[str] = "source_id"
    SRC_NAME: Final[str] = "src_name"
    RELATIVE_PATH: Final[str] = "relative_path"
    PARSER_TYPE: Final[str] = "parser_type"
    PARSER: Final[str] = "parser_type"
    DOC_TYPE: Final[str] = "doc_type"
    PG_NUM: Final[str] = "pg_num"
    CATEGORY: Final[str] = "category"

    # 계층적 청킹 관련
    PARENT_ID: Final[str] = "parent_id"
    CHUNK_ID: Final[str] = "chunk_id"
    SEC_TITLE: Final[str] = "sec_title"
    HEADER_PATH: Final[str] = "header_path"
    IS_TABLE: Final[str] = "is_table"


class DataFields:
    """
    가공된 JSON 데이터의 최상위 필드명 상수
    """

    TEXT: Final[str] = "text"
    CONTENT: Final[str] = "content"  # 기존 호환성 유지용
    CHILDREN: Final[str] = "children"
    PARENT_TEXT: Final[str] = "parent_text"
    METADATA: Final[str] = "metadata"


class LLMDefaults:
    """LLM 관련 기본 설정 및 모델 명 상수"""

    CLAUDE_DEFAULT: Final[str] = "claude-sonnet-4-6"
    GEMINI_DEFAULT: Final[str] = "gemini-2.0-flash"
    OLLAMA_DEFAULT: Final[str] = "gemma4:e4b"
    TEMPERATURE: Final[float] = 0.1


class LLMPricing:
    """모델별 1M 토큰 당 단가 (USD) - 비용 계산용"""

    PRICING: Final[dict[str, dict[str, float]]] = {
        "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
        "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
        "gemini-2.0-flash": {"input": 0.1, "output": 0.4},
    }


class SupportedFormats:
    """지원하는 원본 파일의 확장자 목록"""

    EXTENSIONS: Final[list[str]] = [".pdf", ".md", ".markdown", ".hwp", ".hwpx"]

