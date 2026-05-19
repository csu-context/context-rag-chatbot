from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 프로젝트 루트 경로 계산
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """시스템 전역 설정을 관리하는 클래스"""

    # 1. LLM 및 모델 설정
    MODEL_TYPE: Literal["gemini", "claude", "ollama"] = Field(default="ollama")
    MODEL_NAME: str = Field(default="llama3.2:1b")
    EMBEDDING_MODEL_NAME: str = Field(default="BAAI/bge-m3")
    TEMPERATURE: float = Field(default=0.1)

    # 2. API 키 (필수 항목은 아니지만 운영 시 필요)
    GEMINI_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    COHERE_API_KEY: str | None = None
    JINA_API_KEY: str | None = None

    # 3. 벡터 DB 및 검색 설정
    RETRIEVER_TYPE: Literal["vector", "hybrid"] = Field(default="hybrid")
    RERANKER_TYPE: Literal["local", "cohere", "jina"] = Field(default="local")
    ALLOW_EXTERNAL_RERANKER: bool = Field(default=False)
    CHROMA_SERVER_HOST: str | None = None
    CHROMA_SERVER_PORT: str = "8000"
    RRF_K: int = 60
    HYBRID_WEIGHT_BM25: float = 0.5
    HYBRID_WEIGHT_VECTOR: float = 0.5

    # 4. 파이프라인 및 파싱 설정
    PARSER_TYPE: Literal["manual", "docling"] = Field(default="docling")
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # 5. 평가(Evaluation) 관련 설정
    EVAL_MAX_SAMPLES: int = 50
    EVAL_JUDGE_TYPE: str = "claude"
    EVAL_JUDGE_MODEL: str = "claude-haiku-4-5"
    EVAL_DATA_GEN_TYPE: str = "claude"
    EVAL_DATA_GEN_MODEL: str = "claude-haiku-4-5"

    # 6. 인프라 및 경로 설정
    LOG_LEVEL: str = "INFO"
    CI: bool = False

    # Pydantic 설정 (env 파일 로드 및 대소문자 무시)
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# 전역 설정 객체 생성
settings = Settings()
