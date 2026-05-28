import logging
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

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
    RERANKER_MODEL_NAME: str = Field(default="BAAI/bge-reranker-base")
    RERANKER_USE_FP16: bool = Field(default=True)
    EMBEDDER_USE_FP16: bool = Field(default=True)
    ALLOW_EXTERNAL_RERANKER: bool = Field(default=False)
    ALLOW_EXTERNAL_API: bool = Field(default=True)
    CHROMA_SERVER_HOST: str | None = None
    CHROMA_SERVER_PORT: str = "8000"
    RRF_K: int = 60
    HYBRID_WEIGHT_BM25: float = 0.5
    HYBRID_WEIGHT_VECTOR: float = 0.5
    RETRIEVER_CANDIDATE_POOL_MIN: int = Field(default=15)
    RERANKER_MAX_DOCS: int = Field(default=10)
    RERANKER_BATCH_SIZE: int = Field(default=8)

    # 4. 파이프라인 및 파싱 설정
    PARSER_TYPE: Literal["manual", "docling"] = Field(default="docling")
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    # PDF 헤더/푸터 동적 제거 임계치: 전체 페이지 중 이 비율 이상 반복되는 라인을 노이즈로 분류
    PDF_HEADER_FOOTER_THRESHOLD: float = Field(default=0.8)
    # PDF 잔여 노이즈 패턴 (범용 정규식). 환경 변수로 오버라이드 가능: PDF_NOISE_PATTERNS='["pattern1"]'
    PDF_NOISE_PATTERNS: list[str] = Field(
        default=[
            r"(?im)^\s*-\s*\d+\s*-\s*$",  # 페이지 번호 (예: "- 1 -", "- 12 -")
        ]
    )

    # 5. 평가(Evaluation) 관련 설정
    EVAL_MAX_SAMPLES: int = 50
    EVAL_JUDGE_TYPE: str = "claude"
    EVAL_JUDGE_MODEL: str = "claude-haiku-4-5"
    EVAL_DATA_GEN_TYPE: str = "claude"
    EVAL_DATA_GEN_MODEL: str = "claude-haiku-4-5"

    # 6. 인프라 및 경로 설정
    LOG_LEVEL: str = "INFO"
    CI: bool = False

    # 7. 시맨틱 캐시 설정
    SEMANTIC_CACHE_COLLECTION_NAME: str = Field(default="semantic_cache")
    SEMANTIC_CACHE_THRESHOLD: float = Field(default=0.90)
    MAX_CHAT_HISTORY_TURNS: int = Field(default=5)

    # 8. Ollama 추론 제어
    OLLAMA_KEEP_ALIVE: int | str = Field(default=-1)
    OLLAMA_NUM_PREDICT: int = Field(default=8192)
    OLLAMA_REPEAT_PENALTY: float = Field(default=1.0)
    OLLAMA_NUM_CTX: int = Field(default=8192)
    OLLAMA_THINK: bool = Field(default=False)

    # 9. LLM 프롬프트 및 Fallback
    PROMPT_FILE: str | None = Field(default=None)
    LLM_FALLBACK_ENABLED: bool = Field(default=True)

    # Pydantic 설정 (env 파일 로드 및 대소문자 무시)
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# 전역 설정 객체 생성
settings = Settings()

if settings.ALLOW_EXTERNAL_API:
    logger.warning(
        "보안 경고: 외부 API 호출이 허용되어 있습니다 (ALLOW_EXTERNAL_API=True). "
        "폐쇄망(On-premise) 환경에서는 ALLOW_EXTERNAL_API=False 설정을 권장합니다."
    )
