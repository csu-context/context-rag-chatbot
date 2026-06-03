import json
import logging
import os
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# 프로젝트 루트 경로 계산
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def _order_units(units: list[str]) -> list[str]:
    """중복 제거 후 길이 내림차순 정렬.

    정규식 교차(alternation)는 좌→우 우선 매칭이라 짧은 접두 단위가 앞서면 긴 단위를
    가린다(예: '학년'이 '학년도'보다 앞이면 '2 학년도'가 '학년'에 먼저 걸려 '도'가 남는다).
    길이 내림차순으로 고정해 긴 단위가 항상 먼저 매칭되도록 보장한다.
    """
    deduped = dict.fromkeys(u for u in units if u)
    return sorted(deduped, key=len, reverse=True)


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
    RERANKER_MODEL_NAME: str = Field(default="BAAI/bge-reranker-v2-m3")
    RERANKER_USE_FP16: bool = Field(default=True)
    EMBEDDER_USE_FP16: bool = Field(default=True)
    ALLOW_EXTERNAL_RERANKER: bool = Field(default=False)
    ALLOW_EXTERNAL_API: bool = Field(default=True)
    CHROMA_SERVER_HOST: str | None = None
    CHROMA_SERVER_PORT: str = "8000"
    RRF_K: int = 60
    HYBRID_WEIGHT_BM25: float = 0.5
    HYBRID_WEIGHT_VECTOR: float = 0.5
    RETRIEVAL_K: int = Field(default=50)
    RETRIEVER_CANDIDATE_POOL_MIN: int = Field(default=15)
    # 하이브리드 검색 두 leg 병렬 실행용 공유 스레드풀 크기 (동시 쿼리 x 2 leg 수용)
    RETRIEVER_EXECUTOR_MAX_WORKERS: int = Field(default=8)
    RERANKER_MAX_DOCS: int = Field(default=5)
    RERANKER_BATCH_SIZE: int = Field(default=5)
    RERANKER_THRESHOLD: float = Field(default=0.5)
    RERANKER_TIMEOUT_SEC: int = Field(default=5)
    RERANKER_GPU_RECOVERY_INTERVAL_SEC: int = Field(default=300)
    MAX_INGESTION_WORKERS: int = Field(default=4)
    MAX_PICKLE_CACHE_FILES: int = Field(default=50)
    TABLE_RETRIEVAL_ENABLED: bool = Field(default=True)
    # Redis 기반 세션 스토어 (수평 확장용, 없으면 in-memory 폴백)
    REDIS_URL: str | None = Field(default=None)
    REDIS_SESSION_TTL_SEC: int = Field(default=86400)  # 세션 TTL 24시간

    # 모니터링/트레이싱 설정 (LangSmith)
    LANGSMITH_TRACING: bool = Field(default=False)

    # 4. 파이프라인 및 파싱 설정
    PARSER_TYPE: Literal["manual", "docling"] = Field(default="docling")
    DOC_TYPE: Literal["legal", "general"] = Field(default="legal")
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    # PDF 헤더/푸터 동적 제거 임계치: 전체 페이지 중 이 비율 이상 반복되는 라인을 노이즈로 분류
    PDF_HEADER_FOOTER_THRESHOLD: float = Field(default=0.8)
    # PDF 잔여 노이즈 패턴 (범용 정규식). 환경 변수로 오버라이드 가능: PDF_NOISE_PATTERNS='["pattern1"]'
    PDF_NOISE_PATTERNS: list[str] = Field(
        default=[
            r"(?im)^\s*-\s*\d+\s*-\s*$",  # 페이지 번호 (예: "- 1 -", "- 12 -")
        ]
    )
    # 표 컨텍스트 추출 시 페이지 상단에서 제거할 줄 수 (헤더 스킵)
    PDF_CONTEXT_HEADER_LINES: int = Field(default=1)
    # 표 컨텍스트로 사용할 최대 줄 수
    PDF_CONTEXT_WINDOW_LINES: int = Field(default=5)
    # 숫자+단위 재결합 대상 한국어 단위. 코드 기본값은 범용 카운터(개·명·시간 등)만 둔다.
    # DOC_TYPE="legal"에서만 적용된다. 학사·법령 등 도메인 종속 단위는 코드에 하드코딩하지 않고
    # data/config/numeric_units.json(JSON 배열)로 분리해 주입한다(_merge_external_numeric_units).
    # 우선순위: 환경변수 KOREAN_NUMERIC_UNITS='["..."]' 를 명시하면 그 값이 전부이고 파일은 무시한다.
    # 매칭 순서(긴 단위 우선)는 _order_units가 길이 내림차순으로 자동 보정한다.
    KOREAN_NUMERIC_UNITS: list[str] = Field(
        default=[
            "개월",
            "단계",
            "등급",
            "시간",
            "개",
            "월",
            "일",
            "년",
            "번",
            "회",
            "차",
            "주",
            "명",
            "원",
            "점",
            "편",
            "권",
            "절",
            "관",
            "목",
        ]
    )

    # 5. 평가(Evaluation) 관련 설정
    EVAL_MAX_SAMPLES: int = 50
    EVAL_JUDGE_TYPE: str = "claude"
    EVAL_JUDGE_MODEL: str = "claude-haiku-4-5"
    EVAL_DATA_GEN_TYPE: str = "claude"
    EVAL_DATA_GEN_MODEL: str = "claude-haiku-4-5"
    EVAL_DATA_GEN_SAMPLES: int = Field(default=50)
    EVAL_MAX_WORKERS: int = Field(default=2)

    # 6. 인프라 및 경로 설정
    LOG_LEVEL: str = "INFO"
    CI: bool = False

    # 7. 시맨틱 캐시 설정
    SEMANTIC_CACHE_COLLECTION_NAME: str = Field(default="semantic_cache")
    SEMANTIC_CACHE_THRESHOLD: float = Field(default=0.95)
    MAX_CHAT_HISTORY_TURNS: int = Field(default=5)

    # 8. Ollama 추론 제어
    OLLAMA_KEEP_ALIVE: int | str = Field(default=-1)
    OLLAMA_NUM_PREDICT: int = Field(default=2048)
    OLLAMA_REPEAT_PENALTY: float = Field(default=1.1)
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

    @model_validator(mode="after")
    def _merge_external_numeric_units(self) -> "Settings":
        """도메인 종속 숫자 단위를 외부 파일(data/config/numeric_units.json)에서 병합한다.

        - 환경변수 KOREAN_NUMERIC_UNITS를 명시하면 그 값이 전부이고 파일은 무시한다(명시 우선).
        - 파일이 없거나 비면 코드 기본(범용 단위)만 사용한다(범용 매뉴얼 RAG 기본 동작).
        - NUMERIC_UNITS_PATH 환경변수로 외부 볼륨 경로를 오버라이드할 수 있다.
        - 결과는 길이 내림차순으로 정렬해 정규식 매칭 정확성을 보장한다(_order_units).
        """
        if "KOREAN_NUMERIC_UNITS" in self.model_fields_set:
            self.KOREAN_NUMERIC_UNITS = _order_units(self.KOREAN_NUMERIC_UNITS)
            return self

        units_file = Path(os.getenv("NUMERIC_UNITS_PATH") or (ROOT_DIR / "data" / "config" / "numeric_units.json"))
        merged = list(self.KOREAN_NUMERIC_UNITS)
        if units_file.exists():
            try:
                extra = json.loads(units_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("numeric_units.json 로드 실패, 기본 단위만 사용: %s", exc)
            else:
                if isinstance(extra, list):
                    merged.extend(str(u) for u in extra)
                else:
                    logger.warning("numeric_units.json 형식 오류(list가 아님), 무시")
        self.KOREAN_NUMERIC_UNITS = _order_units(merged)
        return self


# 전역 설정 객체 생성
settings = Settings()

if settings.ALLOW_EXTERNAL_API:
    logger.warning(
        "보안 경고: 외부 API 호출이 허용되어 있습니다 (ALLOW_EXTERNAL_API=True). "
        "폐쇄망(On-premise) 환경에서는 ALLOW_EXTERNAL_API=False 설정을 권장합니다."
    )
