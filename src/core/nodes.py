import logging
import re

from langchain_core.documents import Document

from src.common.config import settings
from src.common.constants import MetadataFields

logger = logging.getLogger(__name__)

_INJECTION_PATTERN = re.compile(
    r"(###|---+|\bSystem:|\bAssistant:|\bHuman:|\[INST\]|\[/INST\]|<\|system\|>|<\|user\|>)",
    re.IGNORECASE,
)

_TOKEN_RATIO: dict[str, float] = {
    "ollama": 1.5,
    "gemini": 1.3,
    "claude": 1.2,
}

_MAX_HISTORY_MESSAGES = 6
_CTX_USAGE_RATIO: float = 0.75


def estimate_tokens(text: str, ratio: float) -> int:
    return max(1, int(len(text) / ratio))


class ContextBuilderNode:
    """RAG 파이프라인의 컨텍스트 빌드, 샌드박싱, 캐시 쿼리 생성 등을 처리하는 노드 함수들"""

    @staticmethod
    def escape_injection(text: str) -> str:
        """문서 내 프롬프트 인젝션 유발 패턴을 이스케이프합니다."""
        return _INJECTION_PATTERN.sub(lambda m: f"[{m.group(0)}]", text)

    @classmethod
    def format_docs(cls, docs: list[Document]) -> str:
        """프롬프트 주입 방어 XML 샌드박싱 적용 컨텍스트 포맷팅"""
        formatted = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get(MetadataFields.SRC_NAME) or "알 수 없는 파일"
            page = doc.metadata.get(MetadataFields.PG_NUM) or "-"
            safe_content = cls.escape_injection(doc.page_content)
            entry = f'<document index="{i}">\n내용: {safe_content}\n출처: [{source}, p.{page}]\n</document>'
            formatted.append(entry)
        return "\n\n".join(formatted)

    @staticmethod
    def trim_docs_to_token_limit(docs: list[Document], system_prompt: str, history: list[dict]) -> list[Document]:
        """토큰 한도 초과 시 낮은 점수 문서부터 제거합니다."""
        limit = int(settings.OLLAMA_NUM_CTX * _CTX_USAGE_RATIO)
        ratio = _TOKEN_RATIO.get(settings.MODEL_TYPE, 1.5)

        fixed_tokens = estimate_tokens(system_prompt, ratio) + sum(
            estimate_tokens(m.get("content", ""), ratio) for m in history[-_MAX_HISTORY_MESSAGES:]
        )
        available = limit - fixed_tokens

        ranked = sorted(docs, key=lambda d: d.metadata.get("rerank_score", d.metadata.get("score", 0.0)), reverse=True)
        kept, total = [], 0
        for doc in ranked:
            doc_tokens = estimate_tokens(doc.page_content, ratio)
            if total + doc_tokens > available:
                continue  # 큰 문서 skip, 더 작은 나머지 문서는 계속 시도
            kept.append(doc)
            total += doc_tokens

        if len(kept) < len(docs):
            logger.warning(f"컨텍스트 토큰 한도 초과: {len(docs)}개 → {len(kept)}개 문서로 트리밍")

        return kept
