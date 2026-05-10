import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import requests
import torch
from langchain_core.documents import Document

from src.utils.paths import CROSS_ENCODER_CACHE_DIR

# sentence_transformers는 선택적 의존성이므로, 필요할 때만 import 시도
try:
    from sentence_transformers.cross_encoder import CrossEncoder
except ImportError:
    CrossEncoder = None

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """리랭킹 결과를 담는 데이터 클래스"""

    documents: list[Document]
    scores: list[float]
    filtered_count: int = 0
    elapsed_time_sec: float = 0.0


class BaseReranker(ABC):
    """리랭커 기본 클래스"""

    MAX_INFER_TIME_SEC = 5  # API 타임아웃
    PERFORMANCE_THRESHOLD_SEC = 3.0  # 지연 기준 시간 (이 시간 초과 시 top_k 동적 조정)

    def __init__(self, top_k: int = 5, threshold: float = 0.3):
        self.top_k = top_k
        self.threshold = threshold
        self._last_latency = 0.0

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> RerankResult:
        """문서 목록을 재정렬하고 관련성이 높은 순으로 반환합니다."""
        pass

    def _adjust_top_k(self, top_k: int) -> int:
        """이전 추론 지연 시간에 따라 top_k를 동적으로 조정합니다."""
        if self._last_latency > self.PERFORMANCE_THRESHOLD_SEC:
            adjusted = max(1, top_k // 2)
            model_ident = getattr(self, "name", self.__class__.__name__)
            logger.warning(
                f"[{model_ident}] High latency detected ({self._last_latency:.2f}s). "
                f"Adjusting top_k from {top_k} to {adjusted}."
            )
            return adjusted
        return top_k

    def rerank_with_timeout(self, query: str, documents: list[Document], **kwargs: Any) -> RerankResult:
        """
        타임아웃, 예외 처리 및 성능 모니터링을 포함한 리랭킹을 수행합니다.
        실패 시 원본 문서 목록의 일부를 그대로 반환합니다.
        """
        target_top_k = kwargs.get("top_k") or self.top_k
        adjusted_top_k = self._adjust_top_k(target_top_k)
        kwargs["top_k"] = adjusted_top_k

        start_time = time.time()
        try:
            result = self.rerank(query, documents, **kwargs)
            self._last_latency = time.time() - start_time
            return result
        except Exception as e:
            self._last_latency = time.time() - start_time
            model_ident = getattr(self, "name", self.__class__.__name__)
            logger.warning(f"Reranking failed in {model_ident}: {e}. Returning original documents.")
            # 실패 시 원본 문서에서 top_k만큼 잘라서 반환
            return RerankResult(
                documents=documents[:adjusted_top_k],
                scores=[0.0] * min(len(documents), adjusted_top_k),
                filtered_count=max(0, len(documents) - adjusted_top_k),
            )


class CrossEncoderReranker(BaseReranker):
    """로컬 모델 (BGE-Reranker, MS-MARCO 등) 기반 리랭커 싱글톤."""

    _instance: Optional["CrossEncoderReranker"] = None
    _model: CrossEncoder | None = None
    _singleton_lock = threading.Lock()

    DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        top_k: int = 5,
        threshold: float | None = None,
        device: str | None = None,
    ):
        super().__init__(top_k, threshold or 0.3)
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.name = "Local CrossEncoder"

        if threshold is None:
            self._set_model_defaults()

    def _set_model_defaults(self) -> None:
        model_lower = self.model_name.lower()
        if "bge" in model_lower:
            self.threshold = 0.2
        elif "skesarmom" in model_lower or "kor" in model_lower:
            self.threshold = 0.4
        else:
            self.threshold = 0.3

    @classmethod
    def get_instance(
        cls,
        model_name: str = DEFAULT_MODEL_NAME,
        top_k: int = 5,
        threshold: float | None = None,
        device: str | None = None,
    ) -> "CrossEncoderReranker":
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = cls(model_name, top_k, threshold, device)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """테스트용 싱글톤 리셋"""
        with cls._singleton_lock:
            cls._instance = None

    def _load_model(self) -> CrossEncoder:
        if self._model is not None:
            return self._model
        if CrossEncoder is None:
            raise ImportError("sentence_transformers is not installed.")

        with self._singleton_lock:
            if self._model is not None:
                return self._model
            try:
                self._model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                    cache_dir=str(CROSS_ENCODER_CACHE_DIR),
                )
            except Exception as e:
                logger.error(f"Failed to load CrossEncoder: {e}")
                raise
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> RerankResult:
        effective_top_k = top_k or self.top_k
        effective_threshold = threshold if threshold is not None else self.threshold

        if len(documents) < 2:
            return RerankResult(documents=documents, scores=[0.5] * len(documents))

        pairs = [(query, doc.page_content) for doc in documents]
        model = self._load_model()

        start_time = time.time()
        scores = model.predict(pairs).tolist()
        elapsed_time = time.time() - start_time

        scored_docs = sorted(zip(scores, documents, strict=True), key=lambda x: x[0], reverse=True)
        filtered = [(s, d) for s, d in scored_docs if s >= effective_threshold]
        ranked = filtered[:effective_top_k]

        return RerankResult(
            documents=[d for _, d in ranked],
            scores=[s for s, _ in ranked],
            filtered_count=len(documents) - len(ranked),
            elapsed_time_sec=elapsed_time,
        )


class APIBaseReranker(BaseReranker):
    """API 기반 리랭커 공통 로직 추상화 클래스"""

    def __init__(self, api_key: str | None, model_name: str, api_url: str, top_k: int, threshold: float, name: str):
        super().__init__(top_k, threshold)
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = api_url
        self.name = name

    def _build_payload(self, query: str, documents: list[Document], top_k: int) -> dict:
        return {
            "model": self.model_name,
            "query": query,
            "documents": [doc.page_content for doc in documents],
            "top_n": top_k,
        }

    @staticmethod
    def _parse_results(result: dict) -> list[dict]:
        return result.get("results", [])

    @staticmethod
    def _extract_score(result_item: dict) -> float:
        """API 결과 항목에서 관련도 점수를 추출합니다. (하위 클래스에서 오버라이딩 가능)"""
        return float(result_item.get("relevance_score", 0.0))

    @staticmethod
    def _extract_index(result_item: dict) -> int:
        """API 결과 항목에서 원본 문서 인덱스를 추출합니다. (하위 클래스에서 오버라이딩 가능)"""
        return int(result_item.get("index", -1))

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> RerankResult:
        effective_top_k = top_k or self.top_k
        effective_threshold = threshold if threshold is not None else self.threshold

        if not documents:
            return RerankResult(documents=[], scores=[])

        if not self.api_key:
            logger.warning(f"{self.name}_API_KEY is not set. Failing over.")
            raise ValueError(f"API Key missing for {self.name}")

        start_time = time.time()
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = self._build_payload(query, documents, effective_top_k)

        response = requests.post(self.api_url, headers=headers, json=payload, timeout=self.MAX_INFER_TIME_SEC)
        response.raise_for_status()

        result = response.json()
        elapsed_time = time.time() - start_time

        results = self._parse_results(result)
        final_docs, final_scores = [], []

        for res in results:
            idx = self._extract_index(res)
            score = self._extract_score(res)
            if idx != -1 and score >= effective_threshold:
                final_docs.append(documents[idx])
                final_scores.append(score)

        return RerankResult(
            documents=final_docs[:effective_top_k],
            scores=final_scores[:effective_top_k],
            filtered_count=len(documents) - len(final_docs[:effective_top_k]),
            elapsed_time_sec=elapsed_time,
        )


class CohereReranker(APIBaseReranker):
    """Cohere API 기반 리랭커."""

    def __init__(self, api_key: str | None = None, top_k: int = 5, threshold: float = 0.3):
        super().__init__(
            api_key=api_key or os.getenv("COHERE_API_KEY"),
            model_name="rerank-multilingual-v3.0",
            api_url="https://api.cohere.ai/v1/rerank",
            top_k=top_k,
            threshold=threshold,
            name="Cohere",
        )

    def _build_payload(self, query: str, documents: list[Document], top_k: int) -> dict:
        payload = super()._build_payload(query, documents, top_k)
        payload["return_documents"] = False
        return payload


class JinaReranker(APIBaseReranker):
    """Jina API 기반 리랭커."""

    def __init__(self, api_key: str | None = None, top_k: int = 5, threshold: float = 0.3):
        super().__init__(
            api_key=api_key or os.getenv("JINA_API_KEY"),
            model_name="jina-reranker-v2-base-multilingual",
            api_url="https://api.jina.ai/v1/rerank",
            top_k=top_k,
            threshold=threshold,
            name="Jina",
        )


class RerankerFactory:
    """환경 변수(RERANKER_TYPE)에 따라 적절한 리랭커를 생성합니다."""

    @staticmethod
    def create(top_k: int = 5) -> BaseReranker:
        reranker_type = os.getenv("RERANKER_TYPE", "local").lower()

        if reranker_type == "cohere":
            logger.info("Using CohereReranker")
            return CohereReranker(top_k=top_k)
        elif reranker_type == "jina":
            logger.info("Using JinaReranker")
            return JinaReranker(top_k=top_k)
        else:
            logger.info("Using CrossEncoderReranker (Local)")
            return CrossEncoderReranker.get_instance()
