"""다양한 문서 리랭커(Reranker) 구현 모듈.

Two-stage Search의 2단계로, Bi-Encoder 검색 결과 상위 N개를
Cross-Encoder 또는 외부 API(Cohere, Jina)로 재정렬하여 관련성 스코어를 재계산합니다.

Failover 메커니즘: API 장애 또는 할당량 초과 시, 1차 검색 결과를 원본 순서대로 반환합니다.
"""

import abc
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import requests
import torch
from langchain_core.documents import Document

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

from src.utils.paths import CROSS_ENCODER_CACHE_DIR

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """리랭킹 결과."""
    documents: list[Document]
    scores: list[float]
    filtered_count: int = 0
    elapsed_time_sec: float = field(default=0.0)


class BaseReranker(abc.ABC):
    """리랭커 공통 인터페이스 (추상 클래스)."""

    def __init__(self, top_k: int = 5, threshold: float = 0.3):
        self.top_k = top_k
        self.threshold = threshold
        self.MAX_INFER_TIME_SEC = 5.0
        self.SAFETY_RATIO = 0.8
        self._lock = threading.Lock()

    @abc.abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> RerankResult:
        """관련성 스코어로 재정렬 및 필터링."""
        pass

    def rerank_with_timeout(
        self,
        query: str,
        documents: list[Document],
        max_k: int | None = None,
    ) -> RerankResult:
        """지연 시간을 고려하여 동적으로 조절하며 리랭킹을 수행합니다.
        API 또는 로컬 모델 추론 실패 시 Failover 역할을 겸합니다.
        """
        effective_max_k = max_k or self.top_k * 2

        if len(documents) > effective_max_k:
            documents = documents[:effective_max_k]

        try:
            result = self.rerank(query, documents)
        except Exception as e:
            logger.error(f"Reranking failed in timeout mode: {e}")
            # Failover: 장애 시 1차 검색 결과 상위 K개 그대로 반환 (점수는 0.5로 고정)
            fallback_k = min(len(documents), self.top_k)
            return RerankResult(
                documents=documents[:fallback_k],
                scores=[0.5] * fallback_k,
                filtered_count=len(documents) - fallback_k,
                elapsed_time_sec=0.0,
            )

        # 실제 추론 시간이 목표 초과 시 로그
        if result.elapsed_time_sec > self.MAX_INFER_TIME_SEC * self.SAFETY_RATIO:
            logger.warning(
                f"Reranking took {result.elapsed_time_sec:.2f}s. Approaching timeout."
            )

        return result


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


class CohereReranker(BaseReranker):
    """Cohere API 기반 리랭커."""

    def __init__(self, api_key: str | None = None, top_k: int = 5, threshold: float = 0.5):
        super().__init__(top_k, threshold)
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.model_name = "rerank-multilingual-v3.0"

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
            logger.warning("COHERE_API_KEY is not set. Failing over.")
            raise ValueError("API Key missing")

        start_time = time.time()
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": [doc.page_content for doc in documents],
            "top_n": effective_top_k
        }

        # 예외 발생 시 rerank_with_timeout에서 잡아서 Failover 처리됨
        response = requests.post(
            "https://api.cohere.com/v1/rerank",
            headers=headers,
            json=payload,
            timeout=self.MAX_INFER_TIME_SEC
        )
        response.raise_for_status()

        result = response.json()
        elapsed_time = time.time() - start_time

        results = result.get("results", [])
        final_docs, final_scores = [], []

        for res in results:
            idx = res["index"]
            score = res["relevance_score"]
            if score >= effective_threshold:
                final_docs.append(documents[idx])
                final_scores.append(score)

        return RerankResult(
            documents=final_docs[:effective_top_k],
            scores=final_scores[:effective_top_k],
            filtered_count=len(documents) - len(final_docs[:effective_top_k]),
            elapsed_time_sec=elapsed_time,
        )


class JinaReranker(BaseReranker):
    """Jina API 기반 리랭커."""

    def __init__(self, api_key: str | None = None, top_k: int = 5, threshold: float = 0.3):
        super().__init__(top_k, threshold)
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        self.model_name = "jina-reranker-v2-base-multilingual"

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
            logger.warning("JINA_API_KEY is not set. Failing over.")
            raise ValueError("API Key missing")

        start_time = time.time()
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": [doc.page_content for doc in documents],
            "top_n": effective_top_k
        }

        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers=headers,
            json=payload,
            timeout=self.MAX_INFER_TIME_SEC
        )
        response.raise_for_status()

        result = response.json()
        elapsed_time = time.time() - start_time

        results = result.get("results", [])
        final_docs, final_scores = [], []

        for res in results:
            idx = res["index"]
            score = res["relevance_score"]
            if score >= effective_threshold:
                final_docs.append(documents[idx])
                final_scores.append(score)

        return RerankResult(
            documents=final_docs[:effective_top_k],
            scores=final_scores[:effective_top_k],
            filtered_count=len(documents) - len(final_docs[:effective_top_k]),
            elapsed_time_sec=elapsed_time,
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
            return CrossEncoderReranker.get_instance(top_k=top_k)
