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

from src.common.config import settings
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
    model_name: str = "unknown"
    filtered_count: int = 0
    elapsed_time_sec: float = 0.0


class BaseReranker(ABC):
    """리랭커 기본 클래스"""

    MAX_INFER_TIME_SEC = 5  # API 타임아웃

    def __init__(self, name: str, top_k: int = 5, threshold: float | None = None):
        self.name = name
        self.top_k = top_k
        self.threshold = threshold if threshold is not None else settings.RERANKER_THRESHOLD

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

    def rerank_with_timeout(self, query: str, documents: list[Document], **kwargs: Any) -> RerankResult:
        target_top_k = kwargs.get("top_k") or self.top_k
        kwargs["top_k"] = target_top_k

        start_time = time.time()
        try:
            return self.rerank(query, documents, **kwargs)
        except Exception as e:
            elapsed = time.time() - start_time
            logger.warning(f"Reranking failed in {self.name}: {e}. Returning original documents.")
            # 실패 시 원본 문서에서 top_k만큼 잘라서 반환
            return RerankResult(
                documents=documents[:target_top_k],
                scores=[0.0] * min(len(documents), target_top_k),
                model_name=self.name,
                filtered_count=max(0, len(documents) - target_top_k),
                elapsed_time_sec=elapsed,
            )


class CrossEncoderReranker(BaseReranker):
    """로컬 모델 (BGE-Reranker, MS-MARCO 등) 기반 리랭커 싱글톤."""

    _instance: Optional["CrossEncoderReranker"] = None
    _model: CrossEncoder | None = None
    _singleton_lock = threading.Lock()

    def __init__(
        self,
        model_name: str | None = None,
        top_k: int = 5,
        threshold: float | None = None,
        device: str | None = None,
    ):
        super().__init__(name="Local CrossEncoder", top_k=top_k, threshold=threshold)
        self.model_name = model_name or settings.RERANKER_MODEL_NAME
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def get_instance(
        cls,
        model_name: str | None = None,
        top_k: int = 5,
        threshold: float | None = None,
        device: str | None = None,
    ) -> "CrossEncoderReranker":
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = cls(model_name, top_k, threshold, device)
        else:
            # 기존 인스턴스가 존재하면 top_k와 threshold만 업데이트하여 재사용
            cls._instance.top_k = top_k
            if threshold is not None:
                cls._instance.threshold = threshold
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
                import warnings

                automodel_args = (
                    {"torch_dtype": torch.float16}
                    if settings.RERANKER_USE_FP16 and self.device in ("cuda", "mps")
                    else {}
                )
                # Issue 40: sentence-transformers/transformers 버전업 시 Deprecation Warning 억제
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning, module="sentence_transformers")
                    warnings.filterwarnings("ignore", category=DeprecationWarning, module="transformers")
                    self._model = CrossEncoder(
                        self.model_name,
                        device=self.device,
                        cache_dir=str(CROSS_ENCODER_CACHE_DIR),
                        automodel_args=automodel_args,
                    )
            except Exception as e:
                logger.error(f"Failed to load CrossEncoder: {e}")
                raise
        return self._model

    @staticmethod
    def _to_list(scores_pred) -> list:
        return scores_pred.tolist() if hasattr(scores_pred, "tolist") else list(scores_pred)

    def _predict_with_cpu_fallback(self, pairs: list) -> list:
        try:
            model = self._load_model()
            try:
                scores_pred = model.predict(pairs, batch_size=settings.RERANKER_BATCH_SIZE)
            finally:
                if self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return self._to_list(scores_pred)
        except RuntimeError as e:
            err_msg = str(e).lower()
            if self.device != "cpu" and any(x in err_msg for x in ["cuda", "mps", "device", "out of memory", "oom"]):
                logger.warning(f"[{self.name}] GPU/MPS error detected: {e}. Falling back to CPU mode...")
                self.device = "cpu"
                with self._singleton_lock:
                    self._model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                try:
                    model = self._load_model()
                    scores_pred = model.predict(pairs, batch_size=settings.RERANKER_BATCH_SIZE)
                    return self._to_list(scores_pred)
                except Exception as cpu_err:
                    logger.error(f"[{self.name}] Failed to run even on CPU fallback: {cpu_err}")
                    raise
            else:
                raise

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> RerankResult:
        # RAG 응답 시간 5초 이내 사수를 위해 최종 제공 문서를 엄격하게 최대 RERANKER_MAX_DOCS개로 하드캡(Hard-cap) 제한
        effective_top_k = min(settings.RERANKER_MAX_DOCS, top_k or self.top_k)
        effective_threshold = threshold if threshold is not None else self.threshold

        if not documents:
            return RerankResult(documents=[], scores=[], model_name=self.name, elapsed_time_sec=0.0)

        pairs = [(query, doc.page_content) for doc in documents]

        start_time = time.time()
        raw_scores = self._predict_with_cpu_fallback(pairs)

        elapsed_time = time.time() - start_time
        # ms-marco 등 raw logit을 [0, 1] 확률값으로 직관적으로 정규화 (temperature 스케일링 제거)
        scores = torch.sigmoid(torch.tensor(raw_scores)).tolist()

        scored_docs = sorted(zip(scores, documents, strict=True), key=lambda x: x[0], reverse=True)
        filtered = [(s, d) for s, d in scored_docs if s >= effective_threshold]
        ranked = filtered[:effective_top_k]

        return RerankResult(
            documents=[d for _, d in ranked],
            scores=[s for s, _ in ranked],
            model_name=self.name,
            filtered_count=len(documents) - len(ranked),
            elapsed_time_sec=elapsed_time,
        )


class APIBaseReranker(BaseReranker):
    """API 기반 리랭커 공통 로직 추상화 클래스 (Session 재사용)"""

    def __init__(self, api_key: str | None, model_name: str, api_url: str, top_k: int, threshold: float, name: str):
        super().__init__(name=name, top_k=top_k, threshold=threshold)
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = api_url
        self._session = requests.Session()

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
        return float(result_item.get("relevance_score", 0.0))

    @staticmethod
    def _extract_index(result_item: dict) -> int:
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
            return RerankResult(documents=[], scores=[], model_name=self.name)

        if not self.api_key:
            logger.warning(f"[{self.name}] API_KEY is not set. Failing over.")
            raise ValueError(f"API Key missing for {self.name}")

        start_time = time.time()
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = self._build_payload(query, documents, effective_top_k)

        try:
            response = self._session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.MAX_INFER_TIME_SEC,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"[{self.name}] API request failed: {e}")
            raise

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
            model_name=self.name,
            filtered_count=len(documents) - len(final_docs[:effective_top_k]),
            elapsed_time_sec=elapsed_time,
        )


class CohereReranker(APIBaseReranker):
    """Cohere API 기반 리랭커."""

    def __init__(self, api_key: str | None = None, top_k: int = 5, threshold: float | None = None):
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

    def __init__(self, api_key: str | None = None, top_k: int = 5, threshold: float | None = None):
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
        reranker_type = settings.RERANKER_TYPE.lower()
        allow_external = settings.ALLOW_EXTERNAL_RERANKER and settings.ALLOW_EXTERNAL_API

        if reranker_type in ["cohere", "jina"]:
            if not allow_external:
                logger.warning(
                    f"보안 정책: 외부 리랭커 '{reranker_type}' 사용이 차단되었습니다. "
                    "ALLOW_EXTERNAL_API 또는 ALLOW_EXTERNAL_RERANKER 설정을 확인하십시오. 로컬 모델로 전환합니다."
                )
                reranker_type = "local"
            else:
                logger.warning(
                    f"보안 경고: 외부 리랭커 '{reranker_type}' 사용이 허용되어 있습니다 "
                    "(ALLOW_EXTERNAL_API=True 및 ALLOW_EXTERNAL_RERANKER=True)."
                )

        _registry: dict[str, type] = {"cohere": CohereReranker, "jina": JinaReranker}
        if reranker_type in _registry:
            cls = _registry[reranker_type]
            logger.info(f"{cls.__name__} 리랭커를 사용합니다.")
            return cls(top_k=top_k)

        logger.info(f"로컬 CrossEncoder 리랭커를 사용합니다. (모델: {settings.RERANKER_MODEL_NAME})")
        return CrossEncoderReranker.get_instance(model_name=settings.RERANKER_MODEL_NAME, top_k=top_k)
