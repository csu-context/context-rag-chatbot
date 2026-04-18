"""Cross-Encoder 기반 문서 리랭킹 모듈.

Two-stage Search의 2단계로, Bi-Encoder(Vector DB) 검색 결과 상위 N개를
Cross-Encoder로 재정렬하여 관련성 스코어를 재계산하고 임계치 이하 문서를 필터링합니다.

Thread-safe 싱글톤 패턴을 사용하여 동시 접근 시 Race Condition을 방지합니다.

모델 선택 가이드:
    - 영어/다국어: "cross-encoder/ms-marco-MiniLM-L-6-v2" (기본값)
    - 한국어 최적화: "skesarmom/cross-encoder-kor" 또는 "bge-reranker-base"
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """리랭킹 결과."""

    documents: list[Document]
    scores: list[float]
    filtered_count: int = 0
    elapsed_time_sec: float = field(default=0.0)


class CrossEncoderReranker:
    """Cross-Encoder 기반 리랭커 싱글톤.

    로컬 환경에서 Cross-Encoder 모델을 효율적으로 로드하고,
    쿼리와 문서 리스트를 입력받아 관련성 스코어를 재산출합니다.

    Thread-safe 이중 체크 락 패턴을 사용하여 동시 접근 시 Race Condition을 방지합니다.
    """

    _instance: Optional["CrossEncoderReranker"] = None
    _model: CrossEncoder | None = None
    _lock = threading.Lock()

    # --- 기본 설정값 ---
    DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    DEFAULT_TOP_K = 5
    MAX_INFER_TIME_SEC = 5.0
    SAFETY_RATIO = 0.8

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        top_k: int = DEFAULT_TOP_K,
        threshold: float | None = None,
        device: str | None = None,
    ):
        """리랭커 초기화.

        Args:
            model_name: 사용할 Cross-Encoder 모델 이름 (HuggingFace 모델 경로)
                        - 영어/다국어: "cross-encoder/ms-marco-MiniLM-L-6-v2"
                        - 한국어 최적화: "skesarmom/cross-encoder-kor"
            top_k: 리랭킹 후 유지할 상위 문서 수
            threshold: 임계치 이하 문서를 필터링하는 기준 점수 (None이면 모델별 자동 설정)
            device: inference 장치 ('cuda', 'cpu') - None이면 자동 선택
        """
        self.model_name = model_name
        self.top_k = top_k
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # --- 모델별 기본 임계치 자동 설정 (명시적 지정 시 무시) ---
        if threshold is not None:
            self.threshold = threshold
            logger.info(f"Threshold explicitly set to {threshold}")
        else:
            self._set_model_defaults()

    def _set_model_defaults(self) -> None:
        """모델 이름에 따라 기본 임계치를 동적으로 조정."""
        model_lower = self.model_name.lower()

        if "bge" in model_lower:
            # BGE Reranker는 점수 범위가 0~1 (sigmoid 활성화)
            self.threshold = 0.2
            logger.info("BGE reranker detected - threshold auto-set to 0.2")
        elif "skesarmom" in model_lower or "kor" in model_lower:
            # 한국어 특화 모델은 상대적으로 높은 임계치 권장
            self.threshold = 0.4
            logger.info("Korean-specific reranker detected - threshold auto-set to 0.4")
        else:
            # 기본 (MS MARCO MiniLM) - tanh 활성화로 -1~1 범위, 0.3이 적정
            self.threshold = 0.3
            logger.info("Default reranker (MS MARCO MiniLM) detected - threshold auto-set to 0.3")

    @classmethod
    def get_instance(
        cls,
        model_name: str = DEFAULT_MODEL_NAME,
        top_k: int = DEFAULT_TOP_K,
        threshold: float | None = None,
        device: str | None = None,
    ) -> "CrossEncoderReranker":
        """싱글톤 인스턴스 반환. 최초 호출 시 모델도 로드.

        Thread-safe 이중 체크 락 패턴을 사용합니다.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        model_name=model_name,
                        top_k=top_k,
                        threshold=threshold,
                        device=device,
                    )
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """테스트용 싱글톤 리셋."""
        with cls._lock:
            cls._instance = None
            cls._model = None

    def _load_model(self) -> CrossEncoder:
        """Cross-Encoder 모델을 로드하거나 캐시된 인스턴스 반환.

        Thread-safe하게 모델을 초기화합니다.
        """
        # 첫 번째 체크 (락 없이 - 성능 최적화)
        if self._model is not None:
            return self._model

        with CrossEncoderReranker._lock:  # 전역 락 사용
            # 두 번째 체크 (락 내에서 - Race Condition 방지)
            if self._model is not None:
                return self._model

            logger.info(f"Loading CrossEncoder model: {self.model_name} on {self.device}")
            try:
                self._model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                    cache_dir="./.cache/cross-encoder",
                )
                logger.info("CrossEncoder model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load CrossEncoder model: {e}")
                raise

        return self._model

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> RerankResult:
        """쿼리와 문서 리스트를 입력받아 관련성 스코어로 재정렬 및 필터링.

        Args:
            query: 사용자 질문 (쿼리)
            documents: Bi-Encoder 검색 결과 문서 리스트
            top_k: 유지할 상위 문서 수 (None이면 초기화 값 사용)
            threshold: 임계치 (None이면 초기화 값 사용)

        Returns:
            RerankResult: 재정렬 및 필터링된 문서와 점수 정보

        Note:
            - 문서 수가 2개 미만이면 리랭킹 생략 후 원본 반환
            - 임계치 이하 문서는 과감히 제거 (환각 방지)
            - 필터링 후 결과가 비어있을 경우, 최고 점수 문서 1개를 유지
        """
        effective_top_k = top_k or self.top_k
        effective_threshold = threshold if threshold is not None else self.threshold

        # --- 검색 결과가 너무 적으면 리랭킹 생략 ---
        if len(documents) < 2:
            logger.debug(f"Search results too few ({len(documents)} docs). Skipping reranking.")
            return RerankResult(
                documents=documents,
                scores=[0.5] * len(documents),
                filtered_count=0,
                elapsed_time_sec=0.0,
            )

        # --- Cross-Encoder 점수 계산 (시간 측정 포함) ---
        try:
            pairs = [(query, doc.page_content) for doc in documents]
            model = self._load_model()

            start_time = time.time()
            scores = model.predict(pairs).tolist()
            elapsed_time = time.time() - start_time

            if elapsed_time > self.MAX_INFER_TIME_SEC * self.SAFETY_RATIO:
                logger.warning(
                    f"Reranking took {elapsed_time:.2f}s (target: <{self.MAX_INFER_TIME_SEC}s). "
                    f"Consider reducing top_k or using a faster model."
                )

        except Exception as e:
            logger.warning(f"Reranking failed (keeping original order): {e}")
            return RerankResult(
                documents=documents,
                scores=[0.5] * len(documents),
                filtered_count=0,
                elapsed_time_sec=0.0,
            )

        # --- 점수 기반 정렬 (내림차순) ---
        scored_docs = sorted(
            zip(scores, documents, strict=True),
            key=lambda x: x[0],
            reverse=True,
        )

        # --- 임계치 필터링 ---
        filtered = [(score, doc) for score, doc in scored_docs if score >= effective_threshold]

        # --- 최소 1개 이상 보장 로직 ---
        ranked = [scored_docs[0]] if not filtered and scored_docs else filtered[:effective_top_k]

        final_scores = [score for score, _ in ranked]
        final_docs = [doc for _, doc in ranked]

        return RerankResult(
            documents=final_docs,
            scores=final_scores,
            filtered_count=len(documents) - len(final_docs),
            elapsed_time_sec=elapsed_time,
        )

    def rerank_with_timeout(
        self,
        query: str,
        documents: list[Document],
        max_k: int | None = None,
    ) -> RerankResult:
        """지연 시간 고려하여 동적으로 top_k를 조절하는 리랭킹.

        문서 수가 많거나 추론 시간이 길 경우, 동적으로 유지할 문서 수를 줄여
        목표 응답 시간(MAX_INFER_TIME_SEC) 내 완수를 시도합니다.

        Args:
            query: 사용자 질문
            documents: 검색 결과 문서 리스트
            max_k: 최대 유지 문서 수 (기본값: top_k * 2)

        Returns:
            RerankResult: 리랭킹 결과
        """
        effective_max_k = max_k or self.top_k * 2

        # --- 1단계: 문서 수가 많으면 먼저 상위 K개만 추출 ---
        if len(documents) > effective_max_k:
            documents = documents[:effective_max_k]

        # --- 2단계: 리랭킹 실행 및 시간 측정 (예외 처리 포함) ---
        try:
            result = self.rerank(query, documents)
        except Exception as e:
            logger.warning(f"Reranking failed in timeout mode: {e}")
            return RerankResult(
                documents=documents if documents else [],
                scores=[0.5] * len(documents) if documents else [],
                filtered_count=0,
                elapsed_time_sec=0.0,
            )

        # --- 3단계: 실제 추론 시간이 목표 초과 시 동적 조절 (스레드 안전) ---
        if result.elapsed_time_sec > self.MAX_INFER_TIME_SEC * self.SAFETY_RATIO:
            reduced_k = max(1, self.top_k // 2)
            with self._lock:
                # 인스턴스 설정을 스레드 안전하게 업데이트 (다음 호출부터 적용)
                self.top_k = reduced_k
                logger.warning(
                    f"Reranking took {result.elapsed_time_sec:.2f}s. Reduced top_k to {reduced_k} for subsequent calls."
                )

        return result
