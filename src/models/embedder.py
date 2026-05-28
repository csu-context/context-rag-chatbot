import logging
import threading
from typing import Optional

import torch
from sentence_transformers import SentenceTransformer

from src.common.config import settings
from src.utils.paths import MODELS_DIR

logger = logging.getLogger(__name__)


class BGEEmbedder:
    _instance: Optional["BGEEmbedder"] = None
    _singleton_lock = threading.Lock()

    @classmethod
    def get_instance(cls, model_name="BAAI/bge-m3") -> "BGEEmbedder":
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = cls(model_name=model_name)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """테스트용 싱글톤 리셋"""
        with cls._singleton_lock:
            cls._instance = None

    def __init__(self, model_name="BAAI/bge-m3"):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        if not settings.ALLOW_EXTERNAL_API:
            if model_name != "BAAI/bge-m3":
                logger.warning(
                    f"보안 정책: 외부 API 및 다운로드가 비허용되었습니다. "
                    f"요청된 임베딩 모델 '{model_name}' 대신 로컬 기본 모델 'BAAI/bge-m3'로 전환합니다."
                )
                model_name = "BAAI/bge-m3"
        else:
            logger.warning(
                "보안 경고: 외부 API 호출 및 허깅페이스 다운로드가 허용되어 있습니다 (ALLOW_EXTERNAL_API=True)."
            )

        model_kwargs = (
            {"torch_dtype": torch.float16} if settings.EMBEDDER_USE_FP16 and self.device in ("cuda", "mps") else {}
        )
        self.model = SentenceTransformer(
            model_name,
            cache_folder=str(MODELS_DIR),
            local_files_only=not settings.ALLOW_EXTERNAL_API,
            model_kwargs=model_kwargs,
        )
        self.model.to(self.device)
        logger.info(f"모델이 다음 장치에 로드되었습니다: {self.device}")

    def encode(self, sentences):
        return self.model.encode(sentences, normalize_embeddings=True)

    def get_dimension(self):
        return self.model.get_sentence_embedding_dimension()
