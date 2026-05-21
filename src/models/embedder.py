# src/models/embedder.py
import logging

import torch
from sentence_transformers import SentenceTransformer

from src.common.config import settings
from src.utils.paths import MODELS_DIR

logger = logging.getLogger(__name__)


class BGEEmbedder:
    def __init__(self, model_name="BAAI/bge-m3"):
        # 1. 장치 우선순위 결정: CUDA -> MPS -> CPU
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # 보안 설정에 따른 스위칭 및 경고 로깅
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

        # 2. 모델 로드 및 장치 할당
        self.model = SentenceTransformer(
            model_name, cache_folder=str(MODELS_DIR), local_files_only=not settings.ALLOW_EXTERNAL_API
        )
        self.model.to(self.device)
        print(f"모델이 다음 장치에 로드되었습니다: {self.device}")

    def encode(self, sentences):
        # 문장을 벡터로 변환 (Dense Embedding)
        return self.model.encode(sentences, normalize_embeddings=True)

    def get_dimension(self):
        # 벡터 차원 확인
        return self.model.get_sentence_embedding_dimension()
