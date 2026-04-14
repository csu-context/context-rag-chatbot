# src/models/embedder.py
import torch
from sentence_transformers import SentenceTransformer

from src.utils.paths import MODELS_DIR


class BGEEmbedder:
    def __init__(self, model_name="BAAI/bge-m3"):
        # 1. 장치 우선순위 결정: CUDA -> MPS -> CPU
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # 2. 모델 로드 및 장치 할당
        self.model = SentenceTransformer(model_name, cache_folder=str(MODELS_DIR))
        self.model.to(self.device)
        print(f"Model loaded on: {self.device}")

    def encode(self, sentences):
        # 문장을 벡터로 변환 (Dense Embedding)
        return self.model.encode(sentences, normalize_embeddings=True)

    def get_dimension(self):
        # 벡터 차원 확인
        return self.model.get_sentence_embedding_dimension()
