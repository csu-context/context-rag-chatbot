import logging

import torch
from sentence_transformers import SentenceTransformer

from src.utils.paths import MODELS_DIR

logger = logging.getLogger(__name__)

class BGEEmbedder:
    def __init__(self, model_name="BAAI/bge-m3"):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.model = SentenceTransformer(model_name, cache_folder=str(MODELS_DIR))
        self.model.to(self.device)
        logger.info(f"Model loaded on: {self.device}")

    def encode(self, sentences):
        return self.model.encode(sentences, normalize_embeddings=True)

    def get_dimension(self):
        return self.model.get_sentence_embedding_dimension()
