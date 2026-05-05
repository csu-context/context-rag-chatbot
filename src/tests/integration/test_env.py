from sentence_transformers import SentenceTransformer
from src.utils.paths import MODELS_DIR

def test_model_loading():
    """BGE-M3 모델 로딩 및 임베딩 생성 테스트"""
    model = SentenceTransformer("BAAI/bge-m3", cache_folder=str(MODELS_DIR))
    sentences = ["안녕하세요, 조선대학교 산학프로젝트입니다.", "RAG 챗봇 환경 세팅 완료!"]
    embeddings = model.encode(sentences)
    assert embeddings.shape[1] == 1024  # BGE-M3 차원 확인

