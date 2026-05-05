import pytest
import psutil
import os
from sklearn.metrics.pairwise import cosine_similarity
from src.models.embedder import BGEEmbedder

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_embedding_performance():
    """BGE-M3 임베딩 모델 성능 및 정확도 테스트"""
    mem_before = get_memory_usage()
    embedder = BGEEmbedder()
    mem_after = get_memory_usage()
    
    # 1. 벡터 차원 확인
    assert embedder.get_dimension() == 1024
    
    # 2. 임베딩 생성 및 유사도 검증
    s1, s2, s3 = "사내 보안 규정은?", "회사 보안 가이드라인은?", "오늘 점심 메뉴는?"
    vecs = embedder.encode([s1, s2, s3])
    
    sim_similar = cosine_similarity([vecs[0]], [vecs[1]])[0][0]
    sim_different = cosine_similarity([vecs[0]], [vecs[2]])[0][0]
    
    # 유사한 문장 간의 유사도가 다른 문장보다 높아야 함
    assert sim_similar > sim_different
    assert sim_similar > 0.7  # 한국어 BGE-M3 기준

