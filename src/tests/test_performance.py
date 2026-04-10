# src/tests/test_performance.py 업데이트 버전
import os
import time

import psutil
import torch
from sklearn.metrics.pairwise import cosine_similarity

from src.models.embedder import BGEEmbedder


def get_memory_usage():
    # 현재 프로세스의 시스템 RAM 사용량 (MB 단위)
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024


def test_embedding_performance():
    # 1. 모델 로드 전 메모리 측정
    mem_before = get_memory_usage()

    embedder = BGEEmbedder()

    # 2. 모델 로드 후 메모리 및 차원 확인
    mem_after = get_memory_usage()
    dimension = embedder.get_dimension()

    print(f"\n[1] 벡터 차원(Dimension): {dimension}")
    print(f"[2] 시스템 RAM 점유율 증가: {mem_after - mem_before:.2f} MB")

    if torch.cuda.is_available():
        # GPU VRAM 점유율 측정
        vram_used = torch.cuda.memory_allocated() / 1024 / 1024
        vram_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        print(f"[3] GPU VRAM 사용량: {vram_used:.2f} MB (할당: {vram_reserved:.2f} MB)")

    # 3. 속도 및 유사도 테스트 (기존 로직)
    start_time = time.time()
    sentences = ["신규 입사자 가이드 위치 알려줘."] * 10
    embedder.encode(sentences)
    end_time = time.time()

    print(f"[4] 10개 문장 임베딩 소요 시간: {end_time - start_time:.4f}초")

    # 유사도 테스트
    s1, s2, s3 = "사내 보안 규정은?", "회사 보안 가이드라인은?", "오늘 점심 메뉴는?"
    vecs = embedder.encode([s1, s2, s3])
    sim_similar = cosine_similarity([vecs[0]], [vecs[1]])[0][0]
    print(f"[5] 유사 문장 간 유사도: {sim_similar:.4f}")


if __name__ == "__main__":
    test_embedding_performance()
