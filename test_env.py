from sentence_transformers import SentenceTransformer

print("모델 로딩 중...(처음엔 다운로드 시간이 조금 걸립니다)")

model = SentenceTransformer('BAAI/bge-m3')

sentences = ["안녕하세요, 조선대학교 산학프로젝트입니다.", "RAG 챗봇 환경 세팅 완료!"]
embeddings = model.encode(sentences)

print(f"임베딩 완료! 벡터 차원 수: {embeddings.shape[1]}")
print("모든 환경이 완벽하게 준비되었습니다!")