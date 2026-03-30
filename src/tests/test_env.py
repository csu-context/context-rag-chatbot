from sentence_transformers import SentenceTransformer
from src.utils.paths import MODELS_DIR, ensure_directories

# 1. 프로젝트에 필요한 디렉토리 자동 생성
ensure_directories()

print("모델 로딩 중...(처음엔 다운로드 시간이 조금 걸립니다)")

# 2. 모델을 특정 경로(models/)에 명시적으로 저장하거나 참조하도록 설정
# cache_folder를 지정하면 모델 데이터가 프로젝트 내 models/ 디렉토리에 관리됩니다.
model = SentenceTransformer('BAAI/bge-m3', cache_folder=str(MODELS_DIR))

sentences = ["안녕하세요, 조선대학교 산학프로젝트입니다.", "RAG 챗봇 환경 세팅 완료!"]
embeddings = model.encode(sentences)

print(f"임베딩 완료! 벡터 차원 수: {embeddings.shape[1]}")
print("모든 환경이 완벽하게 준비되었습니다!")