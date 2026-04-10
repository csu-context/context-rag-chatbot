import os
import sys
import warnings
# 경고 숨기기 로직 추가
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", module="huggingface_hub")

from typing import List, Dict, Any, Optional
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가하여 src 모듈 임포트 에러 방지
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# DB 영구 저장 경로 설정 (/.gitignore에 등록된 /vector_db/chroma/ 경로 사용)
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db" / "chroma"

import chromadb
from chromadb.config import Settings
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from src.models.embedder import BGEEmbedder

class BGEChromaEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB 커스텀 임베딩 함수
    src.models.embedder의 BGEEmbedder를 사용하여 문서를 벡터화합니다.
    """
    def __init__(self, model_name: str = 'BAAI/bge-m3'):
        self.embedder = BGEEmbedder(model_name=model_name)

    def __call__(self, input: Documents) -> Embeddings:
        # BGEEmbedder.encode는 내부적으로 torch tensor 또는 numpy array를 반환할 수 있으므로
        # ChromaDB 호환성을 위해 파이썬 내장 기본 리스트 형식(float)으로 변환하여 반환합니다.
        embeddings = self.embedder.encode(input)
        return embeddings.tolist()


class ChromaDBManager:
    def __init__(self, collection_name: str = "rag_collection"):
        """
        ChromaDB 클라이언트 및 컬렉션을 초기화합니다.
        """
        # 1. 영구 저장(Persistence) 로컬 클라이언트 설정
        os.makedirs(VECTOR_DB_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR),
        settings=Settings(anonymized_telemetry=False)
        )
        
        # 2. 커스텀 BGE 임베딩 함수 초기화
        self.embedding_fn = BGEChromaEmbeddingFunction()
        
        # 3. 컬렉션 가져오기 또는 생성
        # BGE-M3 모델은 주로 코사인 유사도(cosine similarity) 검색에 최적화되어 있습니다.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

        print(f"✅ ChromaDB 로드 완료 (컬렉션: {collection_name}, 데이터 개수: {self.collection.count()})")

    def upsert_documents(self, ids: List[str], documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        """
        문서 청크를 DB에 업서트(Upsert)합니다.
        기존에 동일한 ID가 존재하면 업데이트(Update)를 수행하여 중복 저장을 방지합니다.
        """
        if not ids or not documents:
            print("⚠️ 업서트할 문서가 없습니다.")
            return

        # ChromaDB 메타데이터에는 None 값이 들어갈 수 없으므로(validation 에러 발생),
        # 딕셔너리에서 None 값을 제거하는 전처리 로직 추가 (chunking_strategy.py의 Optional 값 대응)
        cleaned_metadatas = None
        if metadatas:
            cleaned_metadatas = []
            for meta in metadatas:
                cleaned = {k: v for k, v in meta.items() if v is not None}
                
                # ChromaDB는 빈 딕셔너리({}) 삽입을 허용하지 않으므로 더미 키 추가
                if not cleaned:
                    cleaned = {"_is_empty": True}
                    
                cleaned_metadatas.append(cleaned)

        # Upsert 실행
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=cleaned_metadatas
        )
        print(f"✅ {len(ids)}개의 문서 청크가 ChromaDB에 성공적으로 업서트되었습니다.")

    def query(self, query_texts: List[str], n_results: int = 3) -> dict:
        """
        주어진 쿼리 텍스트와 가장 유사한 문서를 검색합니다.
        """
        results = self.collection.query(
            query_texts=query_texts,
            n_results=n_results
        )
        return results

    def get_count(self) -> int:
        """현재 컬렉션에 저장된 총 청크 수를 반환합니다."""
        return self.collection.count()


if __name__ == "__main__":
    # ==========================================
    # 검증 계획 1 & 2 테스트용 스크립트
    # ==========================================
    print("--- 1. ChromaDB Manager 초기화 ---")
    db_manager = ChromaDBManager(collection_name="test_collection")
    initial_count = db_manager.get_count()
    
    print("\n--- 2. 샘플 데이터 준비 및 Upsert ---")
    # 5개의 샘플 텍스트와 메타데이터 준비
    sample_ids = ["chunk_001", "chunk_002", "chunk_003", "chunk_004", "chunk_005"]
    sample_texts = [
        "회사의 정규 출근 시간은 오전 9시이며, 퇴근 시간은 오후 6시입니다.",
        "연차 휴가는 입사 후 1년이 지나면 15일이 발생합니다.",
        "식대는 매월 급여와 함께 20만원씩 비과세로 지급됩니다.",
        "사내 복지 포인트는 매년 1월 1일에 100만 포인트가 지급되며, 복지몰에서 사용 가능합니다.",
        "신규 입사자는 첫 3개월 동안 수습 기간을 거치며, 수습 기간에도 급여의 100%가 지급됩니다."
    ]
    sample_metadatas = [
        {"doc_id": "doc_01", "sec_title": "출퇴근 규정"},
        {"doc_id": "doc_01", "sec_title": "휴가 규정"},
        {"doc_id": "doc_02", "sec_title": "급여 및 복지"},
        {"doc_id": "doc_02", "sec_title": "급여 및 복지"},
        {"doc_id": "doc_03", "sec_title": "채용 및 수습"}
    ]
    
    # 첫 번째 업서트 (최초 삽입)
    db_manager.upsert_documents(ids=sample_ids, documents=sample_texts, metadatas=sample_metadatas)
    
    # 두 번째 업서트 (동일 ID로 덮어쓰기 - 중복 데이터 방지 검증)
    print("\n--- 3. 중복 저장 방지(Upsert) 검증 ---")
    db_manager.upsert_documents(ids=sample_ids, documents=sample_texts, metadatas=sample_metadatas)
    final_count = db_manager.get_count()
    
    print(f"-> 초기 데이터 개수: {initial_count}")
    print(f"-> 현재 데이터 개수: {final_count} (동일 ID 재업로드 시 데이터가 중복 증가하지 않음 확인)")
    
    print("\n--- 4. 유사도 검색 테스트 ---")
    queries = ["수습 기간 동안 월급은 어떻게 되나요?", "1년 다니면 휴가 며칠 나와요?"]
    search_results = db_manager.query(query_texts=queries, n_results=2)
    
    for i, query_str in enumerate(queries):
        print(f"\n[질문]: {query_str}")
        # 반환되는 데이터 구조가 List[List[str]] 형태이므로 인덱싱 접근
        for j in range(len(search_results['documents'][i])):
            doc_text = search_results['documents'][i][j]
            doc_meta = search_results['metadatas'][i][j]
            doc_dist = search_results['distances'][i][j]
            # hnsw:space 가 cosine일 경우, distance 값이 작을수록 (0에 가까울수록) 유사도가 높습니다.
            print(f"  - 결과 {j+1} (거리: {doc_dist:.4f}): {doc_text} (출처: {doc_meta['sec_title']})")