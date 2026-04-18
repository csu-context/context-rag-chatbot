import logging
import os
import time
import warnings
from typing import Any

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings

from src.models.embedder import BGEEmbedder
from src.utils.paths import VECTOR_DB_DIR, ensure_directories

# 경고 숨기기 로직 추가
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", module="huggingface_hub")

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BGEChromaEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB 커스텀 임베딩 함수
    src.models.embedder의 BGEEmbedder를 사용하여 문서를 벡터화합니다.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.embedder = BGEEmbedder(model_name=model_name)

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.embedder.encode(input)
        return embeddings.tolist()


class ChromaDBManager:
    def __init__(self, collection_name: str = "rag_collection"):
        """
        ChromaDB 클라이언트 및 컬렉션을 초기화합니다.
        환경 변수 CHROMA_SERVER_HOST 존재 여부에 따라 로컬(Persistent) 또는 서버(Http) 모드로 동작하며,
        DB 연결 실패 시 재시도(Retry) 로직을 수행합니다.
        """
      self.collection_name = collection_name
        self.embedding_fn = BGEChromaEmbeddingFunction()

        # 클라이언트 초기화 및 DB 연결 재시도 로직
        self._initialize_client_with_retry()

    def _initialize_client_with_retry(self, max_retries: int = 3, retry_delay: int = 2):
        chroma_host = os.getenv("CHROMA_SERVER_HOST")
        chroma_port = os.getenv("CHROMA_SERVER_PORT", "8000")

        for attempt in range(max_retries):
            try:
                if chroma_host:
                    logger.info(f"ChromaDB 서버 모드 접속 시도 (Host: {chroma_host}, Port: {chroma_port})")
                    self.client = chromadb.HttpClient(
                        host=chroma_host, port=int(chroma_port), settings=Settings(anonymized_telemetry=False)
                    )
                else:
                    ensure_directories()
                    logger.info(f"ChromaDB 로컬 모드 활성화 (Path: {VECTOR_DB_DIR})")
                    self.client = chromadb.PersistentClient(
                        path=str(VECTOR_DB_DIR), settings=Settings(anonymized_telemetry=False)
                    )

                # 컬렉션 로드 (실질적인 연결 테스트 구간)
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_fn,
                    metadata={"hnsw:space": "cosine"},
                )

                logger.info(
                    f"✅ ChromaDB 로드 완료 (컬렉션: {self.collection_name}, 데이터 개수: {self.collection.count()})"
                )
                return  # 성공 시 루프 탈출

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"⚠️ ChromaDB 연결 실패. {retry_delay}초 후 재시도... ({attempt + 1}/{max_retries}) | 오류: {e}"
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error("❌ ChromaDB 연결에 최종 실패했습니다. DB 상태를 확인해주세요.")
                    # 재시도 최종 실패 시 빈 컬렉션 객체 방지 처리가 필요할 수 있으나, 여기서는 에러를 발생시킵니다.
                    raise RuntimeError("ChromaDB initialization failed.") from e

    def embed_query(self, query_text: str) -> list[float]:
        """
        사용자 쿼리를 벡터(Embedding)로 변환합니다.
        """
        try:
            return self.embedding_fn([query_text])[0]
        except Exception as e:
            logger.error(f"쿼리 임베딩 중 오류 발생: {e}")
            return []

    def upsert_documents(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ):
        """
        문서 청크를 DB에 업서트(Upsert)합니다.
        기존에 동일한 ID가 존재하면 업데이트(Update)를 수행하여 중복 저장을 방지합니다.
        """
        if not ids or not documents:
            logger.warning("업서트할 문서가 없습니다.")
            return

        cleaned_metadatas = None
        if metadatas:
            cleaned_metadatas = []
            for meta in metadatas:
                cleaned = {k: v for k, v in meta.items() if v is not None}
                if not cleaned:
                    cleaned = {"_is_empty": True}
                cleaned_metadatas.append(cleaned)

        try:
            self.collection.upsert(ids=ids, documents=documents, metadatas=cleaned_metadatas)
            logger.info(f"{len(ids)}개의 문서 청크가 ChromaDB에 성공적으로 업서트되었습니다.")
        except Exception as e:
            logger.error(f"문서 업서트 중 오류 발생: {e}")

    def search(self, query_text: str, k: int = 3) -> list[dict[str, Any]]:
        """
        주어진 쿼리 텍스트와 가장 유사한 문서를 검색하고,
        하이브리드 리트리버와 호환되는 표준화된 dict 리스트 형태로 반환합니다.
        """
        try:
            # ChromaDB query 호출
            results = self.collection.query(query_texts=[query_text], n_results=k)

            # 검색 결과가 없는 경우 빈 리스트 반환 (에러 전파 방지)
            if not results or not results.get("documents") or len(results["documents"][0]) == 0:
                logger.info("검색 결과가 없습니다.")
                return []

            standardized_results = []
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]

            for doc, meta, dist in zip(docs, metas, dists, strict=False):
                # 정규화 (Normalization): 거리(Distance)를 0~1 사이의 유사도 점수(Score)로 변환
                # ChromaDB의 코사인 거리 = 1 - 코사인 유사도
                # 음수 유사도(거리가 1을 초과하는 경우)는 RAG 환경에서 의미가 없으므로 0으로 보정
                score = max(0.0, 1.0 - dist)

                standardized_results.append(
                    {
                        "content": doc,
                        "metadata": meta,
                        "score": round(score, 4),  # 소수점 4자리까지 반올림
                    }
                )

            return standardized_results

        except Exception as e:
            logger.error(f"DB 검색 중 오류 발생: {e}")
            return []  # 에러 발생 시에도 빈 리스트를 반환하여 프로세스 중단 방지

    def get_count(self) -> int:
        """현재 컬렉션에 저장된 총 청크 수를 반환합니다."""
        try:
            return self.collection.count()
        except Exception:
            return 0


if __name__ == "__main__":
    # ==========================================
    # 아키텍처 가이드 반영 검증 테스트
    # ==========================================
    logger.info("--- 1. ChromaDB Manager 초기화 (Retry 로직 포함) ---")
    db_manager = ChromaDBManager(collection_name="test_collection")
    initial_count = db_manager.get_count()

    logger.info("--- 2. 샘플 데이터 준비 및 Upsert ---")
    sample_ids = ["chunk_001", "chunk_002", "chunk_003"]
    sample_texts = [
        (
            "제1조(목적) 이 학칙은 조선대학교의 교육목표를 달성하기 위하여 필요한 "
            "학사운영 등에 관한 사항을 규정함을 목적으로 한다."
        ),
        "제40조(성적평가) 학업성적은 각 교과목별로 시험성적, 과제물, 출석 등을 종합하여 평가한다.",
        "제45조(졸업) 소정의 전 과정을 이수하고 졸업요건을 충족한 자에게는 학사학위를 수여한다.",
    ]
    sample_metadatas = [
        {"source": "조선대학교_학칙.pdf", "page": 1, "sec_title": "총칙"},
        {"source": "조선대학교_학칙.pdf", "page": 12, "sec_title": "성적평가"},
        {"source": "조선대학교_학칙.pdf", "page": 15, "sec_title": "졸업 및 학위"},
    ]

    db_manager.upsert_documents(ids=sample_ids, documents=sample_texts, metadatas=sample_metadatas)

    logger.info("\n--- 3. 쿼리 임베딩(embed_query) 단위 테스트 ---")
    query = "조선대학교 학칙의 목적이 뭐야?"
    query_vector = db_manager.embed_query(query)
    logger.info(f"'{query}' 임베딩 결과 (차원 수): {len(query_vector)}")

    logger.info("\n--- 4. 표준화된 검색(search) 및 점수 정규화 검증 ---")
    search_queries = [
        "조선대학교 학칙의 목적이 뭐야?",
        "졸업하려면 어떻게 해야 하나요?",
        "전혀 상관없는 이상한 질문입니다.",  # 낮은 점수 또는 예외 처리 확인용
    ]

    for q_text in search_queries:
        logger.info(f"\n[질문]: {q_text}")
        results = db_manager.search(query_text=q_text, k=2)

        if not results:
            logger.warning("검색 결과가 없습니다.")
            continue

        for i, res in enumerate(results):
            # 0~1 사이로 정규화된 score가 포함된 표준화된 dict 구조 확인
            logger.info(f"  - 순위 {i + 1} (Score: {res['score']})")
            logger.info(f"    Content: {res['content']}")
            logger.info(f"    Metadata: {res['metadata']}")
