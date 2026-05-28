import logging
import os
import shutil
import time
import warnings
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings

from src.common.config import settings
from src.common.constants import MetadataFields
from src.core.base_retriever import BaseRetriever
from src.models.embedder import BGEEmbedder
from src.utils.paths import BM25_CACHE_DIR, CACHE_DIR, PROCESSED_DATA_DIR, VECTOR_DB_DIR, ensure_directories
from src.utils.unicode import normalize_to_nfc, normalize_to_nfd

# 경고 숨기기 로직 추가
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", module="huggingface_hub")

# 로깅 설정
logger = logging.getLogger(__name__)


class BGEChromaEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB 커스텀 임베딩 함수
    src.models.embedder의 BGEEmbedder를 사용하여 문서를 벡터화합니다.
    """

    def __init__(self, model_name: str | None = None):
        # 모델명만 미리 저장하고, 임베더 인스턴스는 실제 임베딩 요청 시 생성 (Lazy Loading)
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self._embedder = None

    @property
    def embedder(self) -> BGEEmbedder:
        if self._embedder is None:
            logger.info("Initializing BGEEmbedder (Lazy Loading)...")
            self._embedder = BGEEmbedder(model_name=self.model_name)
        return self._embedder

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.embedder.encode(input)
        return embeddings.tolist()


class ChromaDBManager(BaseRetriever):
    _shared_client = None  # 동일 프로세스 내에서 중복 클라이언트 생성 및 파일 락 충돌 방지를 위한 공유 클라이언트

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

        # 공통 설정 변수로 추출 (DRY 원칙 적용)
        common_settings = Settings(anonymized_telemetry=False)

        for attempt in range(max_retries):
            try:
                if ChromaDBManager._shared_client is not None:
                    self.client = ChromaDBManager._shared_client
                else:
                    self.client = self._create_client(chroma_host, chroma_port, common_settings)
                    ChromaDBManager._shared_client = self.client

                # 컬렉션 로드 (실질적인 연결 테스트 구간)
                # hnsw:num_threads=1: Python 3.13 + chromadb Rust 바인딩의 멀티스레드 segfault 방지
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_fn,
                    metadata={"hnsw:space": "cosine", "hnsw:num_threads": 1},
                )

                self._check_config_and_auto_reset()
                logger.info(f"ChromaDB 로드 완료. (컬렉션: {self.collection_name})")
                return  # 성공 시 루프 탈출

            except Exception as e:
                # DB 손상 또는 메타데이터 에러 감지 시 자가 치유 시도
                if not chroma_host and attempt < max_retries - 1:
                    logger.warning(
                        f"[자가 치유] ChromaDB 초기화 중 오류 감지 (DB 손상 또는 버전 불일치 가능성): {e}. "
                        "로컬 DB 디렉토리를 완전히 삭제하고 자동 재구성을 수행합니다."
                    )
                    ChromaDBManager._shared_client = None
                    self.client = None
                    if VECTOR_DB_DIR.exists():
                        try:
                            shutil.rmtree(VECTOR_DB_DIR)
                        except Exception as rm_e:
                            logger.error(f"로컬 DB 디렉토리 삭제 실패: {rm_e}")
                    ensure_directories()
                    time.sleep(retry_delay)
                    continue

                if attempt < max_retries - 1:
                    logger.warning(
                        f"ChromaDB 연결 실패. {retry_delay}초 후 재시도합니다. "
                        f"({attempt + 1}/{max_retries}) | 오류: {e}"
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error("ChromaDB 연결에 최종적으로 실패했습니다. DB 상태를 확인하시기 바랍니다.")
                    raise RuntimeError("ChromaDB initialization failed.") from e

    def _create_client(self, chroma_host, chroma_port, common_settings):
        if chroma_host:
            logger.info(f"ChromaDB 서버 모드 접속 시도 (Host: {chroma_host}, Port: {chroma_port})")
            return chromadb.HttpClient(host=chroma_host, port=int(chroma_port), settings=common_settings)

        ensure_directories()
        logger.info(f"ChromaDB 로컬 모드 활성화 (Path: {VECTOR_DB_DIR})")
        try:
            return chromadb.PersistentClient(path=str(VECTOR_DB_DIR), settings=common_settings)
        except Exception as e:
            logger.error(f"ChromaDB PersistentClient 초기화 실패 (DB 파일 손상 가능성): {e}")
            logger.info("기존 DB 폴더를 삭제하고 재생성하여 자동 초기화를 시도합니다.")
            if VECTOR_DB_DIR.exists():
                shutil.rmtree(VECTOR_DB_DIR)
            ensure_directories()
            return chromadb.PersistentClient(path=str(VECTOR_DB_DIR), settings=common_settings)

    def _check_config_and_auto_reset(self):
        """임베딩 모델 및 청크 크기 변경을 감지하여 자동 리셋을 수행합니다."""
        try:
            existing_metadata = self.collection.metadata
            current_model = self.embedding_fn.model_name

            from src.processing.chunking import HierarchicalChunker

            chunker = HierarchicalChunker()
            current_parent_size = chunker.parent_chunk_size
            current_child_size = chunker.child_chunk_size

            needs_reset = False
            reset_reason = ""

            if existing_metadata:
                existing_model = existing_metadata.get("embedding_model")
                existing_parent = existing_metadata.get("parent_chunk_size")
                existing_child = existing_metadata.get("child_chunk_size")

                if existing_model is not None and existing_model != current_model:
                    needs_reset = True
                    reset_reason = f"임베딩 모델 변경 ({existing_model} -> {current_model})"
                elif existing_parent is not None and int(existing_parent) != current_parent_size:
                    needs_reset = True
                    reset_reason = f"부모 청크 크기 변경 ({existing_parent} -> {current_parent_size})"
                elif existing_child is not None and int(existing_child) != current_child_size:
                    needs_reset = True
                    reset_reason = f"자식 청크 크기 변경 ({existing_child} -> {current_child_size})"

            if needs_reset:
                logger.warning(
                    f"[자가 치유] {reset_reason} 감지. "
                    "데이터 정합성 및 무결성 유지를 위해 기존 데이터를 자동 초기화하고 전체 재색인을 유도합니다."
                )
                self.reset_collection()
                self._clear_processed_and_cache_files()

            # 메타데이터 업데이트 (120자 라인 한도 준수하여 가로 분할)
            new_metadata = dict(existing_metadata or {})
            new_metadata["hnsw:space"] = "cosine"
            new_metadata["hnsw:num_threads"] = 1
            new_metadata["embedding_model"] = current_model
            new_metadata["parent_chunk_size"] = current_parent_size
            new_metadata["child_chunk_size"] = current_child_size

            # 메타데이터에 변화가 있는 경우에만 modify 호출
            if not existing_metadata or any(new_metadata.get(k) != existing_metadata.get(k) for k in new_metadata):
                # ChromaDB는 컬렉션 생성 후 hnsw: 관련 메타데이터 변경을 지원하지 않으므로 제외하고 수정함
                modify_metadata = {k: v for k, v in new_metadata.items() if not k.startswith("hnsw:")}
                self.collection.modify(metadata=modify_metadata)

        except Exception as e:
            logger.error(f"설정 검증 및 자동 초기화 중 오류 발생: {e}")
            raise

    def embed_query(self, query_text: str) -> list[float]:
        """
        사용자 쿼리를 벡터(Embedding)로 변환합니다.

        Raises:
            RuntimeError: 임베딩 생성 실패 시 하이브리드 리트리버 등
                          상위 모듈에서 방어 로직을 수행할 수 있도록 에러를 전파합니다.
        """
        try:
            return self.embedding_fn([query_text])[0]
        except Exception as e:
            logger.error(f"쿼리 임베딩 중 오류 발생: {e}")
            # 빈 리스트를 반환하지 않고 에러를 명시적으로 발생시킴
            raise RuntimeError(f"Failed to embed query: '{query_text}'") from e

    def _get_valid_collection(self):
        """
        ChromaDB 서버 리셋 등으로 컬렉션 레퍼런스가 만료되었을 때
        'does not exist' 에러를 방지하기 위해 유효성을 검증하고 필요시 재로딩합니다.
        """
        try:
            self.collection.count()
        except Exception as e:
            if "does not exist" in str(e):
                logger.warning(f"컬렉션 '{self.collection_name}'이 존재하지 않아 재초기화합니다.")
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_fn,
                    metadata={"hnsw:space": "cosine", "hnsw:num_threads": 1},
                )
        return self.collection

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
            collection = self._get_valid_collection()
            collection.upsert(ids=ids, documents=documents, metadatas=cleaned_metadatas)
            logger.info(f"{len(ids)}개의 문서 청크가 ChromaDB에 성공적으로 업서트되었습니다.")
        except Exception as e:
            logger.error(f"문서 업서트 중 오류 발생: {e}")

    def search(self, query_text: str, k: int = 3) -> list[dict[str, Any]]:
        """
        주어진 쿼리 텍스트와 가장 유사한 문서를 검색하고,
        하이브리드 리트리버와 호환되는 표준화된 dict 리스트 형태로 반환합니다.
        """
        try:
            collection = self._get_valid_collection()
            # ChromaDB query 호출
            results = collection.query(query_texts=[query_text], n_results=k)

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

    def retrieve(self, query: str, n: int = 5) -> list[dict[str, Any]]:
        """BaseRetriever 인터페이스 구현. ChromaDB 검색을 실행합니다."""
        return self.search(query_text=query, k=n)

    def get_count(self) -> int:
        """현재 컬렉션에 저장된 총 청크 수를 반환합니다."""
        try:
            collection = self._get_valid_collection()
            return collection.count()
        except Exception:
            return 0

    def get_source_count(self, source_name: str) -> int:
        """특정 소스 파일명(metadata.src_name)에 해당하는 청크 수를 반환합니다."""
        nfc_name = normalize_to_nfc(source_name)
        nfd_name = normalize_to_nfd(source_name)
        try:
            collection = self._get_valid_collection()
            results = collection.get(
                where={MetadataFields.SRC_NAME: {"$in": [nfc_name, nfd_name]}},
                include=[],  # 실제 데이터는 필요 없으므로 빈 리스트
            )
            return len(results["ids"])
        except Exception as e:
            logger.error(f"소스별 카운트 조회 중 오류 발생 ({source_name}): {e}")
            return 0

    def get_source_chunks(self, source_name: str) -> list[dict[str, Any]]:
        """특정 소스 파일명(metadata.src_name)에 해당하는 청크 텍스트와 메타데이터 목록을 반환합니다."""
        nfc_name = normalize_to_nfc(source_name)
        nfd_name = normalize_to_nfd(source_name)
        try:
            collection = self._get_valid_collection()
            results = collection.get(
                where={MetadataFields.SRC_NAME: {"$in": [nfc_name, nfd_name]}},
                include=["documents", "metadatas"],
            )
            chunks = []
            if results and "ids" in results:
                for i in range(len(results["ids"])):
                    chunks.append(
                        {
                            "id": results["ids"][i],
                            "content": results["documents"][i] if results["documents"] else "",
                            "metadata": results["metadatas"][i] if results["metadatas"] else {},
                        }
                    )
            return chunks
        except Exception as e:
            logger.error(f"소스별 청크 데이터 조회 중 오류 발생 ({source_name}): {e}")
            return []

    def delete_documents(self, where: dict[str, Any]):
        """
        조건(where)에 맞는 도큐먼트들을 컬렉션에서 삭제합니다.
        where 필터로 ID를 먼저 조회한 후 ID 기반으로 삭제하여 신뢰성을 높입니다.
        """
        if not where:
            logger.warning("삭제 조건이 없어 DB 삭제를 건너뜁니다.")
            return

        try:
            collection = self._get_valid_collection()
            # 1. where 필터를 사용해 삭제 대상 문서들의 ID를 먼저 조회
            results_to_delete = collection.get(where=where, include=[])
            ids_to_delete = results_to_delete["ids"]

            if not ids_to_delete:
                logger.info(f"삭제할 도큐먼트가 없습니다 (조건: {where})")
                return

            # 2. 조회된 ID 리스트를 기반으로 명시적 삭제
            logger.info(f"ChromaDB에서 {len(ids_to_delete)}개 도큐먼트 삭제 시도 (조건: {where})")
            collection.delete(ids=ids_to_delete)
            logger.info("삭제 작업 완료.")

        except Exception as e:
            logger.error(f"ChromaDB 도큐먼트 삭제 실패: {e}", exc_info=True)
            raise

    def reset_collection(self):
        """컬렉션의 모든 문서를 삭제하여 초기화합니다."""
        try:
            collection = self._get_valid_collection()
            # delete_collection() + create_collection() 방식은 Windows에서
            # 다른 클라이언트가 세그먼트 파일을 열고 있을 때 WinError 32가 발생하므로,
            # 문서 전체를 ID 기반으로 삭제하는 방식을 사용합니다.
            all_ids = collection.get(include=[])["ids"]
            if all_ids:
                collection.delete(ids=all_ids)
            logger.info(f"ChromaDB 컬렉션 '{self.collection_name}' 초기화 완료. ({len(all_ids)}개 문서 삭제)")
        except Exception as e:
            logger.error(f"ChromaDB 컬렉션 초기화 중 오류 발생: {e}")
            raise

    def _clear_processed_and_cache_files(self):
        """임베딩 모델 또는 설정 변경 시, 로컬 가공 및 캐시 파일들을 제거합니다."""

        logger.info("가공 데이터(processed) 및 파서 캐시(cache) 초기화 중...")

        if PROCESSED_DATA_DIR.exists():
            for f in PROCESSED_DATA_DIR.glob("*.json"):
                self._safe_unlink(f)
            self._safe_unlink(PROCESSED_DATA_DIR / "manifest.json")

        if CACHE_DIR.exists():
            for f in CACHE_DIR.glob("*.pkl"):
                self._safe_unlink(f)

        self._clear_bm25_cache_directory()
        logger.info("가공 및 캐시 파일 물리적 삭제 완료.")

    def _safe_unlink(self, file_path: Path):
        """안전하게 파일을 삭제합니다."""
        try:
            file_path.unlink(missing_ok=True)
        except Exception as ex:
            logger.error(f"파일 {file_path.name} 삭제 실패: {ex}")

    def _clear_bm25_cache_directory(self):
        """BM25 캐시 디렉토리를 안전하게 삭제합니다."""
        bm25_cache_dir = BM25_CACHE_DIR
        if bm25_cache_dir.exists():
            try:
                shutil.rmtree(bm25_cache_dir)
            except Exception as ex:
                logger.error(f"BM25 캐시 디렉토리 삭제 실패: {ex}")


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
