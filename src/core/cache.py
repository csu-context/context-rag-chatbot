import json
import logging
import uuid
from typing import Any

from src.common.config import settings
from src.vector_db.chroma_manager import ChromaDBManager

logger = logging.getLogger(__name__)


class SemanticCache:
    def __init__(self):
        self.collection_name = settings.SEMANTIC_CACHE_COLLECTION_NAME
        self.threshold = settings.SEMANTIC_CACHE_THRESHOLD
        self.db_manager = ChromaDBManager(collection_name=self.collection_name)
        self.collection = self.db_manager.collection

    def _get_valid_collection(self) -> Any:
        # 파이프라인이 flush하면 stale 컬렉션 참조가 'does not exist' 에러를 냄 → 사용 전 재검증
        try:
            self.collection.count()
        except Exception:
            logger.info("캐시 컬렉션이 만료되었거나 존재하지 않아 새로 생성합니다.")
            self.collection = self.db_manager.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.db_manager.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
            self.db_manager.collection = self.collection
        return self.collection

    def get(self, query_text: str) -> dict | None:
        # Issue 51: 멀티세션 Data Bleed 방지 — SEMANTIC_CACHE_ENABLED=False로 비활성화 가능
        if not settings.SEMANTIC_CACHE_ENABLED:
            return None
        try:
            collection = self._get_valid_collection()
            query_embedding = self.db_manager.embed_query(query_text)
            results = collection.query(query_embeddings=[query_embedding], n_results=1)

            if not results or not results.get("distances") or len(results["distances"][0]) == 0:
                return None

            distance = results["distances"][0][0]
            score = max(0.0, 1.0 - distance)

            logger.info(f"캐시 유사도 점수: {score:.4f} (임계값: {self.threshold})")

            if score >= self.threshold:
                metadata = results["metadatas"][0][0]
                sources = metadata.get("sources")
                if sources:
                    sources = json.loads(sources)

                logger.info(f"시맨틱 캐시 적중 (Query: '{query_text}')")
                return {"answer": metadata.get("answer"), "sources": sources}
            logger.info(f"시맨틱 캐시 미스 (Query: '{query_text}', Score: {score:.4f})")
        except Exception as e:
            logger.error(f"시맨틱 캐시 조회 중 오류 발생: {e}")
        return None

    def add(self, query_text: str, answer: str, sources: list) -> None:
        if not settings.SEMANTIC_CACHE_ENABLED:
            return
        try:
            collection = self._get_valid_collection()
            sources_json = json.dumps(sources) if sources else ""
            cache_id = str(uuid.uuid4())

            collection.add(
                embeddings=[self.db_manager.embed_query(query_text)],
                documents=[query_text],
                metadatas=[{"answer": answer, "sources": sources_json}],
                ids=[cache_id],
            )
            logger.info(f"시맨틱 캐시에 추가됨: '{query_text}'")
        except Exception as e:
            logger.error(f"시맨틱 캐시 추가 중 오류 발생: {e}")

    def flush(self) -> None:
        try:
            self.db_manager.client.delete_collection(name=self.collection_name)
            self.collection = self.db_manager.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.db_manager.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
            self.db_manager.collection = self.collection
            logger.info("시맨틱 캐시가 성공적으로 초기화되었습니다.")
        except Exception as e:
            logger.error(f"시맨틱 캐시 초기화 중 오류 발생: {e}")
