import json
import logging
import uuid

from src.vector_db.chroma_manager import ChromaDBManager

logger = logging.getLogger(__name__)


class SemanticCache:
    def __init__(self, collection_name="semantic_cache", threshold=0.95):
        self.db_manager = ChromaDBManager(collection_name=collection_name)
        self.collection = self.db_manager.collection
        self.threshold = threshold

    def _get_valid_collection(self):
        """
        다른 프로세스(파이프라인)에 의해 컬렉션이 삭제(flush)되었을 경우,
        기존의 stale한 collection 객체를 계속 사용하면 'does not exist' 에러가 발생합니다.
        이를 방지하기 위해 사용 전 컬렉션의 유효성을 검증하고 필요시 재할당합니다.
        """
        try:
            self.collection.count()
        except Exception:
            logger.info("캐시 컬렉션이 만료되었거나 존재하지 않아 새로 생성합니다.")
            self.collection = self.db_manager.client.get_or_create_collection(
                name=self.db_manager.collection_name,
                embedding_function=self.db_manager.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
            self.db_manager.collection = self.collection
        return self.collection

    def get(self, query_text):
        try:
            collection = self._get_valid_collection()
            query_embedding = self.db_manager.embed_query(query_text)
            results = collection.query(query_embeddings=[query_embedding], n_results=1)

            if not results or not results.get("distances") or len(results["distances"][0]) == 0:
                return None

            distance = results["distances"][0][0]
            score = max(0.0, 1.0 - distance)

            logger.info(f"Cache similarity score for '{query_text}': {score:.4f} (Threshold: {self.threshold})")

            if score >= self.threshold:
                metadata = results["metadatas"][0][0]
                sources = metadata.get("sources")
                if sources:
                    sources = json.loads(sources)

                logger.info(f"Cache HIT for query: '{query_text}'")
                return {"answer": metadata.get("answer"), "sources": sources}
            logger.info(f"Cache MISS for query: '{query_text}' (score {score:.4f} < {self.threshold})")
        except Exception as e:
            logger.error(f"Error accessing semantic cache: {e}")
        return None

    def add(self, query_text, answer, sources):
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
            logger.info(f"Added to semantic cache: '{query_text}'")
        except Exception as e:
            logger.error(f"Error adding to semantic cache: {e}")

    def flush(self):
        try:
            self.db_manager.client.delete_collection(name=self.db_manager.collection_name)
            self.collection = self.db_manager.client.get_or_create_collection(
                name=self.db_manager.collection_name,
                embedding_function=self.db_manager.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
            self.db_manager.collection = self.collection
            logger.info("Semantic cache flushed successfully.")
        except Exception as e:
            logger.error(f"Error flushing semantic cache: {e}")
