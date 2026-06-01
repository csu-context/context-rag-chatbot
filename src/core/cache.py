import json
import logging
import uuid

from src.common.config import settings
from src.vector_db.chroma_manager import ChromaDBManager

logger = logging.getLogger(__name__)


class SemanticCache:
    def __init__(self):
        self.collection_name = settings.SEMANTIC_CACHE_COLLECTION_NAME
        self.threshold = settings.SEMANTIC_CACHE_THRESHOLD
        self.db_manager = ChromaDBManager(collection_name=self.collection_name)

    def get(self, query_text: str) -> dict | None:
        try:
            results = self.db_manager.search(query_text, k=1)

            if not results:
                return None

            top_result = results[0]
            score = top_result["score"]

            logger.info(f"캐시 유사도 점수: {score:.4f} (임계값: {self.threshold})")

            if score >= self.threshold:
                metadata = top_result.get("metadata", {})
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
        try:
            sources_json = json.dumps(sources) if sources else ""
            cache_id = str(uuid.uuid4())

            self.db_manager.upsert_documents(
                ids=[cache_id], documents=[query_text], metadatas=[{"answer": answer, "sources": sources_json}]
            )
            logger.info(f"시맨틱 캐시에 추가됨: '{query_text}'")
        except Exception as e:
            logger.error(f"시맨틱 캐시 추가 중 오류 발생: {e}")

    def flush(self) -> None:
        try:
            self.db_manager.reset_collection()
            logger.info("시맨틱 캐시가 성공적으로 초기화되었습니다.")
        except Exception as e:
            logger.error(f"시맨틱 캐시 초기화 중 오류 발생: {e}")
