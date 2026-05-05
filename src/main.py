import logging
import sys

from src.pipeline import PipelineOrchestrator
from src.utils.logger import setup_global_logging
from src.utils.paths import ensure_directories
from src.vector_db.chroma_manager import ChromaDBManager

# 로깅 설정
logger = logging.getLogger(__name__)


def main():
    """
    RAG 챗봇 데이터 구축(Ingestion) CLI 도구.
    """
    setup_global_logging()
    logger.info("=" * 50)
    logger.info("🚀 RAG 데이터 구축 시스템 (Orchestrator)을 시작합니다.")
    logger.info("=" * 50)

    try:
        # 1. 필수 디렉토리 확인 및 생성
        ensure_directories()

        # 2. 오케스트레이터 초기화 및 실행
        orchestrator = PipelineOrchestrator()
        orchestrator.run_ingestion()

        # 3. 결과 확인
        db_manager = ChromaDBManager(collection_name="rag_collection")
        final_count = db_manager.get_count()
        logger.info(f"최종 DB 데이터 수: {final_count}")

    except Exception as e:
        logger.error(f"❌ 데이터 구축 중 오류 발생: {e}", exc_info=True)
        sys.exit(1)

    logger.info("시스템이 최신 데이터로 업데이트되었습니다. 이제 'streamlit run src/app.py'를 실행하세요.")


if __name__ == "__main__":
    main()
