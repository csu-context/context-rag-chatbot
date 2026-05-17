import logging
import sys

from dotenv import load_dotenv

from src.core.retriever import EnsembleRetriever
from src.pipeline import PipelineOrchestrator
from src.utils.logger import setup_global_logging
from src.utils.paths import ensure_directories
from src.vector_db.chroma_manager import ChromaDBManager

load_dotenv()
logger = logging.getLogger(__name__)

def main():
    """
    RAG 챗봇 데이터 구축(Ingestion) CLI 도구
    """
    setup_global_logging()
    logger.info("=" * 50)
    logger.info(" RAG 데이터 구축 시스템 (Orchestrator)을 시작합니다.")
    logger.info("=" * 50)

    try:
        # 1. 필수 디렉토리 확인
        ensure_directories()

        # 2. 오케스트레이터 초기화 및 적재 실행
        orchestrator = PipelineOrchestrator()
        orchestrator.run_ingestion()

        # 3. 최종 데이터 개수 확인
        db_manager = ChromaDBManager(collection_name="rag_collection")
        final_count = db_manager.get_count()
        logger.info(f" 최종 DB 데이터 수: {final_count}")

        # 4. 앙상블 리트리버 연동 테스트
        ensemble_retriever = EnsembleRetriever()
        logger.info(f" EnsembleRetriever 로드 완료: {ensemble_retriever}")


        if final_count > 0:
            logger.info("데이터 적재가 완료되었습니다.")
        else:
            logger.info("데이터 적재가 완료되었습니다.")
            logger.warning("데이터 개수가 0입니다. data/raw/ 폴더의 PDF 파일을 확인하십시오.")

    except Exception as e:
        logger.error(f" 오류 발생: {e}", exc_info=True)
        sys.exit(1)

    logger.info("시스템 업데이트 완료. 이제 'streamlit run src/app.py'를 실행하세요.")

if __name__ == "__main__":
    # 위에서 정의한 main() 함수를 호출
    main()
