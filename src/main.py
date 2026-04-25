import logging
import sys

from src.pipeline import PreprocessingPipeline
from src.utils.paths import ensure_directories
from src.vector_db.chroma_manager import ChromaDBManager

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """
    RAG 챗봇 데이터 구축(Ingestion) CLI 도구.
    
    기능:
    1. data/raw 디렉토리의 문서(PDF, MD) 스캔
    2. 문서 파싱 및 계층적 청킹 (Parent-Child)
    3. 전처리 결과를 JSON으로 저장 (data/processed)
    4. ChromaDB 벡터 데이터베이스에 인덱싱 (Upsert)
    """
    logger.info("="*50)
    logger.info("🚀 RAG 데이터 구축 시스템을 시작합니다.")
    logger.info("="*50)

    try:
        # 1. 필수 디렉토리 확인 및 생성
        ensure_directories()

        # 2. 벡터 DB 상태 확인
        db_manager = ChromaDBManager(collection_name="rag_collection")
        initial_count = db_manager.get_count()
        logger.info(f"현재 DB에 저장된 데이터 수: {initial_count}")

        # 3. 전처리 및 인덱싱 파이프라인 실행
        logger.info("데이터 전처리 및 인덱싱 시작 (data/raw -> ChromaDB)...")
        pipeline = PreprocessingPipeline()
        processed_data = pipeline.run()

        if processed_data:
            total_parents = len(processed_data)
            total_children = sum(len(p["children"]) for p in processed_data)
            
            final_count = db_manager.get_count()
            
            logger.info("-" * 50)
            logger.info("✅ 데이터 구축 완료!")
            logger.info(f"- 처리된 문서 섹션 수: {total_parents}")
            logger.info(f"- 생성된 총 청크 수: {total_children}")
            logger.info(f"- 최종 DB 데이터 수: {final_count} (증가량: {final_count - initial_count})")
            logger.info("-" * 50)
        else:
            logger.warning("처리할 데이터가 없거나 전처리에 실패했습니다.")
            logger.info("data/raw 폴더에 PDF 또는 Markdown 파일이 있는지 확인해주세요.")

    except Exception as e:
        logger.error(f"❌ 데이터 구축 중 오류 발생: {e}", exc_info=True)
        sys.exit(1)

    logger.info("시스템이 최신 데이터로 업데이트되었습니다. 이제 'streamlit run src/app.py'를 실행하세요.")


if __name__ == "__main__":
    main()
