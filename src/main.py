import logging
from src.utils.paths import ensure_directories
from src.pipeline import PreprocessingPipeline

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    RAG 챗봇 프로젝트의 메인 진입점.
    1. 필수 디렉토리 확인
    2. 데이터 전처리 파이프라인 실행 (PDF/MD 파싱 및 청킹)
    """
    logger.info("RAG 챗봇 시스템 초기화 중...")
    
    # 1. 필수 디렉토리 확인 및 생성
    ensure_directories()
    
    # 2. 전처리 파이프라인 실행
    logger.info("데이터 전처리 파이프라인 시작...")
    pipeline = PreprocessingPipeline()
    processed_data = pipeline.run()
    
    if processed_data:
        logger.info(f"전처리 완료: {len(processed_data)}개의 섹션(부모 청크)이 처리되었습니다.")
    else:
        logger.warning("전처리된 데이터가 없습니다. data/raw 폴더를 확인하세요.")
    
    logger.info("초기화 완료. 메인 로직을 시작할 준비가 되었습니다.")
    # TODO: 3. 벡터 DB 로드 및 챗봇 엔진 실행

if __name__ == "__main__":
    main()
