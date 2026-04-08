import json
import logging
from pathlib import Path
from collections import Counter
from src.utils.paths import PROCESSED_DATA_DIR
from src.common.constants import MetadataFields

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class ParsingValidator:
    def __init__(self, processed_dir: Path = PROCESSED_DATA_DIR):
        self.processed_dir = processed_dir

    def get_latest_result(self) -> Path:
        """가장 최근에 생성된 전처리 결과 파일 경로 반환"""
        # pipeline.py에서 생성하는 파일 패턴으로 검색
        json_files = sorted(self.processed_dir.glob("preprocessed_v1_*.json"), reverse=True)
        return json_files[0] if json_files else None

    def analyze(self):
        latest_file = self.get_latest_result()
        if not latest_file:
            logger.error("분석할 결과 파일을 찾을 수 없습니다. 먼저 pipeline.py 또는 main.py를 실행하세요.")
            return

        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info("[전처리 품질 분석 리포트]")
        logger.info(f"대상 파일: {latest_file.name}")
        logger.info("=" * 60)

        total_sections = len(data)
        
        # 메타데이터 추출 (부모 청크 기준)
        sources = [d['metadata'][MetadataFields.SRC_NAME] for d in data]
        source_counts = Counter(sources)

        logger.info(f"총 추출 섹션(부모 청크) 수: {total_sections}")
        for src, count in source_counts.items():
            logger.info(f"   - {src}: {count} 섹션")

        # 자식 청크 통계
        total_child_chunks = sum(len(d['children']) for d in data)
        logger.info(f"총 생성된 자식 청크 수: {total_child_chunks}")

        # 탐지 실패율 계산 (기본값 포함 여부 확인)
        no_chapter = [d for d in data if "장 없음" in d['metadata'].get('chapter', '')]
        no_article = [d for d in data if "조 없음" in d['metadata'].get('article', '')]
        
        chapter_success_rate = (1 - len(no_chapter) / total_sections) * 100 if total_sections > 0 else 0
        article_success_rate = (1 - len(no_article) / total_sections) * 100 if total_sections > 0 else 0

        logger.info("-" * 60)
        logger.info("[품질 지표 (탐지 성공률)]")
        logger.info(f"장(Chapter) 탐지 성공률: {chapter_success_rate:.1f}%")
        logger.info(f"조(Article) 탐지 성공률: {article_success_rate:.1f}%")

        # 본문 길이 분석
        child_lengths = [len(c['text']) for d in data for c in d['children']]
        if child_lengths:
            avg_len = sum(child_lengths) / len(child_lengths)
            logger.info("-" * 60)
            logger.info("[자식 청크 길이 분석 (글자 수)]")
            logger.info(f"   - 평균 길이: {avg_len:.1f} 자")
            logger.info(f"   - 최소/최대 길이: {min(child_lengths)} / {max(child_lengths)} 자")
        
        logger.info("-" * 60)
        logger.info("[표준 스키마 검증 샘플]")
        if data:
            sample_parent = data[0]
            sample_child = sample_parent['children'][0] if sample_parent['children'] else None
            
            logger.info(f"   [Source ID] : {sample_parent['metadata'][MetadataFields.SOURCE_ID]}")
            logger.info(f"   [Doc Type]  : {sample_parent['metadata'][MetadataFields.DOC_TYPE]}")
            if sample_child:
                logger.info(f"   [Child ID]  : {sample_child['metadata'][MetadataFields.CHUNK_ID]}")
                logger.info(f"   [Preview]   : {sample_child['text'][:100]}...")
        
        logger.info("=" * 60)

if __name__ == "__main__":
    validator = ParsingValidator()
    validator.analyze()
