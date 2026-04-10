import json
import logging
from collections import Counter
from pathlib import Path

from src.utils.paths import PROCESSED_DATA_DIR

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class ParsingValidator:
    def __init__(self, processed_dir: Path = PROCESSED_DATA_DIR):
        self.processed_dir = processed_dir

    def get_latest_result(self) -> Path:
        """가장 최근에 생성된 파싱 결과 파일 경로 반환"""
        json_files = sorted(self.processed_dir.glob("parsed_documents_*.json"), reverse=True)
        return json_files[0] if json_files else None

    def analyze(self):
        latest_file = self.get_latest_result()
        if not latest_file:
            logger.error("분석할 결과 파일을 찾을 수 없습니다. 먼저 loader.py를 실행하세요.")
            return

        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info("[파싱 품질 분석 리포트]")
        logger.info(f"대상 파일: {latest_file.name}")
        logger.info("=" * 60)

        total_chunks = len(data)
        sources = [d["metadata"]["source"] for d in data]
        source_counts = Counter(sources)

        logger.info(f"총 추출 청크 수: {total_chunks}")
        for src, count in source_counts.items():
            logger.info(f"   - {src}: {count} 청크")

        # 탐지 실패율 계산 (기본값 포함 여부 확인)
        no_chapter = [d for d in data if "장 없음" in d["chapter"]]
        no_article = [d for d in data if "조 없음" in d["article"]]

        chapter_success_rate = (1 - len(no_chapter) / total_chunks) * 100
        article_success_rate = (1 - len(no_article) / total_chunks) * 100

        content_lengths = [len(d["content"]) for d in data]
        avg_len = sum(content_lengths) / total_chunks
        min_len = min(content_lengths)
        max_len = max(content_lengths)

        logger.info("-" * 60)
        logger.info("[품질 지표 (탐지 성공률)]")
        logger.info(f"장(Chapter) 탐지 성공률: {chapter_success_rate:.1f}%")
        logger.info(f"조(Article) 탐지 성공률: {article_success_rate:.1f}%")
        logger.info("참고: 탐지 실패는 대개 문서 서두나 말미의 메타데이터 섹션에서 발생합니다.")

        logger.info("-" * 60)
        logger.info("[본문 길이 분석 (글자 수)]")
        logger.info(f"   - 평균 길이: {avg_len:.1f} 자")
        logger.info(f"   - 최소/최대 길이: {min_len} / {max_len} 자")

        short_chunks = [d for d in data if len(d["content"]) < 20]
        logger.info(f"20자 미만 짧은 청크: {len(short_chunks)}개 (품질 점검 필요)")

        logger.info("-" * 60)
        logger.info("[발표용 샘플 데이터 추출]")

        # 실제 '조'가 포함된 첫 번째 샘플 탐색
        sample = next((d for d in data if "제" in d["article"] and "조" in d["article"]), data[0])
        logger.info(f"   [출처]      : {sample['metadata']['source']} (P.{sample['metadata']['page']})")
        logger.info(f"   [계층 구조]  : {sample['chapter']} > {sample['article']}")
        logger.info(f"   [본문 샘플]  : {sample['content'][:150]}...")

        logger.info("=" * 60)


if __name__ == "__main__":
    validator = ParsingValidator()
    validator.analyze()
