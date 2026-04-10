import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.data.parser import ManualParser
from src.utils.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_directories

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DocumentLoader:
    def __init__(self, raw_dir: Path = RAW_DATA_DIR, processed_dir: Path = PROCESSED_DATA_DIR):
        """
        문서 로더 초기화
        :param raw_dir: 원본 파일이 위치한 디렉토리
        :param processed_dir: 파싱 결과가 저장될 디렉토리
        """
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir

        # 디렉토리 생성 및 확인
        ensure_directories()

        # 지원하는 확장자
        self.supported_extensions = [".pdf", ".md", ".markdown"]

    def scan_and_parse(self, save_filename: str = None) -> List[Dict[str, Any]]:
        """
        원본 디렉토리를 스캔하여 지원되는 모든 파일을 파싱함
        :param save_filename: 저장할 파일명 (지정하지 않으면 자동 생성)
        :return: 파싱된 모든 청크 리스트
        """
        all_chunks = []

        # 지원하는 확장자 파일 목록 수집
        files_to_parse = []
        for ext in self.supported_extensions:
            files_to_parse.extend(list(self.raw_dir.glob(f"**/*{ext}")))

        if not files_to_parse:
            logger.warning(f"파싱할 파일을 찾을 수 없습니다: {self.raw_dir}")
            return []

        logger.info(f"총 {len(files_to_parse)}개의 파일을 발견했습니다. 파싱을 시작합니다.")

        for file_path in files_to_parse:
            relative_path = file_path.relative_to(self.raw_dir)
            logger.info(f"파싱 진행 중: {relative_path}")

            try:
                parser = ManualParser(str(relative_path))
                chunks = parser.parse()
                all_chunks.extend(chunks)
                logger.info(f"파싱 완료: {len(chunks)}개 청크 추출됨")
            except Exception as e:
                logger.error(f"파싱 실패 ({relative_path}): {e}")

        # 결과 저장
        if all_chunks:
            if not save_filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_filename = f"parsed_documents_{timestamp}.json"

            save_path = self.processed_dir / save_filename
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, ensure_ascii=False, indent=2)

            logger.info(f"파싱 결과가 저장되었습니다: {save_path}")
            logger.info(f"총 추출된 청크 수: {len(all_chunks)}")

        return all_chunks


if __name__ == "__main__":
    loader = DocumentLoader()
    loader.scan_and_parse()
