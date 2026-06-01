import json
import logging
import pickle
from pathlib import Path
from typing import Any

from src.common.config import settings

logger = logging.getLogger(__name__)


class StorageManager:
    """파이프라인 실행 중 발생하는 파일 캐싱 및 가공 결과(JSON) 입출력을 전담하는 저장소 관리자 클래스입니다."""

    def __init__(self, processed_dir: Path, cache_dir: Path):
        self.processed_dir = processed_dir
        self.cache_dir = cache_dir

    def get_cache_path(self, source_id: str) -> Path:
        """주어진 소스 ID에 부합하는 캐시 파일 경로를 반환합니다."""
        return self.cache_dir / f"{source_id}_parsed.pkl"

    def get_processed_path(self, source_id: str) -> Path:
        """주어진 소스 ID에 부합하는 가공 JSON 파일 경로를 반환합니다."""
        return self.processed_dir / f"{source_id}.json"

    def has_cache(self, source_id: str) -> bool:
        """캐시 파일이 물리적으로 존재하는지 여부를 검증합니다."""
        return self.get_cache_path(source_id).exists()

    def load_cache(self, source_id: str) -> Any:
        """캐시 파일을 읽어와 객체로 로드합니다."""
        path = self.get_cache_path(source_id)
        with open(path, "rb") as f:
            return pickle.load(f)

    def save_cache(self, source_id: str, data: Any) -> None:
        """가공된 파싱 데이터 객체를 캐시 파일로 저장합니다."""
        path = self.get_cache_path(source_id)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        self._evict_cache_if_needed()

    def _evict_cache_if_needed(self) -> None:
        """Issue 15: MAX_PICKLE_CACHE_FILES 초과 시 오래된 캐시 자동 삭제."""
        cache_files = sorted(self.cache_dir.glob("*_parsed.pkl"), key=lambda p: p.stat().st_mtime)
        over = len(cache_files) - settings.MAX_PICKLE_CACHE_FILES
        if over <= 0:
            return
        for old_file in cache_files[:over]:
            try:
                old_file.unlink()
                logger.info(f"Pickle 캐시 Eviction: {old_file.name}")
            except Exception as e:
                logger.warning(f"캐시 삭제 실패 {old_file.name}: {e}")

    def save_processed_data(self, source_id: str, data: Any) -> Path:
        """최종 청크 가공 데이터를 JSON 형식의 물리 파일로 영속화합니다."""
        path = self.get_processed_path(source_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def delete_processed_and_cache(self, source_id: str) -> bool:
        """특정 소스 ID에 매핑되는 캐시와 가공 JSON 파일을 일괄 삭제 처리합니다."""
        removed = False
        cache_path = self.get_cache_path(source_id)
        if cache_path.exists():
            cache_path.unlink()

        json_path = self.get_processed_path(source_id)
        if json_path.exists():
            json_path.unlink()
            removed = True
        return removed

    def scan_processed_files(self) -> list[Path]:
        """가공 파일 디렉토리 하위의 모든 JSON 파일 목록을 스캔합니다."""
        return list(self.processed_dir.glob("*.json"))

    def load_processed_file(self, file_path: Path) -> Any:
        """특정 JSON 데이터 파일을 역직렬화하여 읽습니다."""
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)

    def delete_file(self, file_path: Path) -> None:
        """지정된 물리 파일을 시스템상에서 삭제합니다."""
        if file_path.exists():
            file_path.unlink()
