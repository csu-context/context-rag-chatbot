import hashlib
from pathlib import Path

from src.utils.paths import RAW_DATA_DIR


def generate_file_hash(file_path: Path, parser_type: str = "manual") -> str:
    """
    상대 경로, 수정 시간, 파서 타입을 조합하여 고유 ID(Hash)를 생성합니다.
    파서 타입(.env 설정)이 변경되거나 파일 위치가 변경되면 해시값이 달라져 자동 재색인이 수행됩니다.
    """
    stats = file_path.stat()
    try:
        # RAW_DATA_DIR 기준 상대 경로 포함 (중복 방지 강화)
        relative_path = file_path.relative_to(RAW_DATA_DIR)
    except ValueError:
        # RAW_DATA_DIR 외부에 있는 경우 파일명만 사용
        relative_path = file_path.name

    unique_str = f"{relative_path}_{stats.st_mtime}_{parser_type}"
    return hashlib.md5(unique_str.encode()).hexdigest()[:12]
