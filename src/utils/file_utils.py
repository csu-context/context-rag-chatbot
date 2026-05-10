import hashlib
from pathlib import Path


def generate_file_hash(file_path: Path, parser_type: str = "manual") -> str:
    """
    파일명, 수정 시간, 파서 타입을 조합하여 고유 ID(Hash)를 생성합니다.
    파서 타입(.env 설정)이 변경되면 해시값이 달라져 자동 재색인이 수행됩니다.
    """
    stats = file_path.stat()
    unique_str = f"{file_path.name}_{stats.st_mtime}_{parser_type}"
    return hashlib.md5(unique_str.encode()).hexdigest()[:12]
