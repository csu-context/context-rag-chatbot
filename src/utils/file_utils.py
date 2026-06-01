import hashlib
from pathlib import Path

from src.utils.paths import RAW_DATA_DIR
from src.utils.unicode import normalize_to_nfc


def generate_file_hash(file_path: Path, parser_type: str = "manual") -> str:
    """
    [DataOps] 파일의 실제 본문 콘텐츠 해시와 파서 타입을 조합하여 고유 ID(Hash)를 생성합니다.
    단순 st_mtime 변경으로 인한 무의미한 재색인을 완전히 방지합니다.
    """
    try:
        # RAW_DATA_DIR 기준 상대 경로 포함
        relative_path = file_path.relative_to(RAW_DATA_DIR)
    except ValueError:
        relative_path = file_path.name

    # 파일의 실제 바이트 콘텐츠 해싱
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        content_hash = hasher.hexdigest()
    except Exception:
        # 에러 발생 시 파일 수정 시간/크기로 안전히 폴백
        stats = file_path.stat()
        content_hash = f"fallback_{stats.st_size}_{stats.st_mtime}"

    normalized_path_str = normalize_to_nfc(str(relative_path))
    unique_str = f"{normalized_path_str}_{content_hash}_{parser_type}"
    return hashlib.md5(unique_str.encode()).hexdigest()[:12]

