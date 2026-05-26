import unicodedata
from pathlib import Path


def normalize_to_nfc(text: str) -> str:
    """텍스트를 NFC 형식으로 정규화하여 반환합니다."""
    if not text:
        return text
    return unicodedata.normalize("NFC", text)


def normalize_to_nfd(text: str) -> str:
    """텍스트를 NFD 형식으로 정규화하여 반환합니다."""
    if not text:
        return text
    return unicodedata.normalize("NFD", text)


def normalize_path_to_nfc(path_val: str | Path) -> str:
    """경로 객체 또는 경로 문자열을 NFC 형식의 문자열로 정규화하여 반환합니다."""
    if not path_val:
        return ""
    return unicodedata.normalize("NFC", str(path_val))
