from pathlib import Path

from src.utils.unicode import normalize_path_to_nfc, normalize_to_nfc, normalize_to_nfd


def test_normalize_to_nfc():
    import unicodedata

    nfd_combined_char = unicodedata.normalize("NFD", "한글")

    # 정규화 수행
    nfc_result = normalize_to_nfc(nfd_combined_char)

    assert nfc_result == "한글"
    assert unicodedata.normalize("NFD", nfc_result) != nfc_result


def test_normalize_to_nfd():
    nfc_text = "한글"
    nfd_result = normalize_to_nfd(nfc_text)

    import unicodedata

    assert nfd_result == unicodedata.normalize("NFD", "한글")


def test_normalize_path_to_nfc():
    import unicodedata

    nfd_path = unicodedata.normalize("NFD", "데이터/테스트_문서.pdf")
    path_obj = Path(nfd_path)

    nfc_str_from_str = normalize_path_to_nfc(nfd_path)
    nfc_str_from_path = normalize_path_to_nfc(path_obj)

    assert nfc_str_from_str == "데이터/테스트_문서.pdf"
    assert nfc_str_from_path == "데이터/테스트_문서.pdf"
