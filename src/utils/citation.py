from langchain_core.documents import Document

from src.utils.unicode import normalize_to_nfc


def normalize_text(text: str) -> str:
    """NFD(자소 분리) 한글을 NFC로 정규화하여 깨짐 현상을 방지합니다."""
    return normalize_to_nfc(text)


def format_citations(docs: list[Document]) -> str:
    """
    검색된 문서들(Document 객체 리스트)에서 메타데이터를 추출하여
    번호가 매겨진 '참조된 문서 목록' 형식을 생성합니다.
    """
    if not docs:
        return ""

    sources = []
    seen = set()

    for doc in docs:
        src_name = doc.metadata.get("src_name") or doc.metadata.get("source", "알 수 없는 파일")
        pg_num = doc.metadata.get("pg_num") or doc.metadata.get("page")

        # NFD -> NFC 정규화 적용 (한글 깨짐 방지)
        safe_src_name = normalize_text(src_name)

        # 페이지 정보 포맷팅
        page_info = f"p.{pg_num}" if pg_num else "-"
        citation_str = f"{safe_src_name} ({page_info})"

        if citation_str not in seen:
            sources.append(citation_str)
            seen.add(citation_str)

    if not sources:
        return ""

    # 번호를 매겨서 반환 (본문의 [1], [2]와 매칭)
    citation_text = "\n\n---\n**답변의 근거가 된 문서 목록:**\n"
    for i, s in enumerate(sources, 1):
        citation_text += f"- **[{i}]** {s}\n"

    return citation_text
