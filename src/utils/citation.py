from langchain_core.documents import Document

from src.utils.unicode import normalize_to_nfc


def format_citations(docs: list[Document]) -> str:
    """검색된 문서들에서 메타데이터를 추출하여 번호가 매겨진 참조 문서 목록을 반환합니다."""
    if not docs:
        return ""

    sources = []
    seen = set()

    for doc in docs:
        src_name = doc.metadata.get("src_name") or doc.metadata.get("source", "알 수 없는 파일")
        pg_num = doc.metadata.get("pg_num") or doc.metadata.get("page")

        safe_src_name = normalize_to_nfc(src_name)

        page_info = f"p.{pg_num}" if pg_num else "-"
        citation_str = f"{safe_src_name} ({page_info})"

        if citation_str not in seen:
            sources.append(citation_str)
            seen.add(citation_str)

    if not sources:
        return ""

    citation_text = "\n\n---\n**답변의 근거가 된 문서 목록:**\n"
    for i, s in enumerate(sources, 1):
        citation_text += f"- **[{i}]** {s}\n"

    return citation_text
