from typing import List
from langchain_core.documents import Document

def format_citations(docs: List[Document]) -> str:
    """
    검색된 문서들(Document 객체 리스트)에서 메타데이터를 추출하여
    '참조된 문서 목록' 형식을 생성합니다.
    """
    if not docs:
        return ""
        
    sources = set()
    for doc in docs:
        src_name = doc.metadata.get("src_name") or doc.metadata.get("source", "알 수 없는 파일")
        pg_num = doc.metadata.get("pg_num") or doc.metadata.get("page")
        
        # 페이지 정보가 있으면 파일명 옆에 표시, 없으면 '정보 없음' 안내
        page_info = f"p.{pg_num}" if pg_num else "페이지 정보 없음"
        sources.add(f"{src_name} ({page_info})")
        
    if not sources:
        return ""
        
    # 정렬하여 보기 좋게 반환
    sorted_sources = sorted(list(sources))
    
    citation_text = "\n\n---\n**💡 답변의 근거가 된 문서 목록:**\n"
    citation_text += "\n".join([f"- {s}" for s in sorted_sources])
    
    return citation_text
