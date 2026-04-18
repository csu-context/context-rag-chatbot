from typing import List
from langchain_core.documents import Document

def format_citations(docs: List[Document]) -> str:
    """
    검색된 문서들(Document 객체 리스트)에서 메타데이터를 추출하여
    '참조된 문서 목록' 형식을 생성합니다.
    
    본문의 세부 인용([파일명, p.XX])과 중복되지 않도록 
    전체적인 출처 정보를 요약하여 제공합니다.
    """
    if not docs:
        return ""
        
    sources = set()
    for doc in docs:
        src_name = doc.metadata.get("src_name") or doc.metadata.get("source", "알 수 없는 파일")
        # 파일명만 추출하여 중복 제거 (페이지 단위 중복 방지)
        sources.add(src_name)
        
    if not sources:
        return ""
        
    # 정렬하여 보기 좋게 반환
    sorted_sources = sorted(list(sources))
    
    citation_text = "\n\n---\n**💡 답변의 근거가 된 문서 목록:**\n"
    citation_text += "\n".join([f"- {s}" for s in sorted_sources])
    
    return citation_text
