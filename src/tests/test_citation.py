from langchain_core.documents import Document
from src.utils.citation import format_citations

def run_test():
    print("🔍 [Citation] 기능 단위 테스트 시작")
    print("=" * 40)
    
    # 테스트 1: 기본 포맷팅
    docs1 = [
        Document(page_content="내용1", metadata={"src_name": "manual.pdf", "pg_num": 5}),
        Document(page_content="내용2", metadata={"src_name": "guide.docx", "pg_num": 10})
    ]
    result1 = format_citations(docs1)
    assert "**💡 답변의 근거가 된 문서 목록:**" in result1
    assert "- manual.pdf (p.5)" in result1
    assert "- guide.docx (p.10)" in result1
    print("  ✅ 테스트 1: 기본 포맷팅 성공")
    
    # 테스트 2: 중복 제거
    docs2 = [
        Document(page_content="내용1", metadata={"src_name": "manual.pdf", "pg_num": 5}),
        Document(page_content="내용2", metadata={"src_name": "manual.pdf", "pg_num": 5})
    ]
    result2 = format_citations(docs2)
    assert result2.count("manual.pdf (p.5)") == 1
    print("  ✅ 테스트 2: 중복 제거 성공")
    
    # 테스트 3: 대비책 필드 (fallback) 및 페이지 정보 없음
    docs3 = [
        Document(page_content="내용", metadata={"source": "legacy.pdf", "page": 1}),
        Document(page_content="내용2", metadata={"src_name": "no_page.pdf"}) # pg_num 없음
    ]
    result3 = format_citations(docs3)
    assert "- legacy.pdf (p.1)" in result3
    assert "- no_page.pdf (페이지 정보 없음)" in result3
    print("  ✅ 테스트 3: 대비책 필드 및 페이지 정보 없음 처리 성공")
    
    print("=" * 40)
    print("🚀 모든 단위 테스트 통과!")
    print("\n--- 실제 출력 예시 ---")
    print(result1)
    print("--------------------")

if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(f"  ❌ 테스트 실패: {e}")
        exit(1)
