import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from pathlib import Path

# 파이썬 모듈 경로 강제 추가
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.vector_db.chroma_manager import ChromaDBManager


def test_pdf_to_chroma():
    print("\n🚀 [1단계] 테스트 스크립트가 시작되었습니다!")

    # 1. 파일 경로 확인
    pdf_path = PROJECT_ROOT / "data" / "raw" / "조선대학교_학칙.pdf"

    if not pdf_path.exists():
        print(f"\n❌ [문제 발생] PDF 파일을 찾을 수 없습니다: {pdf_path}")
        return

    # 2. PDF 로드 및 청킹
    print("\n✅ [2단계] PDF 로드 및 텍스트 분할(Chunking) 시작...")
    try:
        loader = PyPDFLoader(str(pdf_path))
        raw_docs = loader.load()
    except Exception as e:
        print(f"❌ PDF 로드 중 에러 발생: {e}")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(raw_docs)
    print(f"✅ 총 {len(chunks)}개의 청크(조각)로 분할 완료!")

    # 3. 데이터 파싱
    ids = [f"rules_chunk_{i}" for i in range(len(chunks))]
    documents = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata if isinstance(chunk.metadata, dict) else dict(chunk.metadata) for chunk in chunks]

    # 4. ChromaDB 연동
    print("\n🗄️ [3단계] ChromaDB에 데이터를 저장(Upsert)합니다...")
    db_manager = ChromaDBManager(collection_name="chosun_rules_collection")
    db_manager.upsert_documents(ids=ids, documents=documents, metadatas=metadatas)
    print("✅ DB 저장 완료!")

    # 5. 검색 테스트
    print("\n================== 🔍 [4단계] 유사도 검색 결과 ==================")
    test_queries = [
        "조선대학교 학칙의 제정 목적이 무엇인가요?",
        "졸업 요건이나 기준은 어떻게 되나요?",
        "성적 평가는 어떤 방식으로 이루어집니까?"
    ]

    for query in test_queries:
        print(f"\n[질문]: {query}")
        results = db_manager.search(query_text=query, k=2)

        if not results:
            print("  ⚠️ 검색 결과가 하나도 없습니다.")
            continue

        for i, res in enumerate(results):
            score = res.get('score', 0)
            content = res.get('content', '').replace('\n', ' ')[:100] + "..."
            page = res.get('metadata', {}).get('page', '알수없음')
            display_page = page + 1 if isinstance(page, int) else page

            print(f"  -> 순위 {i + 1} (Score: {score:.4f}) | 출처: {display_page}페이지")
            print(f"     내용: {content}")
    print("=================================================================")


if __name__ == "__main__":
    test_pdf_to_chroma()