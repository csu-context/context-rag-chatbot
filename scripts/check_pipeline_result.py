"""파이프라인 통합 최종 결과 확인 스크립트"""

import json
import pickle
from pathlib import Path

from src.vector_db.chroma_manager import ChromaDBManager

print("=" * 55)
print("  파이프라인 통합 최종 결과 요약")
print("=" * 55)

# 1. ChromaDB
db = ChromaDBManager(collection_name="rag_collection")
total = db.get_count()
all_data = db.collection.get(include=["metadatas"])
metas = all_data["metadatas"]
table_chunks = sum(1 for m in metas if m.get("is_table") is True)

print("[ChromaDB]")
print(f"  총 청크 수        : {total}")
print(f"  표 포함 청크      : {table_chunks} ({table_chunks / total * 100:.1f}%)")
print(f"  일반 텍스트 청크  : {total - table_chunks}")

# 2. BM25
bm25 = Path(".cache/bm25_index.pkl")
print("[BM25 인덱스]")
print(f"  캐시 파일 크기    : {bm25.stat().st_size // 1024} KB")

# 3. processed JSON
pjson = [f for f in Path("data/processed").glob("*.json") if f.name != "manifest.json"]
with open(pjson[0], encoding="utf-8") as f:
    chunks = json.load(f)
print("[processed JSON]")
print(f"  파일명            : {pjson[0].name}")
print(f"  parent 청크 수    : {len(chunks)}")
child_count = sum(len(c.get("children", [])) for c in chunks)
print(f"  child 청크 수     : {child_count}")

# 4. Docling 파싱 캐시
docling_cache = list(Path(".cache").glob("*_parsed.pkl"))
if docling_cache:
    with open(docling_cache[0], "rb") as f:
        parsed = pickle.load(f)
    meta = parsed[0]["metadata"]
    print("[Docling 파싱 캐시]")
    print(f"  캐시 파일         : {docling_cache[0].name}")
    print(f"  parser            : {meta.get('parser', 'N/A')}")
    print(f"  table_count       : {meta.get('table_count', 'N/A')}")
    print(f"  page_count        : {meta.get('page_count', 'N/A')}")
    print(f"  has_table         : {meta.get('has_table', 'N/A')}")

# 5. 실시간 검색 테스트
print()
print("[검색 테스트]")
test_query = "졸업 요건"
results = db.search(test_query, k=3)
print(f"  질의: '{test_query}'")
for r in results:
    is_table = r["metadata"].get("is_table", False)
    header = r["metadata"].get("header_path", "N/A")
    score = r["score"]
    preview = r["content"][:50].replace("\n", " ")
    print(f"  [{score:.4f}] is_table={is_table} | {header} | {preview}...")

print()
print("==> 파이프라인 통합 완료!")
