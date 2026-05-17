"""Docling 파싱 단독 테스트 — 결과를 마크다운 파일로 저장"""

import json
import time
from pathlib import Path

from src.processing.pdf_parser import DoclingPDFParser

PDF_PATH = Path("data/raw/조선대학교_학칙.pdf")
OUT_MD = Path("reports/docling_parse_result.md")
OUT_JSON = Path("reports/docling_parse_meta.json")

print("=" * 50)
print("  Docling 파싱 단독 테스트")
print("=" * 50)
print(f"  입력: {PDF_PATH}")
print()

parser = DoclingPDFParser()

start = time.time()
result = parser.parse(PDF_PATH)
elapsed = time.time() - start

page_count = result["page_count"]
table_count = result["table_count"]
md_text = result["markdown"]

print(f"  파싱 시간  : {elapsed:.1f}초")
print(f"  페이지     : {page_count}")
print(f"  표 감지    : {table_count}개")
print(f"  마크다운   : {len(md_text)}자")
print()

# 1) 마크다운 전문 저장
OUT_MD.write_text(md_text, encoding="utf-8")
print(f"  [저장] {OUT_MD}  ({OUT_MD.stat().st_size // 1024} KB)")

# 2) 메타데이터 JSON 저장
meta = {
    "file": PDF_PATH.name,
    "parser": "docling",
    "parse_time_sec": round(elapsed, 2),
    "page_count": page_count,
    "table_count": table_count,
    "markdown_length": len(md_text),
    "korean_chars": sum(1 for c in md_text if "\uac00" <= c <= "\ud7a3"),
}
OUT_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"  [저장] {OUT_JSON}")

print()
print("  === 마크다운 미리보기 (앞 600자) ===")
print(md_text[:600])
print("  ...")
print()
print("  완료!")
