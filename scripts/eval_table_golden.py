"""원본 기준 표 골든셋으로 현재 파서를 셀 단위 채점하는 통합 러너.

각 골든셋(data/eval/golden/*.json)의 ground_truth_rows를 (조회 키, 정답값)으로 펼쳐,
해당 페이지를 현재 파서로 파싱한 결과에서 "키가 모두 포함된 행에 정답이 등장하는가"를 확인한다.
RAG를 거치지 않으므로 파싱 신호만 직접 측정한다(검색·생성 노이즈와 Faithfulness 함정 회피).

정답은 사람이 원본 PDF를 보고 만든 것이라 파서를 바꿔도 골든셋이 고정된다 → 파싱 전략 객관 비교 가능.

실행: PYTHONIOENCODING=utf-8 python scripts/eval_table_golden.py
"""

import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
import fitz

from src.processing.pdf_parser import DoclingPDFParser

GOLDEN_DIR = "data/eval/golden"
PDF_DIR = "data/raw"


def parse_table_rows(pdf: str, page: int) -> list[list[str]]:
    """현재 파서로 페이지의 첫 표를 파싱해 마크다운 행(셀 리스트)으로 반환."""
    md = DoclingPDFParser._pymupdf_table_markdown(fitz.open(f"{PDF_DIR}/{pdf}"), page, 0, "legal")
    rows = []
    for line in md.splitlines():
        if line.startswith("|") and "---" not in line:
            rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows


def _norm(s: str) -> str:
    return re.sub(r"[\s()\-·‧,]", "", s or "")


def to_lookups(golden_name: str, row: dict) -> list[tuple[list[str], str]]:
    """골든셋 row를 표별 스키마에 맞춰 [(조회 키 목록, 정답값)] 으로 변환."""
    if golden_name.startswith("degree"):
        return [([row["dept"]], row["degree"])]
    if golden_name.startswith("change"):
        return [([row["dept"]], row["after_college"])]
    if golden_name.startswith("period"):
        return [([row["category"], row["target"]], row["days"])]
    if golden_name.startswith("credit"):
        key = row["note"] or row["grad_credit"]
        # 학년별 값을 펼친다(열 위치 검증은 약식 — 키 행 내 정답 등장으로 판정)
        return [([key], v) for v in row["by_year"].values()]
    return []


def lookup(rows: list[list[str]], keys: list[str], value: str) -> bool:
    """키가 모두 포함된 행을 찾아 그 행에 정답값이 등장하는지."""
    nkeys = [_norm(k) for k in keys if k]
    nval = _norm(value)
    for cells in rows:
        joined = _norm("".join(cells))
        if nkeys and all(k in joined for k in nkeys):
            return bool(nval) and nval in joined
    return False


def score_credit(rows: list[list[str]], gt_rows: list[dict]) -> tuple[int, int, list]:
    """숫자 매트릭스 표: 헤더의 학년 열 인덱스를 찾아 셀 단위로 정밀 채점(열 위치 검증)."""
    year_col: dict[str, int] = {}
    for cells in rows:
        for i, cell in enumerate(cells):
            s = cell.strip()
            if re.fullmatch(r"\d학년", s):
                year_col.setdefault(s, i)
    c = t = 0
    misses = []
    for row in gt_rows:
        key = _norm(row["note"] or row["grad_credit"])
        target = next((cells for cells in rows if key and key in _norm("".join(cells))), None)
        for yr, val in row["by_year"].items():
            t += 1
            ci = year_col.get(yr)
            got = target[ci] if (target and ci is not None and ci < len(target)) else None
            if got is not None and _norm(val) in _norm(got):
                c += 1
            else:
                misses.append(([key, yr], f"정답 '{val}' / 파싱칸 '{got}'"))
    return c, t, misses


def main() -> None:
    grand_c = grand_t = 0
    print("=== 표 파싱 골든셋 셀 채점 (현재 파서, RAG 우회) ===")
    for path in sorted(glob.glob(f"{GOLDEN_DIR}/*.json")):
        name = os.path.basename(path)
        gold = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = parse_table_rows(gold["meta"]["source"], gold["meta"]["page"])
        if name.startswith("credit"):
            c, t, misses = score_credit(rows, gold["ground_truth_rows"])
        else:
            c = t = 0
            misses = []
            for row in gold["ground_truth_rows"]:
                for keys, val in to_lookups(name, row):
                    t += 1
                    if lookup(rows, keys, val):
                        c += 1
                    else:
                        misses.append((keys, val))
        grand_c += c
        grand_t += t
        print(f"  {name:26} {c:3}/{t:3} = {c / t * 100:5.1f}%")
        for keys, val in misses[:5]:
            print(f"        MISS {keys} -> 정답 '{val}'")
    print(f"  {'합계':26} {grand_c:3}/{grand_t:3} = {grand_c / grand_t * 100:5.1f}%")


if __name__ == "__main__":
    main()
