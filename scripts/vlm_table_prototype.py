"""표 영역을 이미지로 crop해 gemma4:e4b(vision)으로 마크다운 복원하는 #166 프로토타입.

텍스트 파싱(`_pymupdf_table_markdown`)이 세로병합 표(p54 등)에서 학위를 오염시키는 문제를,
같은 표를 이미지로 VLM에 입력하면 시각 구조로 복원 가능한지 검증하는 1차 프로토타입이다.
golden 셀채점과 연결하기 전, crop 품질과 VLM 출력 형태를 먼저 눈으로 확인하는 단계.

실행: PYTHONIOENCODING=utf-8 python scripts/vlm_table_prototype.py
전제: ollama 서버 + gemma4:e4b(vision capability) 구동.
"""

import base64
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import fitz
import requests

from src.common.config import settings

PDF = "data/raw/조선대학교_학칙.pdf"
OUT_DIR = "tmp/vlm_out"
SCALE = 3.0  # 확대 배율 — OCR 정확도 우선(OLLAMA_NUM_PARALLEL=1로 메모리 확보 전제)
VLM_MODEL = "qwen2.5vl:7b"  # vision OCR 전용 — 7B(3B는 runner segfault). 메모리는 컨테이너 limit 상향으로 확보

VLM_PROMPT = (
    "이 이미지는 대학의 학위 수여 표다(컬럼: 대학 | 학과(부) | 학위). 표를 마크다운으로 변환하라.\n\n"
    "학위 칸은 세로병합이 많다 — 한 학위가 여러 학과(부)에 걸쳐 한 번만 적혀 있으면 "
    "그 범위의 모든 행이 같은 학위다.\n\n"
    "예시 (세로병합 학위를 각 행에 채운다):\n"
    "[입력 표]\n"
    "| A학부 | 공학사 |\n"
    "| (a전공) |  |\n"
    "| (b전공) |  |\n"
    "| B학부 | 문학사 |\n"
    "| (c전공) |  |\n"
    "[출력]\n"
    "| A학부 | 공학사 |\n"
    "| (a전공) | 공학사 |\n"
    "| (b전공) | 공학사 |\n"
    "| B학부 | 문학사 |\n"
    "| (c전공) | 문학사 |\n\n"
    "규칙:\n"
    "1) 위 예시처럼 학위 칸이 비어 보이는 행은 바로 위에서 이어진 학위로 채워라. 어떤 행도 학위를 비우지 마라.\n"
    "2) 괄호로 들여쓴 (OOO전공)은 위 학부(과)의 하위 전공이다.\n"
    "3) 학과(부)와 학위를 같은 행끼리 정확히 대응시켜라(위/아래로 밀리지 않게).\n"
    "4) 대학명도 세로병합이면 모든 행에 채워라.\n"
    "5) 설명 없이 마크다운 표만 출력하라."
)


def crop_table_png(page_no: int, table_idx: int = 0) -> bytes:
    """페이지의 표 영역(bbox)을 잘라 PNG 바이트로 반환(SCALE배 확대)."""
    doc = fitz.open(PDF)
    page = doc[page_no]
    tables = sorted(page.find_tables().tables, key=lambda t: (round(t.bbox[1], 1), round(t.bbox[0], 1)))
    bbox = fitz.Rect(tables[table_idx].bbox)
    pix = page.get_pixmap(clip=bbox, matrix=fitz.Matrix(SCALE, SCALE))
    return pix.tobytes("png")


def vlm_markdown(png: bytes, retries: int = 2) -> str:
    """qwen2.5vl(vision)으로 표 이미지를 마크다운으로 변환. 500 등 일시 오류는 재시도."""
    b64 = base64.b64encode(png).decode("ascii")
    url = settings.OLLAMA_BASE_URL.rstrip("/") + "/api/generate"
    payload = {
        "model": VLM_MODEL,
        "prompt": VLM_PROMPT,
        "images": [b64],
        "stream": False,
        # OLLAMA_NUM_PARALLEL=1 전제: KvSize=num_ctx x 1 이라 num_ctx 4096도 여유.
        # (parallel 4였을 땐 num_ctx x4로 KV/compute graph가 RAM 8GB를 넘겨 500났다.)
        "options": {"temperature": 0.0, "num_ctx": 4096, "num_predict": 4096},
    }
    last = ""
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=(5.0, 300.0))
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception as e:
            last = str(e)
            if attempt < retries:
                time.sleep(3.0)
    return f"[VLM 실패 {retries + 1}회] {last}"


def main() -> None:
    # (0-index 페이지, 라벨) — 텍스트 파싱이 깨지거나(p54/p36) 정상(p51)인 대조군
    targets = [
        (54, "p54 학위표 (세로병합 -> 텍스트파싱은 컨설팅학사 오염)"),
        (51, "p51 학위표 (외국어대학 -> 텍스트파싱 정상, 대조군)"),
        (36, "p36 변경표 (2단 헤더)"),
    ]
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) crop 이미지를 먼저 모두 저장(육안 확인용) — VLM이 실패해도 crop은 남는다
    crops = {}
    for page_no, label in targets:
        png = crop_table_png(page_no)
        path = f"{OUT_DIR}/p{page_no}.png"
        with open(path, "wb") as f:
            f.write(png)
        crops[page_no] = png
        print(f"[crop] {path}  ({len(png)} bytes)  {label}")

    # 2) VLM 변환
    for page_no, label in targets:
        print("=" * 90)
        print(label)
        try:
            md = vlm_markdown(crops[page_no])
        except Exception as e:
            md = f"[VLM 실패] {e}"
        print(md)


if __name__ == "__main__":
    main()
