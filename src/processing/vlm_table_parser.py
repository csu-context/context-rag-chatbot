"""세로병합·멀티라인으로 텍스트 파싱이 깨지는 표를 vision LLM으로 복원하는 모듈 (#166).

`needs_vlm`으로 라우팅(raw extract의 열별 줄 수 불일치)해 해당 표만 이미지로 crop ->
settings.MODEL_NAME(qwen2.5vl 등 vision) -> forward_fill로 복원한다. 실패 시 None을
반환해 호출측이 기존 텍스트 파싱으로 fallback하게 한다.

전제: ollama 서버 + settings.MODEL_NAME이 vision capability를 가진 모델.
"""

import base64
import logging

import fitz
import requests

from src.common.config import settings

logger = logging.getLogger(__name__)

VLM_SCALE = 3.0  # crop 확대 배율(작은 글자 OCR 보강)
VLM_LINE_MISMATCH = 3  # 한 행에서 열별 줄 수 차이가 이 값 이상이면 세로병합/멀티라인 신호

VLM_PROMPT = (
    "이 이미지는 대학의 학위 수여 표다(컬럼: 대학 | 학과(부) | 학위). 표를 마크다운으로 변환하라.\n\n"
    "학위 칸은 세로병합이 많다 — 한 학위가 여러 학과(부)에 걸쳐 한 번만 적혀 있으면 "
    "그 범위의 모든 행이 같은 학위다.\n\n"
    "예시 (세로병합 학위를 각 행에 채운다):\n"
    "[입력 표]\n| A학부 | 공학사 |\n| (a전공) |  |\n| (b전공) |  |\n| B학부 | 문학사 |\n| (c전공) |  |\n"
    "[출력]\n| A학부 | 공학사 |\n| (a전공) | 공학사 |\n| (b전공) | 공학사 |\n"
    "| B학부 | 문학사 |\n| (c전공) | 문학사 |\n\n"
    "규칙:\n"
    "1) 학위 칸이 비어 보이는 행은 바로 위에서 이어진 학위로 채워라. 어떤 행도 비우지 마라. "
    "한 학위가 길게 이어져도 블록 끝까지 빠짐없이 반복하라.\n"
    "2) 괄호로 들여쓴 (OOO전공)은 위 학부(과)의 하위 전공이다.\n"
    "3) 학과(부)와 학위를 같은 행끼리 정확히 대응시켜라(밀리지 않게).\n"
    "4) 설명 없이 마크다운 표만 출력하라."
)


def needs_vlm(table) -> bool:
    """raw extract에서 한 행의 열별 줄 수가 크게 어긋나면(세로병합/멀티라인으로 텍스트
    파싱이 깨지는 신호) VLM 경로가 필요하다고 판단한다."""
    try:
        ext = table.extract()
    except Exception:
        return False
    for row in ext:
        counts = [len((c or "").split("\n")) for c in row]
        if any(c > 1 for c in counts) and (max(counts) - min(counts)) >= VLM_LINE_MISMATCH:
            return True
    return False


def _crop_png(page, bbox) -> bytes:
    pix = page.get_pixmap(clip=fitz.Rect(bbox), matrix=fitz.Matrix(VLM_SCALE, VLM_SCALE))
    return pix.tobytes("png")


def _vlm_markdown(png: bytes, retries: int = 2) -> str | None:
    b64 = base64.b64encode(png).decode("ascii")
    url = settings.OLLAMA_BASE_URL.rstrip("/") + "/api/generate"
    payload = {
        "model": settings.MODEL_NAME,
        "prompt": VLM_PROMPT,
        "images": [b64],
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 4096, "num_predict": 4096},
    }
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=(5.0, 300.0))
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception as e:
            if attempt >= retries:
                logger.warning(f"VLM 호출 실패({retries + 1}회): {e}")
                return None
    return None


def _forward_fill(md: str) -> str:
    """마크다운 마지막 칸(학위) 빈칸을 위 행 값으로 채운다(세로병합 보강). 헤더/구분선 제외."""
    out = []
    last = ""
    seen_header = False
    for line in md.splitlines():
        s = line.strip()
        if not (s.startswith("|") and "---" not in s and s.count("|") >= 3):
            out.append(line)
            continue
        if not seen_header:
            seen_header = True
            out.append(line)
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if cells and cells[-1]:
            last = cells[-1]
        elif cells and last:
            cells[-1] = last
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def parse_table_vlm(page, table) -> str | None:
    """세로병합 표를 VLM으로 파싱해 마크다운을 반환. 실패 시 None(호출측 fallback)."""
    try:
        png = _crop_png(page, table.bbox)
    except Exception as e:
        logger.warning(f"표 crop 실패: {e}")
        return None
    md = _vlm_markdown(png)
    if not md or not md.strip():
        return None
    return _forward_fill(md)
