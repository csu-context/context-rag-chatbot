# --- Stage 1: Builder ---
FROM python:3.13-slim AS builder

# 빌드에 필요한 시스템 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /build

# 가상환경 생성 및 경로 설정
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 의존성 파일 복사 및 설치 (GPU 맞춤 설정 사용, 캐시 마운트로 초고속화)
COPY requirements-prod.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-prod.txt


# --- Stage 2: Final Runtime ---
FROM python:3.13-slim

# 런타임에 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# builder 스테이지에서 생성된 가상환경 통째로 복사
COPY --from=builder /opt/venv /opt/venv

# 환경 변수 설정: 가상환경을 기본 파이썬으로 사용
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 비루트 실행 사용자 생성 (컨테이너 탈출 시 호스트 권한 획득 경로 차단)
RUN groupadd -r appgroup && useradd -r -g appgroup -u 1000 appuser

# 소스 코드 및 관련 디렉토리 구조 생성 (appuser 소유)
RUN mkdir -p data/raw data/processed vector_db logs && \
    chown -R appuser:appgroup /app

# rapidocr(docling OCR 엔진)가 쓰는 OCR 모델(.onnx, det/cls/rec)을 빌드 시 미리 받아
# 이미지 레이어에 포함한다. 미포함 시 런타임에 site-packages/rapidocr/models 로 받는데,
# 이는 이미지가 아니라 컨테이너 쓰기 레이어라 --force-recreate 마다 유실되어 매번
# 재다운로드(네트워크 의존)된다. 빌드 시 baking 으로 폐쇄망/오프라인 recreate 에도 견고.
RUN python -c "from rapidocr import RapidOCR; RapidOCR()"

# rapidocr 패키지 디렉토리를 appuser 소유로 넘긴다. 빌드 시 받은 모델 읽기 +
# 런타임 잔여 다운로드(다른 모델 변형 등) 시 쓰기를 허용한다(비루트 PermissionError 방지).
RUN chown -R appuser:appgroup /opt/venv/lib/python3.13/site-packages/rapidocr

# 소스 코드 복사 (appuser 소유)
COPY --chown=appuser:appgroup src/ /app/src/
COPY --chown=appuser:appgroup prompts/ /app/prompts/
COPY --chown=appuser:appgroup config/ /app/config/
# Streamlit 설정(.streamlit/config.toml) 복사 — 미포함 시 컨테이너가 기본값으로만 동작하여
# 업로드/WebSocket 방어 설정이 무효화됨(프로덕션 이미지에서도 적용되도록 보장)
COPY --chown=appuser:appgroup .streamlit/ /app/.streamlit/

# 포트 설정
EXPOSE 8501
EXPOSE 9090

# 비루트 사용자로 전환
USER appuser

# 컨테이너 실행 명령
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0"]
