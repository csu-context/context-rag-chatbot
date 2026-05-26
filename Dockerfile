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

# ARG를 통해 CPU / GPU 모드 선택 (기본값은 cpu)
ARG DEVICE_TYPE=cpu

# 의존성 파일 복사 및 설치 (CPU 및 GPU 맞춤 설정 사용, 캐시 마운트로 초고속화)
COPY requirements-prod.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$DEVICE_TYPE" = "gpu" ]; then \
        sed -i 's/+cpu//g' requirements-prod.txt && \
        sed -i 's|whl/cpu|whl/cu130|g' requirements-prod.txt && \
        pip install -r requirements-prod.txt; \
    else \
        pip install -r requirements-prod.txt; \
    fi


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

# 소스 코드 및 관련 디렉토리 구조 생성
RUN mkdir -p data/raw data/processed vector_db logs

# 소스 코드 복사
COPY src/ /app/src/

# 포트 설정
EXPOSE 8501

# 컨테이너 실행 명령
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0"]
