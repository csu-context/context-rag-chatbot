# Python 3.13 슬림 이미지를 베이스로 사용
FROM python:3.13-slim

# 시스템 의존성 설치 (빌드 및 런타임에 필요한 최소 패키지)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt .

# PyTorch 및 기타 라이브러리 설치 (CUDA 13.0 인덱스 반영)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 소스 코드 및 관련 디렉토리는 docker-compose의 볼륨을 통해 연결하므로
# 빌드 시점에는 최소한의 구조만 생성
RUN mkdir -p data/raw data/processed vector_db logs

# 포트 설정 (Streamlit 기본 포트)
EXPOSE 8501

# 컨테이너 실행 시 기본 명령 (Streamlit 실행)
CMD ["streamlit", "run", "src/main_ui.py", "--server.address=0.0.0.0"]
