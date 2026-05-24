#!/bin/bash

# CPU 코어 수 진단하여 환경 변수 할당 (최적화)
if [ -f /proc/cpuinfo ]; then
    CPU_CORES=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || nproc || echo "2")
else
    CPU_CORES=$(nproc 2>/dev/null || echo "2")
fi

echo "Detected CPU cores: $CPU_CORES"

# 동시 처리 요청 및 스레드 수 튜닝
# OLLAMA_NUM_PARALLEL은 동시 질문 처리 수, OLLAMA_NUM_THREADS는 CPU 연산 스레드 수
# PARALLEL은 코어가 여유가 있을 때 코어 수에 비례하여 설정 (최대 4)
PARALLEL_VAL=$(( CPU_CORES > 4 ? 4 : (CPU_CORES > 2 ? 2 : 1) ))
export OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL:-$PARALLEL_VAL}
export OLLAMA_NUM_THREADS=${OLLAMA_NUM_THREADS:-$CPU_CORES}

echo "Setting OLLAMA_NUM_PARALLEL=$OLLAMA_NUM_PARALLEL, OLLAMA_NUM_THREADS=$OLLAMA_NUM_THREADS"

# Ollama 서버를 백그라운드에서 실행
/bin/ollama serve &

# 서버가 준비될 때까지 대기 (ollama list 명령어로 확인)
echo "Waiting for Ollama server to start..."
until ollama list > /dev/null 2>&1; do
    sleep 2
done

# .env 또는 환경 변수에서 모델명 가져오기 (기본값: gemma2:2b)
MODEL_NAME=${MODEL_NAME:-"gemma2:2b"}

# 사용하지 않는 기존 모델 자동 삭제 (디스크 용량 최적화)
echo "Cleaning up outdated models..."
ollama list | tail -n +2 | while read -r name id size modified; do
    if [ -n "$name" ] && [ "$name" != "$MODEL_NAME" ]; then
        echo "Removing outdated model: $name"
        ollama rm "$name"
    fi
done

echo "Checking for model: $MODEL_NAME"

# 현재 설치된 모델 목록 확인 (grep 사용)
if ollama list | grep -q "$MODEL_NAME"; then
    echo "Model $MODEL_NAME already exists."
else
    echo "Model $MODEL_NAME not found. Pulling..."
    ollama pull $MODEL_NAME
    echo "Model $MODEL_NAME installed successfully."
fi

# 포그라운드 프로세스로 유지 (컨테이너 종료 방지)
wait
