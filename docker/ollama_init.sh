#!/bin/bash

# Ollama 서버를 백그라운드에서 실행
/bin/ollama serve &

# 서버가 준비될 때까지 대기 (ollama list 명령어로 확인)
echo "Waiting for Ollama server to start..."
until ollama list > /dev/null 2>&1; do
    sleep 2
done

# .env 또는 환경 변수에서 모델명 가져오기 (기본값: llama3.2:1b)
MODEL_NAME=${MODEL_NAME:-"llama3.2:1b"}

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
