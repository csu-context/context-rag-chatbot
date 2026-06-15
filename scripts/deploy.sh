#!/usr/bin/env bash
# 프라이빗 레포 배포 스크립트 — 무한 대기 방지 (timeout + 헬스체크)
set -euo pipefail

TIMEOUT_SECONDS=${DEPLOY_TIMEOUT:-120}
HEALTH_RETRIES=${HEALTH_RETRIES:-20}
HEALTH_INTERVAL=${HEALTH_INTERVAL:-6}
COMPOSE_FILE=${COMPOSE_FILE:-docker-compose.yml}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# 1. 이미지 빌드 (타임아웃 적용)
log "Building image..."
timeout "$TIMEOUT_SECONDS" docker compose -f "$COMPOSE_FILE" build --no-cache || {
    echo "ERROR: docker compose build timed out (${TIMEOUT_SECONDS}s)" >&2
    exit 1
}

# 2. 호스트 바인드 마운트 디렉토리 소유권 보장(비루트 컨테이너 쓰기 권한)
# root 소유 디렉토리가 있으면 chown 에 sudo 가 필요할 수 있다(스크립트는 실패 시 경고만 남김).
log "Ensuring host bind-mount dirs are writable..."
bash scripts/init-host-dirs.sh

# 3. 컨테이너 기동
log "Starting containers..."
timeout "$TIMEOUT_SECONDS" docker compose -f "$COMPOSE_FILE" up -d || {
    echo "ERROR: docker compose up timed out" >&2
    exit 1
}

# 4. 헬스체크 대기 (최대 HEALTH_RETRIES * HEALTH_INTERVAL 초)
log "Waiting for app health check..."
for i in $(seq 1 "$HEALTH_RETRIES"); do
    STATUS=$(docker inspect --format='{{.State.Health.Status}}' rag-chatbot-app 2>/dev/null || echo "missing")
    if [ "$STATUS" = "healthy" ]; then
        log "App is healthy after $((i * HEALTH_INTERVAL))s"
        exit 0
    fi
    if [ "$STATUS" = "unhealthy" ]; then
        echo "ERROR: App became unhealthy. Logs:" >&2
        docker compose -f "$COMPOSE_FILE" logs --tail=50 app >&2
        exit 1
    fi
    log "Health: $STATUS (attempt $i/$HEALTH_RETRIES)..."
    sleep "$HEALTH_INTERVAL"
done

echo "ERROR: Health check timed out after $((HEALTH_RETRIES * HEALTH_INTERVAL))s" >&2
docker compose -f "$COMPOSE_FILE" logs --tail=50 app >&2
exit 1