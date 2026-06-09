#!/usr/bin/env bash
# 바인드 마운트 호스트 디렉토리를 비루트 컨테이너(appuser, uid 1000)가 쓸 수 있도록 보장한다.
#
# 배경: docker-compose 의 ./logs, ./data 등은 바인드 마운트라 이미지 빌드 시점의
# chown(Dockerfile) 이 런타임에 무효화된다. 또한 호스트에 소스 디렉토리가 없으면
# `docker compose up` 시 Docker 데몬(root)이 root:root 로 자동 생성해 버려,
# uid 1000 으로 도는 컨테이너가 파일을 쓰지 못하고 PermissionError 가 발생한다.
#
# 이 스크립트는 쓰기 대상 디렉토리를 미리 만들고 APP_UID:APP_GID 로 chown 하여
# 위 두 경우(없어서 root 자동생성 / 이전 root 실행의 잔여 파일)를 모두 해소한다.
# 멱등하므로 `up` 전에 매번 호출해도 안전하다.
set -euo pipefail

# Dockerfile 의 appuser 와 동일(useradd -u 1000). 환경변수로 오버라이드 가능.
APP_UID="${APP_UID:-1000}"
APP_GID="${APP_GID:-1000}"

# 스크립트 위치 기준 프로젝트 루트로 이동(어디서 호출해도 동작).
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 컨테이너가 런타임에 쓰는 바인드 마운트 디렉토리 목록.
WRITABLE_DIRS=(
    logs
    logs/trace
    data/processed
    data/backups
    vector_db
    .cache
    models
)

echo "[init-host-dirs] Ensuring writable bind-mount dirs are owned by ${APP_UID}:${APP_GID}..."
for dir in "${WRITABLE_DIRS[@]}"; do
    mkdir -p "$dir"
    if chown -R "${APP_UID}:${APP_GID}" "$dir" 2>/dev/null; then
        echo "  ok: $dir"
    else
        # 비루트로 실행돼 chown 이 막히면 경고만 남기고 진행(이미 소유권이 맞는 경우가 대부분).
        echo "  warn: chown 실패(권한 부족) — sudo 로 재실행하거나 소유권을 확인하세요: $dir" >&2
    fi
done
echo "[init-host-dirs] Done."
