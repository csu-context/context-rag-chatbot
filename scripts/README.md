# 🛠️ Scripts Directory

이 디렉토리는 프로젝트 운영, 테스트 및 성능 분석을 위한 보조 스크립트들을 포함합니다.

## 📋 주요 스크립트 목록

### 1. 성능 및 품질 분석
*   **`compare_eval.py`**: 두 개의 Ragas 평가 결과(JSON)를 비교하여 품질 저하 여부를 판정합니다.
    *   **용도**: 모델 교체 또는 파이프라인 수정 후 '품질 게이트' 통과 여부 확인.
    *   **실행**: `PYTHONPATH=. python scripts/compare_eval.py <baseline_json> <current_json>`
*   **`compare_rerankers.py`**: 다양한 리랭커 모델(Local, Jina, Cohere 등)의 검색 성능을 비교 분석합니다.
    *   **용도**: 최적의 리랭커 모델 선정 및 가중치 튜닝.
*   **`benchmark_ttft.py`**: 시맨틱 캐시의 동작 여부 및 응답 대기 시간(TTFT)을 측정합니다.
    *   **용도**: Cache Miss, 반복 질문(Cache Hit), 유사 질문(Cache Hit) 시나리오에 대한 성능(TTFT) 비교 및 검증.
    *   **실행**: `PYTHONPATH=. python scripts/benchmark_ttft.py`
*   **`benchmark_parallel_retrieval.py`**: 하이브리드 검색의 BM25 leg와 Vector leg를 직렬·병렬로 각각 실행해 소요 시간과 직렬 대비 단축률(%)을 측정합니다.
    *   **용도**: 검색 병렬화 이득의 정량 검증. 실데이터 우선, 없으면 합성 데이터로 폴백.
    *   **실행**: `PYTHONPATH=. python scripts/benchmark_parallel_retrieval.py [--mode synthetic] [--repeats 10 --n 5]`
*   **`demo_reranker_gpu_recovery.py`**: 리랭커의 GPU 장애 → CPU 폴백 → GPU 복구 라이프사이클을 fault injection으로 재현하고 실제 로그로 캡처합니다.
    *   **용도**: 서킷브레이커 기반 GPU 복구 동작 실증 (실 GPU 없이도 검증 가능).
    *   **실행**: `PYTHONPATH=. python scripts/demo_reranker_gpu_recovery.py [--interval 5]` (미지정 시 운영 기본값 300초)

### 2. 개발 및 테스트 유틸리티
*   **`run_pytest.py`**: 프로젝트의 전체 테스트 케이스를 실행하고 결과를 요약합니다.
    *   **용도**: CI/CD 환경 또는 로컬 개발 단계에서의 통합 테스트 실행.

### 3. 데이터베이스 백업 및 복원
백업 및 복원 로직의 핵심은 `src/vector_db/backup_manager.py`에 있으며, `scripts/` 아래의 파일들은 CLI 진입점입니다.

*   **`backup_db.py`**: ChromaDB 벡터 데이터베이스를 압축하여 백업합니다.
    *   **실행**: `PYTHONPATH=. python scripts/backup_db.py`
*   **`restore_db.py`**: 백업 파일을 사용하여 ChromaDB를 복원합니다.
    *   **최신 백업 복원**: `PYTHONPATH=. python scripts/restore_db.py`
    *   **특정 백업 복원**: `PYTHONPATH=. python scripts/restore_db.py --file chromadb_backup_20260531_030000.tar.gz`

#### 데이터 정합성 검사
백업 관리자는 `tar.gz` 아카이브를 생성하기 전에 다음을 수행합니다.
1. 벡터 DB 디렉터리에 작업 락(lock)을 생성합니다.
2. 벡터 DB 파일 매니페스트가 설정된 대기 시간 동안 안정될 때까지 기다립니다.
3. 아카이브 생성 후 매니페스트를 다시 비교합니다.
4. 백업 중 파일이 변경되면 불완전한 아카이브를 삭제합니다.
5. 아카이브에 대한 `.sha256` 사이드카를 작성합니다.

복원 시에는 `.sha256` 사이드카가 존재하면 확인하고, 아카이브의 압축을 푼 다음, 컬렉션 수와 실제 쿼리 경로를 모두 실행하는 진단 프로그램을 실행합니다.

프로덕션 백업의 경우, 데이터 수집 및 대량 쓰기가 일시 중지된 유지보수 기간 또는 다른 기간에 작업을 예약하십시오. 파일이 계속 변경되면 백업은 잠재적으로 일관성 없는 아카이브를 생성하는 대신 실패합니다.

#### 자동화 예시 (Crontab)
매일 새벽 3시에 백업을 실행하고 `logs/backup_cron.log`에 로그를 남깁니다.
```cron
0 3 * * * cd /path/to/context-rag-chatbot && /path/to/venv/bin/python scripts/backup_db.py >> logs/backup_cron.log 2>&1
```
Linux에서는 `flock`을 사용하여 스케줄러 실행이 겹치지 않도록 할 수 있습니다.
```cron
0 3 * * * cd /path/to/context-rag-chatbot && flock -n /tmp/context-rag-chatbot-backup.lock /path/to/venv/bin/python scripts/backup_db.py >> logs/backup_cron.log 2>&1
```
Windows 작업 스케줄러는 다음을 사용하여 동일한 진입점을 실행할 수 있습니다.
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-Location 'C:\path\to\context-rag-chatbot'; .\.venv\Scripts\python.exe scripts\backup_db.py"
```

### 4. 배포 및 운영
*   **`deploy.sh`**: Docker Compose 빌드·기동 후 헬스체크까지 타임아웃을 적용하여 배포 중 무한 대기를 방지하는 배포 스크립트입니다.
    *   **용도**: 프라이빗 레포 환경에서의 자동 배포 및 컨테이너 정상 기동 확인.
    *   **실행**: `bash scripts/deploy.sh`
    *   **환경변수**: `DEPLOY_TIMEOUT`(기본 120초), `HEALTH_RETRIES`(20회), `HEALTH_INTERVAL`(6초), `COMPOSE_FILE`(docker-compose.yml)

## 🧹 관리 원칙
1.  **일회성 스크립트**: 특정 이슈 해결을 위한 임시 디버깅 스크립트는 작업 완료 후 삭제를 원칙으로 합니다.
2.  **문서화**: 새로운 스크립트 추가 시 본 README에 용도와 실행 방법을 반드시 업데이트합니다.
3.  **경로 독립성**: 모든 스크립트는 프로젝트 루트 디렉토리에서 `PYTHONPATH=.` 설정을 포함하여 실행 가능한 구조를 유지해야 합니다.