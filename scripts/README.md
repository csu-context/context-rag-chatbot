# 🛠️ Scripts Directory

이 디렉토리는 프로젝트 운영, 테스트 및 성능 분석을 위한 보조 스크립트들을 포함합니다.

## 📋 주요 스크립트 목록

### 1. 성능 및 품질 분석
*   **`compare_eval.py`**: 두 개의 Ragas 평가 결과(JSON)를 비교하여 품질 저하 여부를 판정합니다.
    *   **용도**: 모델 교체 또는 파이프라인 수정 후 '품질 게이트' 통과 여부 확인.
    *   **실행**: `python scripts/compare_eval.py <baseline_json> <current_json>`
*   **`compare_rerankers.py`**: 다양한 리랭커 모델(Local, Jina, Cohere 등)의 검색 성능을 비교 분석합니다.
    *   **용도**: 최적의 리랭커 모델 선정 및 가중치 튜닝.
*   **`benchmark_ttft.py`**: 시맨틱 캐시의 동작 여부 및 응답 대기 시간(TTFT)을 측정합니다.
    *   **용도**: Cache Miss, 반복 질문(Cache Hit), 유사 질문(Cache Hit) 시나리오에 대한 성능(TTFT) 비교 및 검증.
    *   **실행**: `python scripts/benchmark_ttft.py`

### 2. 개발 및 테스트 유틸리티
*   **`run_pytest.py`**: 프로젝트의 전체 테스트 케이스를 실행하고 결과를 요약합니다.
    *   **용도**: CI/CD 환경 또는 로컬 개발 단계에서의 통합 테스트 실행.
*   **`backup_db.py`**: ChromaDB 벡터 데이터베이스를 압축하여 백업합니다.
    *   **용도**: 데이터 보호 및 마이그레이션을 위한 주기적 백업.
    *   **실행**: `python scripts/backup_db.py`
*   **`restore_db.py`**: 백업 파일을 사용하여 ChromaDB를 복원합니다.
    *   **용도**: 데이터 유실 시 복구 또는 특정 시점의 데이터 롤백.
    *   **실행**: `python scripts/restore_db.py [--file <backup_filename>]`

## ⏰ 백업 자동화 (Crontab)
주기적인 백업을 위해 리눅스/유닉스 계열 환경에서는 `crontab`을 사용할 수 있습니다.

### 설정 예시
매일 새벽 3시에 백업을 수행하고 로그를 남기려면 다음과 같이 설정합니다.
```bash
0 3 * * * cd /path/to/project && /path/to/venv/bin/python scripts/backup_db.py >> logs/backup_cron.log 2>&1
```

## 🧹 관리 원칙
1.  **일회성 스크립트**: 특정 이슈 해결을 위한 임시 디버깅 스크립트는 작업 완료 후 삭제를 원칙으로 합니다.
2.  **문서화**: 새로운 스크립트 추가 시 본 README에 용도와 실행 방법을 반드시 업데이트합니다.
3.  **경로 독립성**: 모든 스크립트는 프로젝트 루트 디렉토리에서 `PYTHONPATH=.` 설정을 포함하여 실행 가능한 구조를 유지해야 합니다.
