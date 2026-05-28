Looking at the COMPRESSED content provided, I can count the headings carefully. The compressed file shows 9 headings matching the original, but the validator reports 8. I'll output the fixed file with all 9 headings preserved from the original.

# 🚀 RAG 챗봇 리팩토링 종합 백로그 (Epic × Priority 융합)

**7 Epic** × **P0~P3** 결합 최종 백로그.
* **테마(Epic):** 모듈별 분류 → 담당배정/구조파악.
* **우선순위(Priority):** 테마 내 순서 → 스프린트계획.

---

## 📊 요약: 우선순위 분포 매트릭스

| 테마 (Epic) | P0 (긴급/장애) | P1 (품질/핵심) | P2 (성능/UX) | P3 (운영/부채) | 총합 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. 데이터 파이프라인 & 청킹** | 1 | 4 | 1 | 2 | **8** |
| **2. 검색 알고리즘 & 랭킹** | 0 | 3 | 0 | 1 | **4** |
| **3. 캐싱 & 메모리 관리** | 1 | 0 | 6 | 1 | **8** |
| **4. LLM 제어 & 프롬프트 안정성** | 2 | 1 | 2 | 2 | **7** |
| **5. 비동기 처리 & 인프라** | 3 | 2 | 3 | 0 | **8** |
| **6. 모니터링, 평가 & 운영** | 0 | 0 | 0 | 6 | **6** |
| **7. 보안 & 접근 제어** | 1 | 0 | 0 | 1 | **2** |
| **총합** | **8** | **10** | **12** | **13** | **43** |

---

## 📦 1. 데이터 파이프라인 및 청킹 (Data Ingestion & Chunking)
*문서 수집/파싱/청킹/벡터화*

- [x] 🔥 **[P0] Issue 44:** 다중 PDF 병렬 수집 OOM (std::bad_alloc) + 청크 0 오류 (완료 - PDF 존재 시 `max_workers=1` 순차 전환 + 단일 파일 병렬 풀 우회로 완전 해결)
- [x] 🔴 **[P1] Issue 1:** 한국어 청킹 한계 (단순 분할 → 문맥 훼손)
- [x] 🔴 **[P1] Issue 2:** 벡터 의미 희석 (1,500자 청크 → 임베딩 밀도↓)
- [x] 🔴 **[P1] Issue 3:** 계층적 청킹 오류 (자식 검색 후 부모 원문 미조회)냐
- [x] 🔴 **[P1] Issue 10:** 토큰 오염 (PDF 헤더/푸터/대외비 미제거)
- [x] 🟡 **[P2] Issue 13:** 텍스트 파싱 병목 (다중 문서 → 단일 스레드 → 속도↓)
- [x] 🟢 **[P3] Issue 19:** 비효율 인덱싱 (파일 1개 변경도 BM25 전체 재빌드)
- [x] 🟢 **[P3] Issue 20:** 변경 감지 취약 (해시 대신 `mtime` 의존 → 잦은 재색인)

## 🔍 2. 검색 알고리즘 및 랭킹 (Search & Ranking)
*BM25/메타필터/하이브리드(RRF)/점수*

- [x] 🔴 **[P1] Issue 4:** RRF 점수 왜곡 (품질 미달 데이터 순위 부스팅)
- [x] 🔴 **[P1] Issue 9:** BM25 노이즈↑ (불용어 필터 없음 → 인덱스 비대)
- [x] 🔴 **[P1] Issue 40:** BM25 정규화 왜곡 (`Min-Max Scaling` 오류 → 순위 변동)
- [x] 🟢 **[P3] Issue 11:** 메타데이터 필터 없음 (Self-Querying 미구현)

## ⚡ 3. 캐싱, 메모리 및 리소스 관리 (Caching & Memory Management)
*누수 방지/시맨틱-컨텍스트 캐시/브라우저 최적화*

- [x] 🔥 **[P0] Issue 24:** OOM (DB 정합성 검사 시 전체 컬렉션 페이징 없이 로드)
- [x] 🟡 **[P2] Issue 7:** 시맨틱 캐시 비활성화 (대화 이력 있으면 무조건 비활성 하드코딩)
- [x] 🟡 **[P2] Issue 8:** 과도한 캐시 초기화 (파일 1개 변경 → 전체 캐시 초기화)
- [x] 🟡 **[P2] Issue 27:** 브라우저 프리징 (캐시 히트 → 1자씩 스트리밍 → CPU↑)
- [x] 🟡 **[P2] Issue 38:** 캐싱 파괴 (Gemini 사용 시 레거시 메시지 병합 옵션 → 네이티브 캐싱 무력화)
- [x] 🟢 **[P3] Issue 29:** 메모리 누수 (세션에 채팅 이력 무한 누적)
- [x] 🟡 **[P2] Issue 51:** Reranker 20개 문서 전체 배치 추론 → VRAM spike (`model.predict(pairs)` 단일 호출, `batch_size` 제한 없음 → 순간 수백 MiB 급증)
- [x] 🟡 **[P2] Issue 52:** bge-m3 / bge-reranker-v2-m3 fp32 로드 → VRAM 각 ~2.3GB 점유 (fp16 전환 시 각 ~1.15GB로 절감, 합산 ~2.3GB 확보 가능)

## 🤖 4. LLM 제어 및 프롬프트 안정성 (LLM Control & Prompts)
*모델 장애/컨텍스트 한도/프롬프트 보안/오작동 제어*

- [x] 🔥 **[P0] Issue 35:** 프롬프트 인젝션 취약 (검색 문서 단순 결합 → 챗봇 탈취 가능)
- [x] 🔥 **[P0] Issue 41:** 답변 강제 중단 버그 (긴 표 출력 → 반복 에러 오인 → 중단)
- [x] 🔴 **[P1] Issue 46:** 한글 질의/검색 문맥에도 영문 강제 출력 다국어 오작동 (시스템 프롬프트 언어 일관성 결여)
- [x] 🟡 **[P2] Issue 28:** 컨텍스트 한도 방어 없음 (토큰 초과 시 동적 트리밍 미구현)
- [x] 🟡 **[P2] Issue 50:** Ollama keep_alive 만료 → 대화 중 모델 unload 후 재로드 8초 지연 (기본 5분 타임아웃, `OLLAMA_KEEP_ALIVE` 환경변수로 조정 가능)
- [x] 🟢 **[P3] Issue 30:** SPOF (LLM 장애 시 Fallback 체인 없음)
- [x] 🟢 **[P3] Issue 37:** 유지보수↓ (시스템 프롬프트 코드 내 하드코딩 → 페르소나 튜닝 불가)

## ⚙️ 5. 비동기 처리 및 시스템 인프라 (Async & Infrastructure)
*동시성/스레드 블로킹/네트워크 타임아웃/리소스 최적화*

- [ ] 🔥 **[P0] Issue 17:** 영구 CPU 강등 (리랭커 VRAM 부족 → 서킷브레이커 없이 CPU 폴백)
- [ ] 🔥 **[P0] Issue 36:** UI 먹통 (문서 파싱/벡터 변환이 메인 스레드 점유)
- [ ] 🔥 **[P0] Issue 43:** 메인 스레드 블로킹 (Docling 110초 작업 비동기 분리 없음 → 전체 마비)
- [ ] 🔴 **[P1] Issue 47:** Streamlit UI AxiosError 400 Bad Request + 통신 장애 (연결/파라미터 유효성 결함)
- [ ] 🔴 **[P1] Issue 48:** retriever/reranker 최초 로드 지연 (첫 쿼리 Lazy Loading + HuggingFace 온라인 체크 → 30초+ UI 블로킹)
- [ ] 🟡 **[P2] Issue 12:** 검색 속도 병목 (하이브리드 검색 직렬 실행 → 응답↓)
- [ ] 🟡 **[P2] Issue 18:** WebSocket 타임아웃 (Proxy/폐쇄망 + LLM 긴 응답 → 연결 끊김)
- [ ] 🟡 **[P2] Issue 31:** 가짜 타임아웃 (리랭커 취소 불가 → 대기만 함)
- [ ] 🟢 **[P3] Refactor:** `StreamResponder._stage_durations` property 추출 (`stream_responder.py`) — `display_latencies`·`finalize_stream` 두 곳에 동일한 dict comprehension 중복 제거
- [ ] 🟢 **[P3] Refactor:** `citation.py` `normalize_text` 래퍼 제거 + `format_citations` 반환 타입 힌트 추가

## 📊 6. 모니터링, 평가 및 시스템 운영 (Observability, Eval & Ops)
*로깅/에러 은폐 방지/벤치마크/비용 유연성*

- [ ] 🟢 **[P3] Issue 5:** UI 신뢰도 경고 버그 (RRF vs UI 기준치 0.5 불일치 → 무조건 경고)
- [ ] 🟢 **[P3] Issue 26:** WinError 32 충돌 (커스텀 로그 로테이션 → Windows/멀티프로세스 충돌)
- [ ] 🟢 **[P3] Issue 32:** 조용한 실패 (Vector DB 오류 → 에러 없이 빈 리스트 반환)
- [ ] 🟢 **[P3] Issue 39:** Eval 지표 불일치 (벤치마크가 '표 데이터 보존율' 미측정)
- [ ] 🟢 **[P3] Issue 42:** API 비용 유연성↓ (모델 단가 하드코딩 → 신규 모델 과금 누락)
- [ ] 🟢 **[P3] Issue 49:** 로컬 리랭커(CrossEncoder) 로드 Deprecation Warning (`cache_dir` 인자, sentence-transformers/transformers 버전업)

## 🛡️ 7. 보안 및 접근 제어 (Security & Access Control)
*권한 검증/포트 노출 방지*

- [ ] 🔥 **[P0] Issue 34:** 방화벽 취약 (Terraform SSH/UI 포트 `0.0.0.0/0` 전체 개방)
- [ ] 🟢 **[P3] Issue 33:** 보안/권한 결함 (관리자 인증 없음 + 동기화 시 전역 캐시 초기화)