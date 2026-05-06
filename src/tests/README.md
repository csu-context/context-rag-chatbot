# 🧪 테스트 가이드 (Test Guide)

이 프로젝트는 `pytest`를 기반으로 한 3계층 테스트 구조를 따릅니다. 새로운 기능을 개발하거나 버그를 수정할 때 아래 가이드에 맞춰 테스트 코드를 추가해 주세요.

## 📁 폴더 구조

### 1. `unit/` (단위 테스트)
- **대상**: 유틸리티 함수, 데이터 스키마, 개별 클래스의 독립적인 메서드.
- **특징**: DB 연결, 네트워크 요청, 모델 로딩이 없어야 하며 매우 빠르게 실행되어야 합니다.
- **예시**: `test_citation.py` (인용구 포맷팅 로직), `test_prompt_structure.py` (프롬프트 템플릿 검증).

### 2. `integration/` (통합 테스트)
- **대상**: DB 매니저, 모델 추상화 레이어, 리트리버 로직.
- **특징**: 실제 ChromaDB나 임베딩 모델(BGE-M3)을 로드하여 모듈 간 상호작용을 검증합니다.
- **예시**: `test_bm25.py` (키워드 검색 엔진), `test_gemini.py` (API 연결 확인).

### 3. `e2e/` (전구간 테스트)
- **대상**: 전체 RAG 파이프라인.
- **특징**: 데이터 입력부터 최종 답변 생성까지 전 과정을 실제 환경과 유사하게 시뮬레이션합니다.
- **예시**: `test_pipeline.py` (전처리 -> DB -> RAG 답변 연동).

## 🚀 테스트 실행 방법

### 전체 테스트 실행
```bash
python -m pytest
```

### 특정 폴더만 실행
```bash
python -m pytest src/tests/unit/
```

## ⚠️ 주의 사항
- 모든 테스트 파일은 `test_*.py` 형식을 따라야 합니다.
- 테스트 함수는 `def test_...`로 시작해야 `pytest`가 자동으로 인식합니다.
- **절대 테스트 코드 내에 `logging.basicConfig`를 넣지 마세요.** (이미 `conftest.py`에서 관리됩니다.)

## 💡 테스트 작성 모범 사례 (Best Practices)

### 1. 외부 API 및 무거운 로직은 Mock 활용
Gemini API 호출이나 복잡한 계산은 `unittest.mock`을 사용하여 속도를 높이고 비용을 절감하세요.
```python
from unittest.mock import patch

@patch("src.models.llm_gemini.GeminiModel.invoke")
def test_logic_with_mock(mock_invoke):
    # 가짜 응답 설정
    mock_invoke.return_value = LLMResponse(content="응답 성공", ...)
    # 테스트 로직 수행...
```

### 2. CI 환경을 배려한 스킵 로직
로컬 모델 로딩(BGE-M3 등)이 필요한 무거운 통합 테스트는 CI 서버의 자원 한계를 고려하여 아래 데코레이터를 추가하세요.
```python
import os
import pytest

@pytest.mark.skipif(os.getenv("CI") == "true", reason="CI 환경에서는 너무 무거워 스킵합니다.")
def test_heavy_model_loading():
    # 모델 로드 로직...
```

### 3. 표준 테스트 템플릿 (Copy & Paste)
새로운 테스트 파일을 만들 때 아래 구조를 복사해서 시작하세요.
```python
import pytest
from src.utils.paths import RAW_DATA_DIR

@pytest.fixture
def sample_fixture():
    """테스트에 필요한 공통 객체 준비"""
    return {"key": "value"}

def test_example_logic(sample_fixture):
    """테스트 설명 작성"""
    assert sample_fixture["key"] == "value"
```
