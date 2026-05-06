import os
import sys
from pathlib import Path

import pytest

# 프로젝트 루트를 sys.path에 추가하여 src 모듈 임포트 보장
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """테스트 실행 전 환경 변수 및 설정을 초기화합니다."""
    from src.utils.logger import setup_global_logging

    # 전역 로깅 설정 (테스트 로그 노이즈 제거)
    setup_global_logging()

    # 테스트용 환경 변수 설정
    os.environ["TESTING"] = "true"
    yield
    # 정리 로직 (필요 시)
