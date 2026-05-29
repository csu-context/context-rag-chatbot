import json
from unittest.mock import patch

import pytest

from src.utils.logger import PerformanceLogger, setup_global_logging


def test_setup_global_logging():
    """전역 로깅 설정이 에러 없이 실행되는지 확인"""
    try:
        setup_global_logging()
    except Exception as e:
        pytest.fail(f"setup_global_logging failed with {e}")


def test_performance_logger_singleton():
    """PerformanceLogger가 싱글톤으로 동작하는지 확인"""
    logger1 = PerformanceLogger()
    logger2 = PerformanceLogger()
    assert logger1 is logger2


def test_performance_logger_logging(tmp_path):
    """PerformanceLogger가 파일을 올바르게 생성하고 기록하는지 확인 (JSONL)"""
    # LOGS_DIR를 임시 디렉토리로 모킹
    test_log_file = tmp_path / "performance.jsonl"

    with patch("src.utils.logger.LOGS_DIR", tmp_path):
        logger = PerformanceLogger()
        # 싱글톤이므로 기존 인스턴스의 설정을 강제로 변경
        logger.log_file = test_log_file
        if not test_log_file.exists():
            test_log_file.touch()

        # 변경된 JSONL **kwargs 기반 로깅 방식에 맞춰 키워드 인자로 전달
        logger.log(log_type="TEST_TYPE", duration=1.23, info="Test Info")

        assert test_log_file.exists()
        with open(test_log_file, encoding="utf-8") as f:
            content = f.readlines()
            assert len(content) >= 1  # 최소 데이터 1줄 (더 이상 CSV 헤더 없음)

            # 가장 마지막에 적재된 로그 파싱 및 검증
            last_log = json.loads(content[-1])
            assert last_log["log_type"] == "TEST_TYPE"
            assert last_log["duration"] == 1.23
            assert last_log["info"] == "Test Info"
            assert "timestamp" in last_log
