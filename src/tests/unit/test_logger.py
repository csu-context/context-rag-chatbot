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
    """PerformanceLogger가 파일을 올바르게 생성하고 기록하는지 확인"""
    # LOGS_DIR를 임시 디렉토리로 모킹
    test_log_file = tmp_path / "performance.log"

    with patch("src.utils.logger.LOGS_DIR", tmp_path):
        logger = PerformanceLogger()
        # 싱글톤이므로 기존 인스턴스의 설정을 강제로 변경
        logger.log_file = test_log_file
        if not test_log_file.exists():
            with open(test_log_file, "w", encoding="utf-8") as f:
                f.write("Timestamp,Type,Duration,Info\n")

        logger.log("TEST_TYPE", 1.23, "Test Info")

        assert test_log_file.exists()
        with open(test_log_file, encoding="utf-8") as f:
            content = f.readlines()
            assert len(content) >= 2  # 헤더 + 데이터
            assert "TEST_TYPE,1.23,Test Info" in content[-1]
