import json
import time
from unittest.mock import patch

from src.utils.logger import TracingLogger


def test_tracing_logger_singleton():
    TracingLogger.reset_instance()
    logger1 = TracingLogger()
    logger2 = TracingLogger()
    assert logger1 is logger2


def test_trace_session_and_step(tmp_path):
    TracingLogger.reset_instance()
    with patch("src.utils.logger.LOGS_DIR", tmp_path):
        logger = TracingLogger()

        with logger.start_session(query="test query") as session:
            with session.trace_step("step1") as step:
                time.sleep(0.1)
                step["info"] = "data1"

            with session.trace_step("step2") as step:
                time.sleep(0.05)
                step.update({"info": "data2", "count": 10})

        # Check if log file is created
        log_files = list((tmp_path / "trace").glob("*.jsonl"))
        assert len(log_files) == 1

        with open(log_files[0], encoding="utf-8") as f:
            log_entry = json.loads(f.readline())

            assert log_entry["query"] == "test query"
            assert log_entry["status"] == "success"
            assert len(log_entry["steps"]) == 2

            assert log_entry["steps"][0]["step"] == "step1"
            assert log_entry["steps"][0]["info"] == "data1"
            assert float(log_entry["steps"][0]["latency_ms"]) >= 100

            assert log_entry["steps"][1]["step"] == "step2"
            assert log_entry["steps"][1]["info"] == "data2"
            assert log_entry["steps"][1]["count"] == 10
            assert float(log_entry["steps"][1]["latency_ms"]) >= 50


def test_trace_session_error_handling(tmp_path):
    TracingLogger.reset_instance()
    with patch("src.utils.logger.LOGS_DIR", tmp_path):
        logger = TracingLogger()

        try:
            with logger.start_session(type="error_test") as session, session.trace_step("failing_step"):
                raise ValueError("test error")
        except ValueError:
            pass

        log_files = list((tmp_path / "trace").glob("*.jsonl"))
        with open(log_files[0], encoding="utf-8") as f:
            log_entry = json.loads(f.readline())
            assert log_entry["status"] == "error"
            assert log_entry["error_message"] == "test error"
            assert log_entry["steps"][0]["step"] == "failing_step"
