import json
from unittest.mock import patch

from src.utils.logger import TracingLogger


def test_tracing_logger_masking(tmp_path):
    TracingLogger.reset_instance()
    with patch("src.utils.logger.LOGS_DIR", tmp_path):
        logger = TracingLogger()

        test_data = {
            "query": "My email is test@example.com and phone is 010-1234-5678.",
            "rrn": "Resident number: 901101-1234567",
            "openai_key": "sk-abcdefghijklmnopqrstuvwxyz1234567890",
            "gemini_key": "AIzaSyAz1234567890123456789012345678901",
            "anthropic_key": "sk-ant-sid01-abcdefghijklmnopqrstuvwxyz1234567890abcdefghijkl",
            "auth_token": "bearer some-sensitive-token",
            "safe_field": "This is a safe field.",
            "embedded_vals": (
                "Here is a key: sk-abcdefghijklmnopqrstuvwxyz1234567890. "
                "And another: AIzaSyAz1234567890123456789012345678901"
            )
        }

        with logger.start_session(**test_data) as session, session.trace_step("step1") as step:
            step["info"] = "Contact me at 02-987-6543 or test2@gmail.com."

        log_files = list((tmp_path / "trace").glob("*.jsonl"))
        assert len(log_files) == 1

        with open(log_files[0], encoding="utf-8") as f:
            log_entry = json.loads(f.readline())

            # 1. 필드 기반 마스킹 (key/token/secret/auth 등이 포함된 키값 검증)
            assert log_entry["openai_key"] == "********"
            assert log_entry["gemini_key"] == "********"
            assert log_entry["anthropic_key"] == "********"
            assert log_entry["auth_token"] == "********"

            # 2. 콘텐츠/정규표현식 기반 문자열 내 민감 정보 마스킹 검증
            assert "[EMAIL_MASKED]" in log_entry["query"]
            assert "test@example.com" not in log_entry["query"]
            assert "[PHONE_MASKED]" in log_entry["query"]
            assert "010-1234-5678" not in log_entry["query"]

            assert "[RRN_MASKED]" in log_entry["rrn"]
            assert "901101-1234567" not in log_entry["rrn"]

            # 문자열 내부의 API 키 마스킹 검증
            embedded = log_entry["embedded_vals"]
            assert "[API_KEY_MASKED]" in embedded
            assert "sk-abcdefghijklmnopqrstuvwxyz1234567890" not in embedded
            assert "AIzaSyAz1234567890123456789012345678901" not in embedded

            # 하위 단계(step) 내 정보 마스킹 검증
            step_info = log_entry["steps"][0]["info"]
            assert "[PHONE_MASKED]" in step_info
            assert "02-987-6543" not in step_info
            assert "[EMAIL_MASKED]" in step_info
            assert "test2@gmail.com" not in step_info

            # 안전한 필드가 훼손되지 않고 보존되는지 검증
            assert log_entry["safe_field"] == "This is a safe field."
