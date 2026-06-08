import bcrypt

from src.utils.auth import verify_password


class TestVerifyPassword:
    def test_correct_password(self):
        hashed = bcrypt.hashpw(b"s3cret", bcrypt.gensalt()).decode("utf-8")
        assert verify_password("s3cret", hashed) is True

    def test_wrong_password(self):
        hashed = bcrypt.hashpw(b"s3cret", bcrypt.gensalt()).decode("utf-8")
        assert verify_password("nope", hashed) is False

    def test_none_hash_denies(self):
        """게이트 미설정(None)이면 어떤 입력도 거부(fail-closed)."""
        assert verify_password("anything", None) is False

    def test_empty_plain_denies(self):
        hashed = bcrypt.hashpw(b"s3cret", bcrypt.gensalt()).decode("utf-8")
        assert verify_password("", hashed) is False

    def test_malformed_hash_denies(self):
        """잘못된 해시 형식이어도 예외 없이 거부(fail-closed)."""
        assert verify_password("s3cret", "not-a-bcrypt-hash") is False
