from src.common.config import Settings


class TestAuthGateConfig:
    def test_auth_passwords_default_none(self):
        """APP_PASSWORD/ADMIN_PASSWORD 기본값은 None이어야 한다.

        opt-in 게이트 — 미설정이면 인증 없음. 기본값에 비밀번호가 박히면
        (게이트 항상 ON 또는 하드코딩 비번) 보안 회귀이므로 가드한다.
        """
        fields = Settings.model_fields
        assert fields["APP_PASSWORD"].default is None
        assert fields["ADMIN_PASSWORD"].default is None
