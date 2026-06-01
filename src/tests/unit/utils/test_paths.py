"""src.utils.paths 의 동의어 사전 경로 외부화/주입 동작 테스트."""

import importlib

import src.utils.paths as paths_mod


def _reload():
    return importlib.reload(paths_mod)


def test_synonyms_file_default_under_data_config(monkeypatch):
    """기본값은 패키지 코드(src/common) 밖 data/config/synonyms.json 을 가리킨다."""
    monkeypatch.delenv("SYNONYMS_PATH", raising=False)
    p = _reload()
    try:
        assert p.SYNONYMS_FILE.parts[-3:] == ("data", "config", "synonyms.json")
        assert "common" not in p.SYNONYMS_FILE.parts
    finally:
        _reload()


def test_synonyms_path_env_override(monkeypatch, tmp_path):
    """SYNONYMS_PATH 환경변수로 외부 볼륨 경로를 주입할 수 있다 (의존성 주입)."""
    custom = tmp_path / "custom_synonyms.json"
    monkeypatch.setenv("SYNONYMS_PATH", str(custom))
    p = _reload()
    try:
        assert custom.resolve() == p.SYNONYMS_FILE
    finally:
        monkeypatch.delenv("SYNONYMS_PATH", raising=False)
        _reload()
