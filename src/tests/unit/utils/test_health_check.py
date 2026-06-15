from pathlib import Path
from unittest.mock import MagicMock, patch

from src.utils.health_check import (
    _get_local_files_info,
    check_data_integrity,
    check_database_status,
    check_env,
    check_model_loading,
    run_full_diagnostics,
)


@patch("src.utils.health_check.settings")
@patch("src.utils.health_check.load_dotenv")
def test_check_env_ollama(mock_env, mock_settings):
    mock_settings.MODEL_TYPE = "ollama"
    assert check_env()


@patch("src.utils.health_check.settings")
@patch("src.utils.health_check.load_dotenv")
def test_check_env_gemini_pass(mock_env, mock_settings):
    mock_settings.MODEL_TYPE = "gemini"
    mock_settings.GOOGLE_API_KEY = "valid_key"
    assert check_env()


@patch("src.utils.health_check.settings")
@patch("src.utils.health_check.load_dotenv")
def test_check_env_gemini_fail(mock_env, mock_settings):
    mock_settings.MODEL_TYPE = "gemini"
    mock_settings.GOOGLE_API_KEY = "your_google_api_key_here"
    assert not check_env()


@patch("src.utils.health_check.REQUIRED_DIRECTORIES", [Path("/tmp/nonexistent123")])
@patch("src.utils.health_check.RAW_DATA_DIR", Path("/tmp"))
@patch("src.utils.health_check.ensure_directories")
def test_check_data_integrity(mock_ensure):
    check_data_integrity()
    mock_ensure.assert_called_once()


@patch("src.models.embedder.BGEEmbedder")
@patch("src.utils.health_check.settings")
def test_check_model_loading_success(mock_settings, mock_embedder):
    mock_settings.EMBEDDING_MODEL_NAME = "test-model"
    mock_inst = MagicMock()
    mock_embedder.return_value = mock_inst
    assert check_model_loading()
    mock_inst.encode.assert_called_once_with(["Health check test"])


@patch("src.models.embedder.BGEEmbedder", side_effect=Exception("Failed"))
@patch("src.utils.health_check.settings")
def test_check_model_loading_fail(mock_settings, mock_embedder):
    assert not check_model_loading()


@patch("src.vector_db.chroma_manager.ChromaDBManager")
@patch("src.utils.health_check._get_local_files_info", return_value={})
@patch("src.utils.health_check._get_db_rel_path_map", return_value={})
@patch("src.utils.health_check._analyze_anomalies", return_value={})
def test_check_database_status(m1, m2, m3, mock_db):
    mock_inst = MagicMock()
    mock_inst.get_count.return_value = 10
    mock_db.return_value = mock_inst
    assert check_database_status() == {}


@patch("src.utils.health_check.check_env", return_value=True)
@patch("src.utils.health_check.check_data_integrity")
@patch("src.utils.health_check.check_model_loading", return_value=True)
@patch("src.utils.health_check.check_database_status", return_value={})
def test_run_full_diagnostics_all_pass(m1, m2, m3, m4):
    is_healthy, _ = run_full_diagnostics(silent=True)
    assert is_healthy


@patch("src.utils.health_check.check_env", return_value=False)
@patch("src.utils.health_check.check_data_integrity")
@patch("src.utils.health_check.check_model_loading", return_value=True)
@patch("src.utils.health_check.check_database_status", return_value={})
def test_run_full_diagnostics_fail(m1, m2, m3, m4):
    is_healthy, _ = run_full_diagnostics(silent=True)
    assert not is_healthy


@patch("src.utils.health_check.generate_file_hash", return_value="hash123")
def test_get_local_files_info_includes_hwp(mock_hash, tmp_path):
    """회귀(#198): 로컬 파일 스캔이 .hwp/.hwpx 를 포함해야 한다.

    스캔 확장자에서 HWP가 누락되면 정상 색인된 HWP 청크가 '로컬에 없음'으로 판정되어
    유령 청크로 오탐되고, 자동 복구 단계에서 삭제되어 유효 데이터가 유실될 수 있다.
    """
    (tmp_path / "학칙.hwp").write_bytes(b"dummy")
    (tmp_path / "안내.hwpx").write_bytes(b"dummy")
    (tmp_path / "문서.pdf").write_bytes(b"dummy")

    with patch("src.utils.health_check.RAW_DATA_DIR", tmp_path):
        info = _get_local_files_info()

    names = {v["name"] for v in info.values()}
    assert "학칙.hwp" in names
    assert "안내.hwpx" in names
    assert "문서.pdf" in names
