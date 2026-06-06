from pathlib import Path
from unittest.mock import MagicMock, patch

from src.utils.health_check import (
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
