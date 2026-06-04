from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from src.utils.metrics_server import MetricsServer, _MetricsHandler


@pytest.fixture(autouse=True)
def reset_server_state():
    MetricsServer._started = False
    yield
    MetricsServer._started = False


@patch("src.utils.metrics_server.HTTPServer")
@patch("src.utils.metrics_server.threading.Thread")
def test_metrics_server_start(mock_thread, mock_http):
    mock_server = MagicMock()
    mock_http.return_value = mock_server

    MetricsServer.start(9999)
    assert MetricsServer._started is True

    mock_http.assert_called_once()
    mock_thread.assert_called_once()

    # 두 번째 호출은 무시되어야 함
    MetricsServer.start(9999)
    assert mock_http.call_count == 1


@patch("src.utils.metrics_server.HTTPServer", side_effect=OSError("Address in use"))
def test_metrics_server_start_fail(mock_http):
    MetricsServer.start(9999)
    assert MetricsServer._started is False


@patch("src.utils.metrics_server.BaseHTTPRequestHandler.__init__", return_value=None)
def test_metrics_handler(mock_init):
    handler = _MetricsHandler(None, None, None)
    handler.path = "/metrics"
    handler.wfile = BytesIO()
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()

    with patch(
        "src.utils.metrics_server.get_system_stats", return_value={"cpu": 10, "memory": 20, "disk": 30, "gpu_vram": 40}
    ):
        handler.do_GET()

        handler.send_response.assert_called_with(200)
        handler.send_header.assert_called_with("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        val = handler.wfile.getvalue()
        assert b"cpu_pct 10" in val
        assert b"gpu_vram_pct 40" in val

    # 404 test
    handler.path = "/notfound"
    handler.do_GET()
    handler.send_response.assert_called_with(404)

    # dummy log_message test
    handler.log_message("test %s", "msg")
