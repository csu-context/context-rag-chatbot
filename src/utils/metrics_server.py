"""외부 수집용 Metrics HTTP 엔드포인트 (백그라운드 스레드 실행).

Prometheus / Grafana Agent 등이 http://localhost:9090/metrics 를 scrape합니다.
사용법: MetricsServer.start() 를 app.py 시작 시 1회 호출.
"""

import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

from src.utils.monitoring import get_system_stats

logger = logging.getLogger(__name__)

_METRICS_PORT = 9090


class _MetricsHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # suppress access logs
        pass

    def do_GET(self):
        if self.path in ("/metrics", "/health"):
            stats = get_system_stats()
            lines = [
                "# HELP cpu_pct CPU usage percentage",
                "# TYPE cpu_pct gauge",
                f"cpu_pct {stats['cpu']}",
                "# HELP memory_pct Memory usage percentage",
                "# TYPE memory_pct gauge",
                f"memory_pct {stats['memory']}",
                "# HELP disk_pct Disk usage percentage",
                "# TYPE disk_pct gauge",
                f"disk_pct {stats['disk']}",
            ]
            if stats.get("gpu_vram") is not None:
                lines += [
                    "# HELP gpu_vram_pct GPU VRAM usage percentage",
                    "# TYPE gpu_vram_pct gauge",
                    f"gpu_vram_pct {stats['gpu_vram']}",
                ]
            body = ("\n".join(lines) + "\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


class MetricsServer:
    _started: ClassVar[bool] = False
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def start(cls, port: int = _METRICS_PORT) -> None:
        with cls._lock:
            if cls._started:
                return

            try:
                server = HTTPServer(("0.0.0.0", port), _MetricsHandler)
            except Exception as e:
                logger.warning(f"Metrics 서버 시작 실패: {e}")
                return

            cls._started = True

        def _run():
            logger.info(f"Metrics 서버 시작: http://0.0.0.0:{port}/metrics")
            server.serve_forever()

        threading.Thread(target=_run, daemon=True, name="metrics-server").start()
