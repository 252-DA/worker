import json
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Callable

from document_chunk.shared.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HealthState:
    service: str
    worker_type: str
    ready: bool = True
    metadata: dict[str, str] = field(default_factory=dict)


class HealthServer:
    def __init__(
        self,
        host: str,
        port: int,
        state: HealthState,
        readiness_check: Callable[[], bool] | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._state = state
        self._readiness_check = readiness_check
        self._server: ThreadingHTTPServer | None = None
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._server is not None:
            return

        state = self._state
        readiness_check = self._readiness_check

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                if self.path not in ("/health", "/ready"):
                    self.send_error(404)
                    return

                ready = state.ready and (readiness_check() if readiness_check else True)
                status_code = 200 if self.path == "/health" or ready else 503
                payload = {
                    "status": "ok" if ready else "not_ready",
                    "service": state.service,
                    "worker_type": state.worker_type,
                    **state.metadata,
                }
                encoded = json.dumps(payload).encode("utf-8")

                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, fmt: str, *args) -> None:
                return

        self._server = ThreadingHTTPServer((self._host, self._port), _Handler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info("worker.health.started", host=self._host, port=self._port)

    def close(self) -> None:
        if self._server is None:
            return

        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        logger.info("worker.health.stopped", host=self._host, port=self._port)
        self._server = None
        self._thread = None

    def set_ready(self, ready: bool) -> None:
        self._state.ready = ready


def start_health_server(
    host: str,
    port: int,
    state: HealthState,
    readiness_check: Callable[[], bool] | None = None,
) -> HealthServer:
    server = HealthServer(host=host, port=port, state=state, readiness_check=readiness_check)
    server.start()
    return server
