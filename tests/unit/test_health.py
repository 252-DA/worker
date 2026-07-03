"""
Tests for worker/health.py — HealthState and HealthServer.
"""
import json
import socket
from unittest.mock import MagicMock

import pytest

from worker.health import HealthServer, HealthState, start_health_server


class TestHealthState:
    def test_defaults(self):
        state = HealthState(service="test-svc", worker_type="document")
        assert state.service == "test-svc"
        assert state.worker_type == "document"
        assert state.ready is True
        assert state.metadata == {}

    def test_ready_can_be_toggled(self):
        state = HealthState(service="svc", worker_type="outbox")
        state.ready = False
        assert state.ready is False

    def test_metadata(self):
        state = HealthState(
            service="svc",
            worker_type="enrichment",
            metadata={"version": "1.0.0"},
        )
        assert state.metadata["version"] == "1.0.0"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestHealthServer:
    def test_start_and_stop(self):
        port = _find_free_port()
        state = HealthState(service="test", worker_type="document")
        server = HealthServer(host="127.0.0.1", port=port, state=state)
        server.start()
        assert server._server is not None
        assert server._thread is not None
        server.close()
        assert server._server is None
        assert server._thread is None

    def test_start_idempotent(self):
        port = _find_free_port()
        state = HealthState(service="test", worker_type="document")
        server = HealthServer(host="127.0.0.1", port=port, state=state)
        server.start()
        first_server = server._server
        server.start()  # should be a no-op
        assert server._server is first_server
        server.close()

    def test_close_idempotent(self):
        port = _find_free_port()
        state = HealthState(service="test", worker_type="document")
        server = HealthServer(host="127.0.0.1", port=port, state=state)
        server.start()
        server.close()
        server.close()  # should not raise
        assert server._server is None

    def test_set_ready(self):
        port = _find_free_port()
        state = HealthState(service="test", worker_type="document")
        server = HealthServer(host="127.0.0.1", port=port, state=state)
        server.start()
        assert state.ready is True
        server.set_ready(False)
        assert state.ready is False
        server.set_ready(True)
        assert state.ready is True
        server.close()


class TestStartHealthServer:
    def test_returns_health_server_instance(self):
        port = _find_free_port()
        state = HealthState(service="test", worker_type="document")
        server = start_health_server(host="127.0.0.1", port=port, state=state)
        assert isinstance(server, HealthServer)
        server.close()
