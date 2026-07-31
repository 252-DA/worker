"""
Tests for worker/config.py — WorkerSettings hierarchy.
"""
import os
from unittest import mock

import pytest

from worker.config import (
    HealthConfig,
    LearningContextMcpConfig,
    WorkerRuntimeConfig,
    WorkerSettings,
    get_worker_settings,
)


class TestWorkerRuntimeConfig:
    def test_defaults(self):
        cfg = WorkerRuntimeConfig()
        assert cfg.type == "document"
        assert cfg.concurrency == 3
        assert cfg.shutdown_grace_seconds == 30.0
        assert cfg.service_name == "document-worker"

    def test_auto_service_name_enrichment(self):
        cfg = WorkerRuntimeConfig(type="enrichment")
        assert cfg.service_name == "enrichment-worker"

    def test_auto_service_name_outbox(self):
        cfg = WorkerRuntimeConfig(type="outbox")
        assert cfg.service_name == "outbox-worker"

    def test_explicit_service_name(self):
        cfg = WorkerRuntimeConfig(type="document", service_name="my-custom-worker")
        assert cfg.service_name == "my-custom-worker"

    def test_explicit_service_name_overrides_auto(self):
        cfg = WorkerRuntimeConfig(type="outbox", service_name="custom-outbox")
        assert cfg.service_name == "custom-outbox"


class TestHealthConfig:
    def test_defaults(self):
        cfg = HealthConfig()
        assert cfg.enabled is True
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8081


class TestLearningContextMcpConfig:
    def test_defaults(self):
        cfg = LearningContextMcpConfig()
        assert cfg.enabled is False
        assert cfg.url.endswith("/mcp")
        assert cfg.fallback_to_local is True


class TestWorkerSettings:
    def test_default_creation(self):
        settings = WorkerSettings()
        assert settings.worker is not None
        assert settings.health is not None
        assert settings.learning_context_mcp is not None
        assert settings.core is not None
        assert settings.app.log_level == "INFO"

    def test_property_aliases(self):
        settings = WorkerSettings()
        assert settings.app is settings.core.app
        assert settings.qdrant is settings.core.qdrant
        assert settings.minio is settings.core.minio
        assert settings.sql is settings.core.sql
        assert settings.neo4j is settings.core.neo4j
        assert settings.redis is settings.core.redis
        assert settings.parser is settings.core.parser
        assert settings.chunker is settings.core.chunker
        assert settings.embedder is settings.core.embedder
        assert settings.llm is settings.core.llm
        assert settings.tracing is settings.core.tracing
        assert settings.metrics is settings.core.metrics

    def test_default_worker_type(self):
        settings = WorkerSettings()
        assert settings.worker.type == "document"

    def test_env_nested_delimiter(self):
        with mock.patch.dict(os.environ, {"WORKER__TYPE": "enrichment"}):
            settings = WorkerSettings()
            assert settings.worker.type == "enrichment"


class TestGetWorkerSettings:
    def test_returns_worker_settings(self):
        settings = get_worker_settings()
        assert isinstance(settings, WorkerSettings)
        assert settings.worker.type == "document"

    def test_with_explicit_type(self):
        settings = get_worker_settings("enrichment")
        assert settings.worker.type == "enrichment"
        assert settings.worker.service_name == "enrichment-worker"

    def test_with_outbox_type(self):
        settings = get_worker_settings("outbox")
        assert settings.worker.type == "outbox"
        assert settings.worker.service_name == "outbox-worker"

    def test_same_type_no_change(self):
        settings = get_worker_settings("document")
        assert settings.worker.type == "document"
        assert settings.worker.service_name == "document-worker"

    def test_none_keeps_default(self):
        settings = get_worker_settings(None)
        assert settings.worker.type == "document"
