"""
Tests for worker/container.py — container builders and dependency wiring.
"""
from unittest.mock import MagicMock, patch

import pytest

from worker.config import WorkerSettings
from worker.container import (
    BaseWorkerContainer,
    DocumentWorkerContainer,
    EnrichmentWorkerContainer,
    OutboxWorkerContainer,
    build_chunker,
    build_document_container,
    build_embedder,
    build_enrichment_container,
    build_file_storage,
    build_graph_store,
    build_job_queue,
    build_learning_context_client,
    build_llm_client,
    build_metadata_store,
    build_outbox_container,
    build_parsers,
    build_vector_store,
)


class TestBaseWorkerContainer:
    def test_track_and_close(self):
        settings = WorkerSettings()
        container = BaseWorkerContainer(settings)

        closable = MagicMock()
        container._track("test", closable)
        container.close()

        closable.close.assert_called_once()

    def test_close_reversed_order(self):
        settings = WorkerSettings()
        container = BaseWorkerContainer(settings)

        calls = []
        a = MagicMock()
        a.close.side_effect = lambda: calls.append("a")
        b = MagicMock()
        b.close.side_effect = lambda: calls.append("b")

        container._track("a", a)
        container._track("b", b)
        container.close()

        assert calls == ["b", "a"]  # reversed

    def test_close_handles_missing_close_method(self):
        settings = WorkerSettings()
        container = BaseWorkerContainer(settings)

        no_close = MagicMock(spec=[])
        container._track("no_close", no_close)
        container.close()  # should not raise

    def test_close_handles_exceptions_gracefully(self):
        settings = WorkerSettings()
        container = BaseWorkerContainer(settings)

        a = MagicMock()
        a.close.side_effect = RuntimeError("close failed")
        b = MagicMock()

        container._track("a", a)
        container._track("b", b)
        container.close()  # should not raise, b.close should still be called

        b.close.assert_called_once()


class TestBuildFunctions:
    def test_build_metadata_store_sql_disabled(self):
        settings = WorkerSettings()
        store = build_metadata_store(settings)
        from document_chunk.adapters.metadata.noop_metadata_store import NoopMetadataStore
        assert isinstance(store, NoopMetadataStore)

    def test_build_graph_store_neo4j_disabled(self):
        settings = WorkerSettings()
        store = build_graph_store(settings)
        from document_chunk.adapters.graph.noop_graph_store import NoopGraphStore
        assert isinstance(store, NoopGraphStore)

    def test_build_parsers(self):
        settings = WorkerSettings()
        parsers = build_parsers(settings)
        assert len(parsers) == 4  # PDF, DOCX, PPTX, Markdown

    def test_build_chunker(self):
        settings = WorkerSettings()
        chunker = build_chunker(settings)
        from document_chunk.adapters.chunkers.heading_chunker import HeadingChunker
        assert isinstance(chunker, HeadingChunker)

    def test_build_embedder_bge_default(self):
        settings = WorkerSettings()
        embedder = build_embedder(settings)
        from document_chunk.adapters.embedders.bge_embedder import BgeEmbedder
        assert isinstance(embedder, BgeEmbedder)

    def test_build_embedder_unsupported_provider_raises(self):
        settings = WorkerSettings()
        settings.core.embedder.provider = "openai"
        with pytest.raises(ValueError, match="bge"):
            build_embedder(settings)

    def test_build_file_storage(self):
        settings = WorkerSettings()
        storage = build_file_storage(settings)
        from document_chunk.adapters.storage.minio_adapter import MinioAdapter
        assert isinstance(storage, MinioAdapter)

    def test_build_vector_store(self):
        settings = WorkerSettings()
        store = build_vector_store(settings)
        from document_chunk.adapters.vector_db.qdrant_adapter import QdrantAdapter
        assert isinstance(store, QdrantAdapter)

    def test_build_llm_client_gemini(self):
        settings = WorkerSettings()
        client = build_llm_client(settings)
        from worker.adapters.ai_runtime_llm_client import AIRuntimeLLMClient

        assert isinstance(client, AIRuntimeLLMClient)

    def test_build_llm_client_unknown_provider_raises(self):
        settings = WorkerSettings()
        settings.core.llm.provider = "unknown"
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            build_llm_client(settings)

    def test_learning_context_client_disabled_by_default(self):
        settings = WorkerSettings()
        assert build_learning_context_client(settings) is None

    def test_learning_context_client_enabled(self):
        settings = WorkerSettings()
        settings.learning_context_mcp.enabled = True
        client = build_learning_context_client(settings)
        from ai_runtime.mcp import LearningContextClient

        assert isinstance(client, LearningContextClient)

    def test_build_job_queue(self):
        settings = WorkerSettings()
        queue = build_job_queue(settings)
        from document_chunk.adapters.queue.bullmq_adapter import BullMQAdapter
        assert isinstance(queue, BullMQAdapter)


class TestContainerClasses:
    def test_document_worker_container(self):
        with patch.object(DocumentWorkerContainer, "__init__", lambda self, s: None):
            settings = WorkerSettings()
            container = DocumentWorkerContainer.__new__(DocumentWorkerContainer)
            container.__init__(settings)

    def test_enrichment_worker_container(self):
        settings = WorkerSettings()
        container = EnrichmentWorkerContainer(settings)
        assert container.run_enrichment_use_case is not None
        container.close()

    def test_outbox_worker_container(self):
        settings = WorkerSettings()
        container = OutboxWorkerContainer(settings)
        assert container.outbox_projector is not None
        container.close()


class TestBuilderShortcuts:
    def test_build_document_container(self):
        settings = WorkerSettings()
        container = build_document_container(settings)
        assert isinstance(container, DocumentWorkerContainer)
        container.close()

    def test_build_enrichment_container(self):
        settings = WorkerSettings()
        container = build_enrichment_container(settings)
        assert isinstance(container, EnrichmentWorkerContainer)
        container.close()

    def test_build_outbox_container(self):
        settings = WorkerSettings()
        container = build_outbox_container(settings)
        assert isinstance(container, OutboxWorkerContainer)
        container.close()
