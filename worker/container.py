from document_chunk.adapters.chunkers.heading_chunker import HeadingChunker
from document_chunk.adapters.embedders.bge_embedder import BgeEmbedder
from document_chunk.adapters.graph.neo4j_graph_store import Neo4jGraphStore
from document_chunk.adapters.graph.noop_graph_store import NoopGraphStore
from document_chunk.adapters.metadata.noop_metadata_store import NoopMetadataStore
from document_chunk.adapters.metadata.postgres_metadata_store import PostgresMetadataStore
from document_chunk.adapters.parsers.docling_pdf_parser import DoclingPdfParser
from document_chunk.adapters.parsers.docx_parser import DocxParser
from document_chunk.adapters.parsers.markdown_parser import MarkdownParser
from document_chunk.adapters.parsers.pptx_parser import PptxParser
from document_chunk.adapters.queue.bullmq_adapter import BullMQAdapter
from document_chunk.adapters.storage.minio_adapter import MinioAdapter
from document_chunk.adapters.vector_db.qdrant_adapter import QdrantAdapter
from document_chunk.domain.ports.graph_store import IGraphStore
from document_chunk.domain.ports.metadata_store import IMetadataStore
from document_chunk.shared.logger import get_logger

from worker.config import WorkerSettings, get_worker_settings
from worker.services.outbox_projector import OutboxProjector
from worker.use_cases.run_enrichment import RunEnrichmentUseCase
from worker.use_cases.run_pipeline import RunPipelineUseCase

logger = get_logger(__name__)


class BaseWorkerContainer:
    def __init__(self, settings: WorkerSettings) -> None:
        self.settings = settings
        self._closables: list[tuple[str, object]] = []

    def _track(self, name: str, component):
        self._closables.append((name, component))
        return component

    def close(self) -> None:
        for name, component in reversed(self._closables):
            close = getattr(component, "close", None)
            if not callable(close):
                continue

            try:
                close()
            except Exception as exc:
                logger.warning("worker_container.close_failed", component=name, error=str(exc))


def build_metadata_store(settings: WorkerSettings) -> IMetadataStore:
    if not settings.sql.enabled:
        logger.warning("worker_container.init", component="NoopMetadataStore", reason="sql.disabled")
        return NoopMetadataStore()
    logger.debug("worker_container.init", component="PostgresMetadataStore")
    return PostgresMetadataStore(settings.sql, settings.outbox)


def build_graph_store(settings: WorkerSettings) -> IGraphStore:
    if not settings.neo4j.enabled:
        logger.warning("worker_container.init", component="NoopGraphStore", reason="neo4j.disabled")
        return NoopGraphStore()
    logger.debug("worker_container.init", component="Neo4jGraphStore")
    return Neo4jGraphStore(settings.neo4j)


def build_parsers(settings: WorkerSettings) -> list:
    return [
        DoclingPdfParser(settings.parser),
        DocxParser(settings.parser),
        PptxParser(settings.parser),
        MarkdownParser(settings.parser),
    ]


def build_chunker(settings: WorkerSettings) -> HeadingChunker:
    logger.debug("worker_container.init", component="HeadingChunker")
    return HeadingChunker(settings.chunker)


def build_embedder(settings: WorkerSettings):
    provider = settings.embedder.provider
    if provider != "bge":
        raise ValueError(
            "worker currently supports only EMBEDDER_PROVIDER=bge; "
            f"received {provider!r}"
        )
    logger.debug("worker_container.init", component="BgeEmbedder")
    return BgeEmbedder(settings.embedder)


def build_file_storage(settings: WorkerSettings) -> MinioAdapter:
    logger.debug("worker_container.init", component="MinioAdapter")
    return MinioAdapter(settings.minio)


def build_vector_store(settings: WorkerSettings) -> QdrantAdapter:
    logger.debug("worker_container.init", component="QdrantAdapter")
    return QdrantAdapter(settings.qdrant)


def build_llm_client(settings: WorkerSettings):
    provider = settings.llm.provider
    if provider == "gemini":
        from document_chunk.adapters.llm.gemini_llm_client import GeminiLLMClient

        logger.debug("worker_container.init", component="GeminiLLMClient")
        return GeminiLLMClient(settings.llm)
    raise ValueError(f"Unknown LLM provider: {provider}")


def build_job_queue(settings: WorkerSettings) -> BullMQAdapter:
    logger.debug("worker_container.init", component="BullMQAdapter")
    return BullMQAdapter(settings.redis)


class DocumentWorkerContainer(BaseWorkerContainer):
    """Build only the dependencies needed by the document processing worker."""

    def __init__(self, settings: WorkerSettings) -> None:
        super().__init__(settings)
        self.file_storage = self._track("file_storage", build_file_storage(settings))
        self.metadata_store = self._track("metadata_store", build_metadata_store(settings))
        self.vector_store = self._track("vector_store", build_vector_store(settings))
        self.job_queue = self._track("job_queue", build_job_queue(settings))
        self.parsers = build_parsers(settings)
        self.chunker = build_chunker(settings)
        self.embedder = build_embedder(settings)
        self.run_pipeline_use_case = RunPipelineUseCase(
            parsers=self.parsers,
            chunker=self.chunker,
            embedder=self.embedder,
            vector_store=self.vector_store,
            metadata_store=self.metadata_store,
            job_queue=self.job_queue,
        )


class EnrichmentWorkerContainer(BaseWorkerContainer):
    """Build only the dependencies needed by the enrichment worker."""

    def __init__(self, settings: WorkerSettings) -> None:
        super().__init__(settings)
        self.metadata_store = self._track("metadata_store", build_metadata_store(settings))
        self.llm_client = self._track("llm_client", build_llm_client(settings))
        self.run_enrichment_use_case = RunEnrichmentUseCase(
            metadata_store=self.metadata_store,
            llm_client=self.llm_client,
            max_section_chars=settings.llm.max_section_chars,
        )


class OutboxWorkerContainer(BaseWorkerContainer):
    """Build only the dependencies needed by the outbox projector worker."""

    def __init__(self, settings: WorkerSettings) -> None:
        super().__init__(settings)
        self.metadata_store = self._track("metadata_store", build_metadata_store(settings))
        self.graph_store = self._track("graph_store", build_graph_store(settings))
        self.vector_store = self._track("vector_store", build_vector_store(settings))
        self.outbox_projector = OutboxProjector(
            metadata_store=self.metadata_store,
            graph_store=self.graph_store,
            vector_store=self.vector_store,
        )


def build_document_container(settings: WorkerSettings | None = None) -> DocumentWorkerContainer:
    return DocumentWorkerContainer(settings or get_worker_settings("document"))


def build_enrichment_container(
    settings: WorkerSettings | None = None,
) -> EnrichmentWorkerContainer:
    return EnrichmentWorkerContainer(settings or get_worker_settings("enrichment"))


def build_outbox_container(settings: WorkerSettings | None = None) -> OutboxWorkerContainer:
    return OutboxWorkerContainer(settings or get_worker_settings("outbox"))
