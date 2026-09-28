from document_chunk.adapters.chunkers.structural import build_chunker as create_chunker
from document_chunk.domain.ports.chunker import IChunker
from document_chunk.adapters.embedders.bge_embedder import BgeEmbedder
from document_chunk.adapters.graph.neo4j_graph_store import Neo4jGraphStore
from document_chunk.adapters.graph.noop_graph_store import NoopGraphStore
from document_chunk.adapters.metadata.noop_metadata_store import NoopMetadataStore
from document_chunk.adapters.metadata.postgres_metadata_store import PostgresMetadataStore
from document_chunk.adapters.parsers.adaptive_pdf_parser import AdaptivePdfParser
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
    return PostgresMetadataStore(settings.sql)


def build_graph_store(settings: WorkerSettings) -> IGraphStore:
    if not settings.neo4j.enabled:
        logger.warning("worker_container.init", component="NoopGraphStore", reason="neo4j.disabled")
        return NoopGraphStore()
    logger.debug("worker_container.init", component="Neo4jGraphStore")
    return Neo4jGraphStore(settings.neo4j)


def build_parsers(settings: WorkerSettings) -> list:
    return [
        AdaptivePdfParser(settings.parser),
        DocxParser(settings.parser),
        PptxParser(settings.parser),
        MarkdownParser(settings.parser),
    ]


def build_chunker(settings: WorkerSettings) -> IChunker:
    return create_chunker(settings.chunker, settings.embedder)


def build_embedder(settings: WorkerSettings):
    provider = settings.embedder.provider
    if provider == "grpc":
        from document_chunk.adapters.embedders.grpc_embedder import GrpcEmbedder

        logger.debug("worker_container.init", component="GrpcEmbedder")
        return GrpcEmbedder(settings.embedder, expected_dimension=settings.qdrant.vector_size)

    if provider != "bge":
        raise ValueError(
            "worker supports only EMBEDDER_PROVIDER=bge or grpc; "
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
    from ai_runtime import AIRuntime, ModelConfig, build_model_client
    from worker.adapters.ai_runtime_llm_client import AIRuntimeLLMClient

    if settings.llm.provider not in {"gemini", "openai-compatible", "deepseek"}:
        raise ValueError(f"Unknown LLM provider: {settings.llm.provider}")

    logger.debug(
        "worker_container.init",
        component="AIRuntimeLLMClient",
        provider=settings.llm.provider,
    )
    model = build_model_client(
        ModelConfig(
            provider=settings.llm.provider,
            model=settings.llm.model,
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
            temperature=settings.llm.temperature,
            timeout_seconds=settings.llm.timeout_seconds,
            max_retries=settings.llm.max_retries,
        )
    )
    return AIRuntimeLLMClient(AIRuntime(model))


def build_learning_context_client(settings: WorkerSettings):
    if not settings.learning_context_mcp.enabled:
        return None

    from ai_runtime.mcp import LearningContextClient, MCPToolClient

    logger.debug(
        "worker_container.init",
        component="LearningContextClient",
        url=settings.learning_context_mcp.url,
    )
    return LearningContextClient(
        MCPToolClient(
            url=settings.learning_context_mcp.url,
            timeout_seconds=settings.learning_context_mcp.timeout_seconds,
        )
    )


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
        self.embedder = self._track("embedder", build_embedder(settings))
        self.graph_store = self._track("graph_store", build_graph_store(settings))
        self.map_chunks_to_los_use_case = self._build_map_chunks_to_los()
        self.run_pipeline_use_case = RunPipelineUseCase(
            parsers=self.parsers,
            chunker=self.chunker,
            embedder=self.embedder,
            vector_store=self.vector_store,
            metadata_store=self.metadata_store,
            job_queue=self.job_queue,
            map_chunks_to_los_use_case=self.map_chunks_to_los_use_case,
        )
        self.curriculum_import_use_case = self._build_curriculum_import()

    def _build_curriculum_import(self):
        from document_chunk.adapters.curriculum.dcmh_extractor import DcmhExtractor
        from document_chunk.application.use_cases.ingest_curriculum import (
            IngestCurriculumUseCase,
        )
        from worker.services.curriculum_import_store import CurriculumImportStore
        from worker.use_cases.curriculum_import import CurriculumImportUseCase

        return CurriculumImportUseCase(
            store=CurriculumImportStore(self.settings.sql),
            file_storage=self.file_storage,
            ingest_curriculum_use_case=IngestCurriculumUseCase(
                parsers=self.parsers,
                curriculum_extractor=DcmhExtractor(),
                metadata_store=self.metadata_store,
                graph_store=self.graph_store,
            ),
            map_chunks_to_los_use_case=self.map_chunks_to_los_use_case,
        )

    def _build_map_chunks_to_los(self):
        from document_chunk.adapters.curriculum.embedding_chapter_matcher import (
            EmbeddingChapterMatcher,
        )
        from document_chunk.adapters.curriculum.heuristic_lo_mapper import HeuristicLoMapper
        from document_chunk.application.use_cases.map_chunks_to_los import (
            MapChunksToLosUseCase,
        )

        return MapChunksToLosUseCase(
            metadata_store=self.metadata_store,
            graph_store=self.graph_store,
            lo_mapper=HeuristicLoMapper(),
            # Khớp chương theo nội dung cho tài liệu không nằm trong module theo chương.
            chapter_matcher=EmbeddingChapterMatcher(self.embedder),
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


class CurriculumQuizContainer(BaseWorkerContainer):
    """Dependencies for synchronous or queued curriculum-aware quiz generation."""

    def __init__(self, settings: WorkerSettings) -> None:
        super().__init__(settings)
        from worker.use_cases.generate_curriculum_quiz import GenerateCurriculumQuizUseCase

        self.metadata_store = self._track("metadata_store", build_metadata_store(settings))
        self.llm_client = self._track("llm_client", build_llm_client(settings))
        self.context_client = build_learning_context_client(settings)
        self.generate_curriculum_quiz_use_case = GenerateCurriculumQuizUseCase(
            metadata_store=self.metadata_store,
            llm_client=self.llm_client,
            context_client=self.context_client,
            fallback_to_local_context=settings.learning_context_mcp.fallback_to_local,
            llm_provider=settings.llm.provider,
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


def build_curriculum_quiz_container(
    settings: WorkerSettings | None = None,
) -> CurriculumQuizContainer:
    return CurriculumQuizContainer(settings or get_worker_settings())
