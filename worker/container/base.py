"""Lớp cha của mọi container và các builder hạ tầng dùng chung."""
from document_chunk.adapters.graph.neo4j_graph_store import Neo4jGraphStore
from document_chunk.adapters.graph.noop_graph_store import NoopGraphStore
from document_chunk.adapters.metadata.noop_metadata_store import NoopMetadataStore
from document_chunk.adapters.metadata.postgres_metadata_store import PostgresMetadataStore
from document_chunk.adapters.queue.bullmq_adapter import BullMQAdapter
from document_chunk.adapters.storage.minio_adapter import MinioAdapter
from document_chunk.adapters.vector_db.qdrant_adapter import QdrantAdapter
from document_chunk.domain.ports.graph_store import IGraphStore
from document_chunk.domain.ports.metadata_store import IMetadataStore
from document_chunk.shared.logger import get_logger

from worker.adapters.result_publisher import WorkerResultPublisher
from worker.config import WorkerSettings
from worker.runtime import build_redis_options

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


def build_file_storage(settings: WorkerSettings) -> MinioAdapter:
    logger.debug("worker_container.init", component="MinioAdapter")
    return MinioAdapter(settings.minio)


def build_vector_store(settings: WorkerSettings) -> QdrantAdapter:
    logger.debug("worker_container.init", component="QdrantAdapter")
    return QdrantAdapter(settings.qdrant)


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


def build_result_publisher(settings: WorkerSettings) -> WorkerResultPublisher:
    """
    Kết quả về core-api qua hàng đợi ``worker_results`` thay cho ghi thẳng bảng
    của core-api. Chưa kết nối Redis cho tới lần gửi đầu tiên.
    """
    logger.debug("worker_container.init", component="WorkerResultPublisher")
    return WorkerResultPublisher(build_redis_options(settings))
