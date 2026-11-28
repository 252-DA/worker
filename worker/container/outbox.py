"""Container của outbox worker (chiếu sự kiện sang Qdrant/Neo4j)."""
from worker.config import WorkerSettings, get_worker_settings
from worker.container.base import (
    BaseWorkerContainer,
    build_graph_store,
    build_metadata_store,
    build_result_publisher,
    build_vector_store,
)
from worker.services.outbox_projector import OutboxProjector


class OutboxWorkerContainer(BaseWorkerContainer):
    """Build only the dependencies needed by the outbox projector worker."""

    def __init__(self, settings: WorkerSettings) -> None:
        super().__init__(settings)
        self.metadata_store = self._track("metadata_store", build_metadata_store(settings))
        self.graph_store = self._track("graph_store", build_graph_store(settings))
        self.vector_store = self._track("vector_store", build_vector_store(settings))
        self.result_publisher = self._track("result_publisher", build_result_publisher(settings))
        self.outbox_projector = OutboxProjector(
            metadata_store=self.metadata_store,
            graph_store=self.graph_store,
            vector_store=self.vector_store,
            result_publisher=self.result_publisher,
        )


def build_outbox_container(settings: WorkerSettings | None = None) -> OutboxWorkerContainer:
    return OutboxWorkerContainer(settings or get_worker_settings("outbox"))
