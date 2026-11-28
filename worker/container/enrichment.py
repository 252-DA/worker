"""Container của enrichment worker."""
from ai_runtime.catalog import ENRICHMENT

from worker.config import WorkerSettings, get_worker_settings
from worker.container.base import BaseWorkerContainer, build_metadata_store, build_result_publisher
from worker.container.llm import TaskRoles, build_model_registry
from worker.use_cases.run_enrichment import RunEnrichmentUseCase


class EnrichmentWorkerContainer(BaseWorkerContainer):
    """Build only the dependencies needed by the enrichment worker."""

    def __init__(self, settings: WorkerSettings) -> None:
        super().__init__(settings)
        self.metadata_store = self._track("metadata_store", build_metadata_store(settings))
        self.result_publisher = self._track("result_publisher", build_result_publisher(settings))
        self.model_registry = build_model_registry(settings)
        self.roles = TaskRoles(self.model_registry)

    @property
    def run_enrichment_use_case(self) -> RunEnrichmentUseCase:
        """Dựng mỗi job với model đang được chọn; runner chỉ lấy một lần mỗi job."""
        role = self.roles.role(ENRICHMENT)
        return RunEnrichmentUseCase(
            metadata_store=self.metadata_store,
            llm_client=role.client,
            result_publisher=self.result_publisher,
            max_section_chars=self.settings.llm.max_section_chars,
            llm_provider=role.provider,
            llm_profile=role.profile,
        )


def build_enrichment_container(
    settings: WorkerSettings | None = None,
) -> EnrichmentWorkerContainer:
    return EnrichmentWorkerContainer(settings or get_worker_settings("enrichment"))
