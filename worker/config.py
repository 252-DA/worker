from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from document_chunk.infrastructure.config import Settings as CoreSettings


class WorkerRuntimeConfig(BaseSettings):
    type: Literal["document", "enrichment", "outbox", "content_generation"] = "document"
    concurrency: int = 3
    shutdown_grace_seconds: float = 30.0
    service_name: str | None = None

    model_config = SettingsConfigDict(env_prefix="WORKER_")

    def model_post_init(self, __context) -> None:
        if self.service_name is None:
            self.service_name = f"{self.type}-worker"


class HealthConfig(BaseSettings):
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8081

    model_config = SettingsConfigDict(env_prefix="HEALTH_")


class LearningContextMcpConfig(BaseSettings):
    enabled: bool = False
    url: str = "http://mcp-server:8001/mcp"
    timeout_seconds: float = 30
    fallback_to_local: bool = True

    model_config = SettingsConfigDict(env_prefix="LEARNING_CONTEXT_MCP_")


class WorkerSettings(BaseSettings):
    worker: WorkerRuntimeConfig = Field(default_factory=WorkerRuntimeConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)
    learning_context_mcp: LearningContextMcpConfig = Field(
        default_factory=LearningContextMcpConfig
    )
    core: CoreSettings = Field(default_factory=CoreSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @property
    def app(self):
        return self.core.app

    @property
    def qdrant(self):
        return self.core.qdrant

    @property
    def minio(self):
        return self.core.minio

    @property
    def sql(self):
        return self.core.sql

    @property
    def neo4j(self):
        return self.core.neo4j

    @property
    def redis(self):
        return self.core.redis

    @property
    def parser(self):
        return self.core.parser

    @property
    def chunker(self):
        return self.core.chunker

    @property
    def embedder(self):
        return self.core.embedder

    @property
    def llm(self):
        return self.core.llm

    @property
    def tracing(self):
        return self.core.tracing

    @property
    def metrics(self):
        return self.core.metrics


@lru_cache
def get_worker_settings(
    worker_type: Literal[
        "document",
        "enrichment",
        "outbox",
        "content_generation",
    ]
    | None = None,
) -> WorkerSettings:
    settings = WorkerSettings()
    if worker_type is None or settings.worker.type == worker_type:
        return settings

    previous_type = settings.worker.type
    previous_service_name = settings.worker.service_name
    settings.worker.type = worker_type
    if previous_service_name in (None, f"{previous_type}-worker"):
        settings.worker.service_name = f"{worker_type}-worker"
    return settings
