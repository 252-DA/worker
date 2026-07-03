"""
RunPipelineUseCase — async worker path: run pipeline on a pre-uploaded document.

Assumes:
  - File is already uploaded to MinIO (storage_key known)
  - Document record already exists in PostgreSQL with QUEUED status
  - file_path is a local temp file downloaded from MinIO by the worker

Delegates the full parse → chunk → embed → upsert → graph cycle to PipelineCore.
"""
from dataclasses import dataclass
from pathlib import Path

from document_chunk.application.use_cases._pipeline_core import PipelineCore
from document_chunk.domain.constants import EXT_MAP, MIME_MAP
from document_chunk.domain.entities.document import Document
from document_chunk.domain.exceptions import UnsupportedFileTypeError
from document_chunk.domain.ports.chunker import IChunker
from document_chunk.domain.ports.embedder import IEmbedder
from document_chunk.domain.ports.job_queue import EnrichmentJobPayload, IJobQueue
from document_chunk.domain.ports.metadata_store import IMetadataStore
from document_chunk.domain.ports.parser import IParser
from document_chunk.domain.ports.preprocessor import IPreprocessor
from document_chunk.domain.ports.vector_store import IVectorStore
from document_chunk.shared.logger import get_logger
from document_chunk.shared.metrics import ACTIVE_REQUESTS
from document_chunk.shared.result import Err, Ok, Result
from document_chunk.shared.tracing import get_tracer

logger = get_logger(__name__)
tracer = get_tracer(__name__)


@dataclass
class RunPipelineRequest:
    document_id: str
    file_path: Path           # temp file downloaded from MinIO
    storage_key: str          # MinIO key — already uploaded by EnqueueDocumentUseCase
    original_file_name: str
    language: str | None = None
    metadata: dict | None = None


@dataclass
class RunPipelineResponse:
    document_id: str
    chunk_count: int
    processing_time_ms: float


class RunPipelineUseCase:
    def __init__(
        self,
        parsers: list[IParser],
        chunker: IChunker,
        embedder: IEmbedder,
        vector_store: IVectorStore,
        metadata_store: IMetadataStore,
        job_queue: IJobQueue,
        preprocessors: list[IPreprocessor] | None = None,
    ) -> None:
        self._job_queue = job_queue
        self._pipeline = PipelineCore(
            parsers=parsers,
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
            metadata_store=metadata_store,
            preprocessors=preprocessors,
        )

    def execute(self, request: RunPipelineRequest) -> Result[RunPipelineResponse, Exception]:
        ACTIVE_REQUESTS.inc()
        try:
            document_id = request.document_id
            metadata = dict(request.metadata or {})
            metadata["storage_key"] = request.storage_key
            metadata["source_file_name"] = request.original_file_name

            with tracer.start_as_current_span("run_pipeline") as span:
                span.set_attribute("document.id", document_id)
                span.set_attribute("file.name", request.original_file_name)

                doc_type = EXT_MAP.get(Path(request.original_file_name).suffix.lower())
                if doc_type is None:
                    return Err(UnsupportedFileTypeError(Path(request.original_file_name).suffix))

                span.set_attribute("document.type", doc_type.value)

                document = Document(
                    id=document_id,
                    name=request.original_file_name,
                    path=request.file_path,
                    doc_type=doc_type,
                    size_bytes=request.file_path.stat().st_size,
                    mime_type=MIME_MAP.get(doc_type, "application/octet-stream"),
                )

                core_result = self._pipeline.run(
                    document=document,
                    metadata=metadata,
                    language=request.language,
                )

                if core_result.is_err():
                    logger.error(
                        "run_pipeline.failed",
                        document_id=document_id,
                        error=str(core_result.error),
                    )
                    return Err(core_result.error)

                core = core_result.unwrap()
                logger.info(
                    "run_pipeline.completed",
                    document_id=document_id,
                    chunks=len(core.chunks),
                    duration_ms=round(core.duration_ms, 2),
                )
                self._enqueue_enrichment(document_id)
                return Ok(
                    RunPipelineResponse(
                        document_id=document_id,
                        chunk_count=len(core.chunks),
                        processing_time_ms=round(core.duration_ms, 2),
                    )
                )
        finally:
            ACTIVE_REQUESTS.dec()

    def _enqueue_enrichment(self, document_id: str) -> None:
        enqueue_result = self._job_queue.enqueue_enrichment(
            EnrichmentJobPayload(document_id=document_id)
        )
        if enqueue_result.is_err():
            logger.warning(
                "run_pipeline.enrichment_enqueue_failed",
                document_id=document_id,
                error=str(enqueue_result.error),
            )
