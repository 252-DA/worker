"""
BullMQ worker — consumer for the document_processing queue.

Run:
    python -m worker.runners.document_worker
"""
import asyncio
import tempfile
from pathlib import Path

from bullmq.custom_errors import UnrecoverableError

from document_chunk.domain.ports.job_queue import DOCUMENT_PROCESSING_QUEUE_NAME, DocumentJobPayload
from document_chunk.shared.logger import get_logger

from worker.config import get_worker_settings
from worker.container import build_document_container
from worker.runtime import (
    build_redis_options,
    install_signal_handlers,
    maybe_start_health_server,
    setup_worker_runtime,
    shutdown_runtime,
)
from worker.use_cases.run_pipeline import RunPipelineRequest

logger = get_logger(__name__)


async def _process_job(container, job, job_token):
    del job_token

    try:
        payload = DocumentJobPayload(**job.data)
    except (TypeError, ValueError) as exc:
        logger.error(
            "document_worker.invalid_payload",
            job_id=getattr(job, "id", None),
            error=str(exc),
            data_keys=sorted(job.data.keys()) if isinstance(job.data, dict) else None,
        )
        raise UnrecoverableError(f"invalid document job payload: {exc}") from exc

    suffix = Path(payload.file_name).suffix or ".tmp"
    tmp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)

        download_result = container.file_storage.download(payload.storage_key, tmp_path)
        if download_result.is_err():
            raise download_result.error

        request = RunPipelineRequest(
            document_id=payload.document_id,
            file_path=tmp_path,
            storage_key=payload.storage_key,
            original_file_name=payload.file_name,
            language=payload.language,
            metadata=payload.metadata,
        )
        result = container.run_pipeline_use_case.execute(request)
        if result.is_err():
            raise result.error

        response = result.unwrap()
        logger.info(
            "document_worker.job.completed",
            document_id=payload.document_id,
            chunks=response.chunk_count,
            duration_ms=response.processing_time_ms,
        )
        return {
            "document_id": payload.document_id,
            "chunk_count": response.chunk_count,
        }
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()


async def _main() -> None:
    settings = get_worker_settings("document")
    health_state = setup_worker_runtime(settings)
    container = build_document_container(settings)
    health_server = maybe_start_health_server(settings, health_state)
    stop_event = asyncio.Event()
    install_signal_handlers(stop_event)

    from bullmq import Worker

    worker = Worker(
        DOCUMENT_PROCESSING_QUEUE_NAME,
        lambda job, job_token: _process_job(container, job, job_token),
        {
            "connection": build_redis_options(settings),
            "concurrency": settings.worker.concurrency,
        },
    )
    logger.info(
        "document_worker.started",
        queue=DOCUMENT_PROCESSING_QUEUE_NAME,
        concurrency=settings.worker.concurrency,
        redis=f"{settings.redis.host}:{settings.redis.port}",
    )

    try:
        await stop_event.wait()
    finally:
        await shutdown_runtime(
            stop_event=stop_event,
            container=container,
            health_server=health_server,
            bullmq_worker=worker,
            timeout_seconds=settings.worker.shutdown_grace_seconds,
        )


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
