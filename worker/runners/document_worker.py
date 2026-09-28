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
from document_chunk.domain.ports.metadata_store import IngestionStatus
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
from document_chunk.application.use_cases.map_chunks_to_los import MapChunksToLosRequest

from worker.use_cases.curriculum_import import CURRICULUM_JOB_NAMES
from worker.use_cases.run_pipeline import RunPipelineRequest

MAP_DOCUMENT_LOS_JOB = "map_document_los"

logger = get_logger(__name__)

_COMPLETED_DOCUMENT_STATUSES = {
    IngestionStatus.INDEXED,
    IngestionStatus.ENRICHING,
    IngestionStatus.GENERATED_DRAFT,
}


def _process_document_job(container, payload: DocumentJobPayload):
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


async def _process_curriculum_job(container, job):
    import_id = job.data.get("import_id") if isinstance(job.data, dict) else None
    if not import_id:
        logger.error("document_worker.invalid_curriculum_payload", job_id=getattr(job, "id", None))
        raise UnrecoverableError("curriculum job requires import_id")
    return await asyncio.to_thread(container.curriculum_import_use_case.run, job.name, import_id)


def _map_document_los(container, document_id: str, course_id: str):
    result = container.map_chunks_to_los_use_case.execute(
        MapChunksToLosRequest(document_id=document_id, course_id=course_id)
    )
    if result.is_err():
        raise result.error
    return {"document_id": document_id, "mapping_count": result.unwrap().mapping_count}


async def _process_map_document_los_job(container, job):
    """Giảng viên đổi vai trò/chương của tài liệu → map lại chunk → LO."""
    data = job.data if isinstance(job.data, dict) else {}
    document_id, course_id = data.get("document_id"), data.get("course_id")
    if not document_id or not course_id:
        logger.error("document_worker.invalid_map_payload", job_id=getattr(job, "id", None))
        raise UnrecoverableError("map_document_los job requires document_id and course_id")
    return await asyncio.to_thread(_map_document_los, container, document_id, course_id)


async def _process_job(container, job, job_token):
    del job_token

    # Đề cương và map lại LO đi chung hàng đợi với tài liệu; phân loại theo tên job.
    if getattr(job, "name", None) in CURRICULUM_JOB_NAMES:
        return await _process_curriculum_job(container, job)
    if getattr(job, "name", None) == MAP_DOCUMENT_LOS_JOB:
        return await _process_map_document_los_job(container, job)

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

    status_result = container.metadata_store.get_document_status(payload.document_id)
    if status_result.is_err():
        raise status_result.error

    stored_status = status_result.unwrap()
    if stored_status is not None and stored_status[0] in _COMPLETED_DOCUMENT_STATUSES:
        chunks_result = container.metadata_store.list_chunks(payload.document_id)
        if chunks_result.is_err():
            raise chunks_result.error

        chunk_count = len(chunks_result.unwrap())
        logger.info(
            "document_worker.job.already_completed",
            document_id=payload.document_id,
            status=stored_status[0].value,
            chunks=chunk_count,
        )
        return {
            "document_id": payload.document_id,
            "chunk_count": chunk_count,
        }

    return await asyncio.to_thread(_process_document_job, container, payload)


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
