"""
BullMQ worker — consumer for the document_enrichment queue.

Run:
    python -m worker.runners.enrichment_worker
"""
import asyncio

from bullmq.custom_errors import UnrecoverableError

from document_chunk.domain.ports.job_queue import DOCUMENT_ENRICHMENT_QUEUE_NAME, EnrichmentJobPayload
from document_chunk.shared.logger import get_logger

from worker.config import get_worker_settings
from worker.container import build_enrichment_container
from worker.runtime import (
    build_redis_options,
    install_signal_handlers,
    maybe_start_health_server,
    setup_worker_runtime,
    shutdown_runtime,
)
from worker.services.document_reporter import job_attempt
from worker.services.processing_generation import newer_generation
from worker.use_cases.run_enrichment import RunEnrichmentRequest

logger = get_logger(__name__)


def _run_enrichment_sync(container, payload: EnrichmentJobPayload, attempt: int):
    return container.run_enrichment_use_case.execute(
        RunEnrichmentRequest(
            document_id=payload.document_id,
            processing_generation=payload.processing_generation,
            attempt=attempt,
        )
    )


async def _process_job(container, job, job_token):
    del job_token

    try:
        payload = EnrichmentJobPayload(**job.data)
    except (TypeError, ValueError) as exc:
        logger.error(
            "enrichment_worker.invalid_payload",
            job_id=getattr(job, "id", None),
            error=str(exc),
            data_keys=sorted(job.data.keys()) if isinstance(job.data, dict) else None,
        )
        raise UnrecoverableError(f"invalid enrichment job payload: {exc}") from exc

    # Enrichment của lượt cũ: tài liệu đã được xử lý lại, enrichment của lượt mới sẽ
    # chạy sau lượt đó. Bỏ để không tốn LLM và không ghi concepts chồng lên.
    # Job cũ chưa mang lượt (0) không kiểm: giữ hành vi cũ, khỏi tốn một lần đọc DB.
    current_generation = None
    if payload.processing_generation > 0:
        current_generation = await asyncio.to_thread(
            newer_generation, container.metadata_store, payload.document_id, payload.processing_generation
        )
    if current_generation is not None:
        logger.warning(
            "enrichment_worker.job.stale_generation_skipped",
            document_id=payload.document_id,
            job_generation=payload.processing_generation,
            current_generation=current_generation,
        )
        return {
            "document_id": payload.document_id,
            "skipped": "stale_generation",
            "job_generation": payload.processing_generation,
            "current_generation": current_generation,
        }

    # Use case gọi LLM, psycopg và publish_sync đồng bộ: chạy ngoài event loop.
    # Lỗi (kể cả không gửi được kết quả về core-api) → job lỗi, BullMQ retry.
    result = await asyncio.to_thread(_run_enrichment_sync, container, payload, job_attempt(job))
    if result.is_err():
        raise result.error

    response = result.unwrap()
    logger.info(
        "enrichment_worker.job.completed",
        document_id=payload.document_id,
        concepts=response.concept_count,
        mentions=response.mention_count,
        cards=response.card_count,
        quiz_items=response.quiz_count,
    )
    return {
        "document_id": payload.document_id,
        "concept_count": response.concept_count,
        "mention_count": response.mention_count,
        "card_count": response.card_count,
        "quiz_count": response.quiz_count,
    }


async def _main() -> None:
    settings = get_worker_settings("enrichment")
    health_state = setup_worker_runtime(settings)
    container = build_enrichment_container(settings)
    health_server = maybe_start_health_server(settings, health_state)
    stop_event = asyncio.Event()
    install_signal_handlers(stop_event)

    from bullmq import Worker

    worker = Worker(
        DOCUMENT_ENRICHMENT_QUEUE_NAME,
        lambda job, job_token: _process_job(container, job, job_token),
        {
            "connection": build_redis_options(settings),
            "concurrency": settings.worker.concurrency,
        },
    )
    logger.info(
        "enrichment_worker.started",
        queue=DOCUMENT_ENRICHMENT_QUEUE_NAME,
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
