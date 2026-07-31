"""BullMQ worker for outbox-driven graph and vector projections.

Run:
    python -m worker.runners.outbox_worker
"""

import asyncio

from bullmq.custom_errors import UnrecoverableError

from document_chunk.domain.outbox_events import OutboxRelayEvent
from document_chunk.domain.ports.job_queue import OUTBOX_RELAY_QUEUE_NAME
from document_chunk.shared.logger import get_logger

from worker.config import get_worker_settings
from worker.container import build_outbox_container
from worker.runtime import (
    build_redis_options,
    install_signal_handlers,
    maybe_start_health_server,
    setup_worker_runtime,
    shutdown_runtime,
)

logger = get_logger(__name__)


async def _process_job(container, bull_job, job_token):
    del job_token

    try:
        event = OutboxRelayEvent.from_mapping(bull_job.data)
    except (KeyError, TypeError, ValueError) as exc:
        logger.error(
            "outbox_worker.invalid_event",
            job_id=getattr(bull_job, "id", None),
            error=str(exc),
        )
        raise UnrecoverableError(f"invalid outbox relay event: {exc}") from exc

    result = await asyncio.to_thread(container.outbox_projector.process, event)
    if result.is_err():
        error = result.error
        logger.error(
            "outbox_worker.projection_failed",
            event_id=event.event_id,
            event_type=event.event_type.value,
            error=str(error),
        )
        if isinstance(error, (KeyError, TypeError, ValueError)):
            raise UnrecoverableError(f"invalid outbox projection event: {error}") from error
        raise error

    logger.info(
        "outbox_worker.event_completed",
        event_id=event.event_id,
        event_type=event.event_type.value,
        aggregate_id=event.aggregate_id,
    )
    return {
        "event_id": event.event_id,
        "event_type": event.event_type.value,
        "aggregate_id": event.aggregate_id,
    }


async def _main() -> None:
    settings = get_worker_settings("outbox")
    health_state = setup_worker_runtime(settings)
    container = build_outbox_container(settings)
    health_server = maybe_start_health_server(settings, health_state)
    stop_event = asyncio.Event()
    install_signal_handlers(stop_event)

    from bullmq import Worker

    worker = Worker(
        OUTBOX_RELAY_QUEUE_NAME,
        lambda job, job_token: _process_job(container, job, job_token),
        {
            "connection": build_redis_options(settings),
            "concurrency": settings.worker.concurrency,
        },
    )
    logger.info(
        "outbox_worker.started",
        queue=OUTBOX_RELAY_QUEUE_NAME,
        concurrency=settings.worker.concurrency,
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
