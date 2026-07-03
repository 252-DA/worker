"""
Outbox worker — polls PostgreSQL outbox and projects derived-store updates.

Run:
    python -m worker.runners.outbox_worker
"""
import asyncio

from document_chunk.shared.logger import get_logger

from worker.config import get_worker_settings
from worker.container import build_outbox_container
from worker.runtime import (
    install_signal_handlers,
    maybe_start_health_server,
    setup_worker_runtime,
    shutdown_runtime,
    sleep_until_stop,
)

logger = get_logger(__name__)


async def _main() -> None:
    settings = get_worker_settings("outbox")
    health_state = setup_worker_runtime(settings)
    container = build_outbox_container(settings)
    health_server = maybe_start_health_server(settings, health_state)
    stop_event = asyncio.Event()
    install_signal_handlers(stop_event)

    logger.info(
        "outbox_worker.started",
        batch_size=settings.outbox.batch_size,
        poll_interval_seconds=settings.outbox.poll_interval_seconds,
    )

    try:
        while not stop_event.is_set():
            result = container.outbox_projector.run_once(limit=settings.outbox.batch_size)
            if result.is_err():
                logger.error("outbox_worker.poll_failed", error=str(result.error))
            else:
                processed = result.unwrap()
                if processed:
                    logger.info("outbox_worker.batch_processed", processed=processed)

            await sleep_until_stop(
                stop_event,
                timeout_seconds=settings.outbox.poll_interval_seconds,
            )
    finally:
        await shutdown_runtime(
            stop_event=stop_event,
            container=container,
            health_server=health_server,
            timeout_seconds=settings.worker.shutdown_grace_seconds,
        )


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
