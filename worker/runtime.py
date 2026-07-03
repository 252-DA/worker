import asyncio
import inspect
import signal

from document_chunk.shared.logger import get_logger, setup_logging
from document_chunk.shared.metrics import start_metrics_server
from document_chunk.shared.tracing import setup_tracing

from worker.config import WorkerSettings
from worker.health import HealthState, start_health_server

logger = get_logger(__name__)


def setup_worker_runtime(settings: WorkerSettings) -> HealthState:
    setup_logging(level=settings.app.log_level, json_logs=settings.app.json_logs)

    if settings.tracing.enabled:
        setup_tracing(
            service_name=settings.worker.service_name,
            otlp_endpoint=settings.tracing.otlp_endpoint,
            debug=settings.app.debug,
        )

    if settings.metrics.enabled:
        start_metrics_server(port=settings.metrics.port)

    return HealthState(
        service=settings.worker.service_name,
        worker_type=settings.worker.type,
    )


def maybe_start_health_server(settings: WorkerSettings, state: HealthState):
    if not settings.health.enabled:
        return None

    return start_health_server(
        host=settings.health.host,
        port=settings.health.port,
        state=state,
    )


def build_redis_options(settings: WorkerSettings) -> dict:
    opts = {
        "host": settings.redis.host,
        "port": settings.redis.port,
        "db": settings.redis.db,
    }
    if settings.redis.password:
        opts["password"] = settings.redis.password
    return opts


def install_signal_handlers(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        if not stop_event.is_set():
            stop_event.set()

    for signame in ("SIGTERM", "SIGINT"):
        signum = getattr(signal, signame)
        try:
            loop.add_signal_handler(signum, _request_stop)
        except NotImplementedError:
            signal.signal(signum, lambda *_: _request_stop())


async def sleep_until_stop(stop_event: asyncio.Event, timeout_seconds: float) -> None:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        return


async def drain_bullmq_worker(worker, timeout_seconds: float) -> None:
    close = getattr(worker, "close", None)
    if not callable(close):
        return

    result = close()
    if inspect.isawaitable(result):
        await asyncio.wait_for(result, timeout=timeout_seconds)


async def shutdown_runtime(
    stop_event: asyncio.Event,
    container,
    health_server,
    bullmq_worker=None,
    timeout_seconds: float = 30.0,
) -> None:
    stop_event.set()
    if health_server is not None:
        health_server.set_ready(False)

    if bullmq_worker is not None:
        try:
            await drain_bullmq_worker(bullmq_worker, timeout_seconds=timeout_seconds)
        except Exception as exc:
            logger.warning("worker_runtime.bullmq_close_failed", error=str(exc))

    if health_server is not None:
        health_server.close()

    container.close()
