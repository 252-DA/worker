"""
BullMQ worker — xử lý sinh video AI cho Micro Content.
Hàng đợi: ``micro_content_video``
Sự kiện:
- ``MICRO_CONTENT_RENDER_REQUESTED``: Render slide + TTS audio + ghép FFmpeg -> MinIO -> cập nhật video READY/FAILED.
- ``MICRO_CONTENT_SCRIPT_REQUESTED``: Sinh kịch bản nếu cần thêm.

Run:
    python -m worker.runners.micro_content_video_worker
"""
from __future__ import annotations

import asyncio
from typing import Any

from bullmq.custom_errors import UnrecoverableError
from document_chunk.shared.logger import get_logger

from worker.config import get_worker_settings
from worker.container import build_micro_content_video_container
from worker.runtime import (
    build_redis_options,
    install_signal_handlers,
    maybe_start_health_server,
    setup_worker_runtime,
    shutdown_runtime,
)
from worker.use_cases.render_micro_content_video import RenderMicroContentVideoRequest

logger = get_logger(__name__)

MICRO_CONTENT_VIDEO_QUEUE_NAME = "micro_content_video"


def is_final_attempt(bull_job: Any) -> bool:
    opts = getattr(bull_job, "opts", None) or {}
    attempts = int(opts.get("attempts") or 1)
    return int(getattr(bull_job, "attemptsMade", 0) or 0) + 1 >= attempts


def parse_render_job(data: Any) -> RenderMicroContentVideoRequest:
    if not isinstance(data, dict) or not isinstance(data.get("payload"), dict):
        raise ValueError("render job must carry an outbox payload")
    payload = data["payload"]
    micro_content_id = payload.get("micro_content_id") or data.get("aggregate_id")
    course_id = payload.get("course_id")
    script = payload.get("script")
    if not isinstance(micro_content_id, str):
        raise ValueError("render job needs micro_content_id")
    return RenderMicroContentVideoRequest(
        micro_content_id=micro_content_id,
        course_id=course_id,
        script=script,
    )


async def process_video_job(container, bull_job, job_token):
    del job_token
    event_type = bull_job.data.get("event_type") if isinstance(bull_job.data, dict) else None

    if event_type == "MICRO_CONTENT_RENDER_REQUESTED":
        try:
            req = parse_render_job(bull_job.data)
        except (KeyError, TypeError, ValueError) as exc:
            raise UnrecoverableError(f"invalid render job: {exc}") from exc

        logger.info("micro_content_video_worker.start_render", micro_content_id=req.micro_content_id)
        result = await container.render_micro_content_video_use_case.execute(req)
        if result.is_err():
            err = result.error
            logger.error("micro_content_video_worker.render_failed", error=str(err), exc_info=err)
            if is_final_attempt(bull_job):
                container.video_store.update_video_failed(
                    req.micro_content_id, str(err)
                )
            raise err

        resp = result.unwrap()
        return {
            "micro_content_id": resp.micro_content_id,
            "video_object_key": resp.video_object_key,
            "duration": resp.duration_seconds,
            "size": resp.size_bytes,
        }

    logger.warning("micro_content_video_worker.unhandled_event", event_type=event_type)
    return {"status": "ignored"}


async def _main() -> None:
    settings = get_worker_settings("micro_content_video")
    health_state = setup_worker_runtime(settings)
    container = build_micro_content_video_container(settings)
    health_server = maybe_start_health_server(settings, health_state)
    stop_event = asyncio.Event()
    install_signal_handlers(stop_event)

    from bullmq import Worker

    worker = Worker(
        MICRO_CONTENT_VIDEO_QUEUE_NAME,
        lambda job, job_token: process_video_job(container, job, job_token),
        {
            "connection": build_redis_options(settings),
            "concurrency": 1,  # Video rendering is CPU/GPU intensive, process 1 by 1 per worker
        },
    )

    logger.info(
        "micro_content_video_worker.started",
        queue=MICRO_CONTENT_VIDEO_QUEUE_NAME,
        concurrency=1,
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
