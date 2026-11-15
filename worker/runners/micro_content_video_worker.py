"""
BullMQ worker — xử lý sinh video AI cho Micro Content.
Hàng đợi: ``micro_content_video``
Sự kiện:
- ``MICRO_CONTENT_RENDER_REQUESTED``: Render slide + TTS audio + ghép FFmpeg -> object
  storage -> gửi MICRO_CONTENT_VIDEO_RESULT (READY/FAILED) về core-api qua
  ``worker_results``; core-api ghi micro_contents/video_assets, worker không ghi.
- ``MICRO_CONTENT_SCRIPT_REQUESTED``: chưa có việc gì (core-api tự ghi kịch bản mặc
  định khi phát lệnh); nhận job và bỏ qua.

Run:
    python -m worker.runners.micro_content_video_worker
"""
from __future__ import annotations

import asyncio
import uuid
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
from worker.use_cases.render_micro_content_video import (
    InvalidRenderRequest,
    RenderMicroContentVideoRequest,
)

logger = get_logger(__name__)

MICRO_CONTENT_VIDEO_QUEUE_NAME = "micro_content_video"


def is_final_attempt(bull_job: Any) -> bool:
    opts = getattr(bull_job, "opts", None) or {}
    attempts = int(opts.get("attempts") or 1)
    return int(getattr(bull_job, "attemptsMade", 0) or 0) + 1 >= attempts


def _uuid_field(payload: dict, key: str) -> str:
    # Dạng chuẩn có gạch nối như core-api kiểm kết quả: id lạ thì kết quả bị
    # core-api từ chối và bài kẹt ở PROCESSING, nên chặn ngay từ lệnh.
    value = payload.get(key)
    try:
        canonical = isinstance(value, str) and str(uuid.UUID(value)) == value.lower()
    except ValueError:
        canonical = False
    if not canonical:
        raise ValueError(f"render job needs {key} as a UUID, got {value!r}")
    return value


def parse_render_job(data: Any) -> RenderMicroContentVideoRequest:
    """
    Lệnh theo contracts/integration-events/v1/micro-content-render-requested.json.
    Thiếu id (kể cả ``render_id`` — lệnh phát trước khi có cột ai_render_id) thì
    không gửi được kết quả core-api áp dụng được: lỗi lệnh, không thử lại.
    Kịch bản để use case kiểm, vì kịch bản hỏng vẫn phải báo FAILED cho bài.
    """
    if not isinstance(data, dict) or not isinstance(data.get("payload"), dict):
        raise ValueError("render job must carry an outbox payload")
    payload = dict(data["payload"])
    payload.setdefault("micro_content_id", data.get("aggregate_id"))
    event_id = data.get("event_id")
    raw_key = payload.get("source_document_key")
    raw_mime = payload.get("source_document_mime_type")
    return RenderMicroContentVideoRequest(
        micro_content_id=_uuid_field(payload, "micro_content_id"),
        course_id=_uuid_field(payload, "course_id"),
        asset_id=_uuid_field(payload, "asset_id"),
        render_id=_uuid_field(payload, "render_id"),
        script=payload.get("script"),
        source_document_key=raw_key.strip() if isinstance(raw_key, str) and raw_key.strip() else None,
        source_document_mime_type=raw_mime.strip() if isinstance(raw_mime, str) and raw_mime.strip() else None,
        source_event_id=event_id if isinstance(event_id, str) and event_id else None,
    )


async def process_video_job(container, bull_job, job_token):
    del job_token
    event_type = bull_job.data.get("event_type") if isinstance(bull_job.data, dict) else None

    if event_type == "MICRO_CONTENT_RENDER_REQUESTED":
        try:
            req = parse_render_job(bull_job.data)
        except (KeyError, TypeError, ValueError) as exc:
            raise UnrecoverableError(f"invalid render job: {exc}") from exc

        logger.info(
            "micro_content_video_worker.start_render",
            micro_content_id=req.micro_content_id,
            render_id=req.render_id,
        )
        use_case = container.render_micro_content_video_use_case
        result = await use_case.execute(req)
        if result.is_err():
            err = result.error
            permanent = isinstance(err, InvalidRenderRequest)
            logger.error("micro_content_video_worker.render_failed", error=str(err), exc_info=err)
            # Chỉ báo FAILED khi không còn thử lại: lần thử sau có thể thành công,
            # và core-api không cho READY đè FAILED của cùng lần render. Gửi FAILED
            # lỗi thì lỗi đó nổi lên (lỗi thường): job gốc lỗi, còn lượt thì thử lại.
            if permanent or is_final_attempt(bull_job):
                await use_case.report_failure(req, err)
            if permanent:
                raise UnrecoverableError(f"invalid render job: {err}") from err
            raise err

        resp = result.unwrap()
        return {
            "micro_content_id": resp.micro_content_id,
            "video_object_key": resp.video_object_key,
            "duration": resp.duration_seconds,
            "size": resp.size_bytes,
            "asset_id": resp.asset_id,
            "render_id": resp.render_id,
        }

    if event_type == "MICRO_CONTENT_SCRIPT_REQUESTED":
        logger.info(
            "micro_content_video_worker.script_requested_ignored",
            micro_content_id=bull_job.data.get("aggregate_id"),
        )
        return {"status": "ignored"}

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
