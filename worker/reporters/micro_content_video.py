"""
MICRO_CONTENT_VIDEO_RESULT: kết quả một lần render video Micro Content (READY kèm
object key, hoặc FAILED) → core-api
(contracts/integration-events/v1/micro-content-video-result.json).

``result_id`` tất định theo ``render_id`` + trạng thái: job gốc retry sau khi đã
gửi sinh lại đúng id, BullMQ bỏ bản trùng. core-api bỏ kết quả của lần render cũ
(so ``render_id``) và không cho READY đè FAILED của cùng lần render.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from worker.adapters.result_publisher import MICRO_CONTENT_VIDEO_RESULT, make_result_id

if TYPE_CHECKING:
    from worker.use_cases.render_micro_content_video import (
        RenderMicroContentVideoRequest,
        RenderMicroContentVideoResponse,
    )

VIDEO_MIME_TYPE = "video/mp4"
# Khớp maxLength của `error` trong micro-content-video-result.json (trước đây
# store cắt cùng độ dài khi ghi micro_contents.video_error).
MAX_ERROR_CHARS = 1000


class ResultPublisher(Protocol):
    async def publish(
        self, kind: str, payload: dict[str, Any], *, result_id: str, source_event_id: str | None = None
    ) -> str: ...


class MicroContentVideoReporter:
    def __init__(self, publisher: ResultPublisher) -> None:
        self._publisher = publisher

    async def ready(
        self, request: RenderMicroContentVideoRequest, response: RenderMicroContentVideoResponse
    ) -> str:
        payload: dict[str, Any] = {
            "micro_content_id": response.micro_content_id,
            "asset_id": response.asset_id,
            "render_id": response.render_id,
            "status": "READY",
            "object_key": response.video_object_key,
            "duration_ms": max(0, int(response.duration_seconds * 1000)),
            "size_bytes": response.size_bytes,
            "mime_type": VIDEO_MIME_TYPE,
        }
        if response.vtt_object_key:
            payload["vtt_object_key"] = response.vtt_object_key
        if response.transcript_text:
            payload["transcript_text"] = response.transcript_text
        if response.scene_timeline:
            payload["scene_timeline"] = response.scene_timeline
        return await self._publisher.publish(
            MICRO_CONTENT_VIDEO_RESULT,
            payload,
            result_id=make_result_id(MICRO_CONTENT_VIDEO_RESULT, request.render_id, "ready"),
            source_event_id=request.source_event_id,
        )

    async def failed(self, request: RenderMicroContentVideoRequest, error: BaseException | str) -> str:
        message = (str(error) or type(error).__name__)[:MAX_ERROR_CHARS]
        return await self._publisher.publish(
            MICRO_CONTENT_VIDEO_RESULT,
            {
                "micro_content_id": request.micro_content_id,
                "asset_id": request.asset_id,
                "render_id": request.render_id,
                "status": "FAILED",
                "error": message,
            },
            result_id=make_result_id(MICRO_CONTENT_VIDEO_RESULT, request.render_id, "failed"),
            source_event_id=request.source_event_id,
        )


__all__ = ["MAX_ERROR_CHARS", "VIDEO_MIME_TYPE", "MicroContentVideoReporter", "ResultPublisher"]
