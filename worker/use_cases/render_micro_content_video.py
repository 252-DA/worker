"""
Render video AI cho Micro Content: slide + giọng đọc + ghép FFmpeg → object
storage, rồi gửi MICRO_CONTENT_VIDEO_RESULT về core-api.

Worker không ghi ``micro_contents`` hay ``video_assets``: core-api áp dụng kết
quả (core-api/src/micro-content/micro-content-video-result.handler.ts). Lệnh
MICRO_CONTENT_RENDER_REQUESTED mang đủ dữ liệu (kịch bản, asset, render_id) nên
cũng không đọc DB. Xem docs/ke-hoach-ranh-gioi-du-lieu.md.
"""
from __future__ import annotations

import inspect
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from document_chunk.domain.ports.file_storage import IFileStorage
from document_chunk.shared.logger import get_logger
from document_chunk.shared.result import Err, Ok, Result

from worker.services.micro_content_subtitles import (
    SubtitleCue,
    build_webvtt_and_timeline,
    normalize_for_tts,
)
from worker.services.result_publisher import MICRO_CONTENT_VIDEO_RESULT, make_result_id
from worker.services.slide_generator import SlideGenerator, render_pdf_slide_page
from worker.services.tts_client import TtsClient, get_default_tts_client
from worker.services.video_composer import SceneMedia, VideoComposer

logger = get_logger(__name__)

DEFAULT_VOICE = "vi-VN-HoaiMyNeural"
VIDEO_MIME_TYPE = "video/mp4"
# Khớp maxLength của `error` trong micro-content-video-result.json (trước đây
# store cắt cùng độ dài khi ghi micro_contents.video_error).
MAX_ERROR_CHARS = 1000


class ResultPublisher(Protocol):
    async def publish(
        self, kind: str, payload: dict[str, Any], *, result_id: str, source_event_id: str | None = None
    ) -> str: ...


class InvalidRenderRequest(ValueError):
    """Lệnh render không render được (không có kịch bản/cảnh): thử lại cũng vậy."""


@dataclass(frozen=True)
class RenderMicroContentVideoRequest:
    micro_content_id: str
    course_id: str
    asset_id: str
    # core-api sinh mới mỗi lần bấm render; kết quả mang lại để core-api bỏ
    # kết quả của lần render cũ.
    render_id: str
    script: Any = None
    source_document_key: str | None = None
    source_document_mime_type: str | None = None
    # event_id outbox của lệnh render.
    source_event_id: str | None = None


@dataclass(frozen=True)
class RenderMicroContentVideoResponse:
    micro_content_id: str
    asset_id: str
    render_id: str
    video_object_key: str
    duration_seconds: float
    size_bytes: int
    vtt_object_key: str | None = None
    transcript_text: str | None = None
    scene_timeline: list[dict[str, Any]] | None = None


def video_object_key(course_id: str, micro_content_id: str, render_id: str) -> str:
    """
    Mỗi lần render một object riêng. Trước đây mọi lần render (và cả video tải
    lên) cùng ghi ``.../video.mp4``: job cũ retry xong muộn sẽ ghi đè file của
    lần render mới, dù DB đã chặn kết quả cũ. Bài khác hay bản fork đang trỏ file
    cũ cũng vẫn phát được file của nó.
    """
    return f"micro-content/{course_id}/{micro_content_id}/ai-{render_id}.mp4"


def vtt_object_key(course_id: str, micro_content_id: str, render_id: str) -> str:
    return f"micro-content/{course_id}/{micro_content_id}/ai-{render_id}.vtt"


def _scene_slide_page(scene: dict[str, Any]) -> int | None:
    raw_page = scene.get("slide_page")
    if isinstance(raw_page, int) and not isinstance(raw_page, bool) and raw_page >= 1:
        return raw_page
    source_pages = scene.get("source_pages")
    if isinstance(source_pages, list) and source_pages:
        first = source_pages[0]
        if isinstance(first, int) and not isinstance(first, bool) and first >= 1:
            return first
    return None


def _is_pdf_document(key: str | None, mime: str | None) -> bool:
    if not key:
        return False
    if mime and mime.lower() == "application/pdf":
        return True
    return key.lower().endswith(".pdf")


def _scenes_of(script: Any) -> tuple[list[dict], str]:
    if not script or not isinstance(script, dict):
        raise InvalidRenderRequest("No valid script found for this render request")
    scenes = script.get("scenes") or []
    if not isinstance(scenes, list) or len(scenes) == 0:
        raise InvalidRenderRequest("Script contains no scenes to render")
    if not all(isinstance(scene, dict) for scene in scenes):
        raise InvalidRenderRequest("Every scene of the script must be an object")
    return scenes, script.get("voice") or DEFAULT_VOICE


class RenderMicroContentVideoUseCase:
    def __init__(
        self,
        result_publisher: ResultPublisher,
        file_storage: IFileStorage,
        tts_client: TtsClient | None = None,
        slide_generator: SlideGenerator | None = None,
        video_composer: VideoComposer | None = None,
    ) -> None:
        self.result_publisher = result_publisher
        self.file_storage = file_storage
        self.tts_client = tts_client or get_default_tts_client()
        self.slide_generator = slide_generator or SlideGenerator()
        self.video_composer = video_composer or VideoComposer()

    async def execute(
        self, request: RenderMicroContentVideoRequest
    ) -> Result[RenderMicroContentVideoResponse, Exception]:
        """
        Render, upload rồi gửi kết quả READY. Lỗi trả về Err và KHÔNG gửi FAILED:
        job gốc còn lượt thử thì người gọi để BullMQ thử lại, chỉ lần cuối (hoặc
        ``InvalidRenderRequest``) mới gọi ``report_failure``.
        """
        try:
            scenes, script_voice = _scenes_of(request.script)
        except InvalidRenderRequest as exc:
            return Err(exc)

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                scene_media_list: list[SceneMedia] = []
                scene_cues: list[list[SubtitleCue]] = []
                speech_durations_ms: list[int] = []
                total_scenes = len(scenes)
                use_cue_tts = inspect.iscoroutinefunction(
                    getattr(type(self.tts_client), "synthesize_with_cues", None)
                )
                local_pdf_path: Path | None = None
                if (
                    _is_pdf_document(request.source_document_key, request.source_document_mime_type)
                    and any(_scene_slide_page(sc) is not None for sc in scenes)
                ):
                    candidate_pdf = Path(temp_dir) / "source.pdf"
                    try:
                        dl_res = self.file_storage.download(request.source_document_key, candidate_pdf)  # type: ignore[arg-type]
                        if dl_res.is_ok() and candidate_pdf.exists():
                            local_pdf_path = candidate_pdf
                        else:
                            logger.warning(
                                "render_micro_content_video.pdf_download_fallback",
                                key=request.source_document_key,
                                error=str(dl_res.error) if dl_res.is_err() else "missing file",
                            )
                    except Exception as exc:
                        logger.warning(
                            "render_micro_content_video.pdf_download_error",
                            key=request.source_document_key,
                            error=str(exc),
                        )

                slide_width = getattr(self.slide_generator, "width", 1280)
                slide_height = getattr(self.slide_generator, "height", 720)

                for idx, scene in enumerate(scenes):
                    title = scene.get("title", f"Cảnh {idx + 1}")
                    bullets = scene.get("bullets", [])
                    narration = scene.get("narration") or title
                    voice = scene.get("voice") or script_voice
                    slide_page = _scene_slide_page(scene)

                    slide_img_path = os.path.join(temp_dir, f"slide_{idx:02d}.png")
                    audio_path = os.path.join(temp_dir, f"audio_{idx:02d}.mp3")

                    rendered_from_pdf = False
                    if local_pdf_path is not None and slide_page is not None:
                        try:
                            render_pdf_slide_page(
                                str(local_pdf_path),
                                slide_page,
                                slide_img_path,
                                width=slide_width,
                                height=slide_height,
                            )
                            rendered_from_pdf = True
                        except Exception as exc:
                            logger.warning(
                                "render_micro_content_video.pdf_page_fallback",
                                slide_page=slide_page,
                                error=str(exc),
                            )

                    if not rendered_from_pdf:
                        self.slide_generator.generate_slide(
                            title=title,
                            bullets=bullets,
                            output_path=slide_img_path,
                            scene_index=idx + 1,
                            total_scenes=total_scenes,
                        )

                    tts_text = normalize_for_tts(narration) or narration
                    if use_cue_tts:
                        tts_result = await self.tts_client.synthesize_with_cues(  # type: ignore[attr-defined]
                            text=tts_text,
                            output_path=audio_path,
                            voice=voice,
                        )
                        duration = float(tts_result.duration_seconds)
                        scene_cues.append(list(tts_result.cues))
                    else:
                        duration = await self.tts_client.synthesize(
                            text=tts_text,
                            output_path=audio_path,
                            voice=voice,
                        )
                        scene_cues.append([])

                    speech_durations_ms.append(max(400, round(duration * 1000)))
                    scene_media_list.append(
                        SceneMedia(
                            slide_image_path=slide_img_path,
                            audio_path=audio_path,
                            duration_seconds=duration,
                        )
                    )

                final_mp4_path = os.path.join(temp_dir, "output.mp4")
                use_timeline_composer = inspect.iscoroutinefunction(
                    getattr(type(self.video_composer), "compose_video_with_timeline", None)
                )
                if use_timeline_composer:
                    composed = await self.video_composer.compose_video_with_timeline(  # type: ignore[attr-defined]
                        scenes=scene_media_list,
                        work_dir=os.path.join(temp_dir, "clips"),
                        output_mp4_path=final_mp4_path,
                    )
                    total_duration = float(composed.duration_seconds)
                    scene_durations_ms = [
                        max(400, round(d * 1000)) for d in composed.scene_durations_seconds
                    ]
                else:
                    total_duration = await self.video_composer.compose_video(
                        scenes=scene_media_list,
                        work_dir=os.path.join(temp_dir, "clips"),
                        output_mp4_path=final_mp4_path,
                    )
                    scene_durations_ms = list(speech_durations_ms)

                vtt_text, transcript_text, timeline = build_webvtt_and_timeline(
                    scenes=scenes,
                    scene_durations_ms=scene_durations_ms,
                    scene_cues=scene_cues,
                    speech_durations_ms=speech_durations_ms,
                )
                vtt_key: str | None = None
                if vtt_text.strip():
                    vtt_path = os.path.join(temp_dir, "subtitles.vtt")
                    Path(vtt_path).write_text(vtt_text, encoding="utf-8")
                    candidate_vtt_key = vtt_object_key(
                        request.course_id, request.micro_content_id, request.render_id
                    )
                    vtt_res = self.file_storage.upload(Path(vtt_path), candidate_vtt_key)
                    if vtt_res.is_err():
                        raise vtt_res.error
                    vtt_key = candidate_vtt_key

                object_key = video_object_key(
                    request.course_id, request.micro_content_id, request.render_id
                )
                upload_res = self.file_storage.upload(Path(final_mp4_path), object_key)
                if upload_res.is_err():
                    raise upload_res.error

                size_bytes = os.path.getsize(final_mp4_path)

            response = RenderMicroContentVideoResponse(
                micro_content_id=request.micro_content_id,
                asset_id=request.asset_id,
                render_id=request.render_id,
                video_object_key=object_key,
                duration_seconds=total_duration,
                size_bytes=size_bytes,
                vtt_object_key=vtt_key,
                transcript_text=transcript_text,
                scene_timeline=timeline,
            )
            # Chỉ báo READY khi file đã nằm trên storage. Gửi lỗi thì nổi lên
            # thành Err: job gốc lỗi và BullMQ render lại, không hoàn tất khi
            # core-api chưa nhận được kết quả.
            await self._publish_ready(request, response)

            logger.info(
                "render_micro_content_video.succeeded",
                micro_content_id=request.micro_content_id,
                render_id=request.render_id,
                object_key=object_key,
                duration=total_duration,
                size_bytes=size_bytes,
                asset_id=request.asset_id,
                voice=script_voice,
            )
            return Ok(response)

        except Exception as exc:
            logger.error(
                "render_micro_content_video.failed",
                micro_content_id=request.micro_content_id,
                render_id=request.render_id,
                error=str(exc),
            )
            return Err(exc)

    async def report_failure(self, request: RenderMicroContentVideoRequest, error: BaseException | str) -> str:
        """
        Gửi FAILED cho lần render này (micro_contents.video_error hiện cho giảng
        viên). Lỗi gửi nổi lên cho người gọi: job gốc phải lỗi theo.
        """
        message = (str(error) or type(error).__name__)[:MAX_ERROR_CHARS]
        return await self.result_publisher.publish(
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

    async def _publish_ready(
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
        return await self.result_publisher.publish(
            MICRO_CONTENT_VIDEO_RESULT,
            payload,
            # Tất định theo lần render: job gốc retry sau khi đã gửi sinh lại
            # đúng id, BullMQ bỏ bản trùng.
            result_id=make_result_id(MICRO_CONTENT_VIDEO_RESULT, request.render_id, "ready"),
            source_event_id=request.source_event_id,
        )
