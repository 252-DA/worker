from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from document_chunk.domain.ports.file_storage import IFileStorage
from document_chunk.shared.logger import get_logger
from document_chunk.shared.result import Err, Ok, Result

from worker.services.micro_content_video_store import MicroContentVideoStore
from worker.services.slide_generator import SlideGenerator
from worker.services.tts_client import TtsClient, get_default_tts_client
from worker.services.video_composer import SceneMedia, VideoComposer

logger = get_logger(__name__)


@dataclass(frozen=True)
class RenderMicroContentVideoRequest:
    micro_content_id: str
    course_id: str | None = None
    script: dict | None = None


@dataclass(frozen=True)
class RenderMicroContentVideoResponse:
    micro_content_id: str
    video_object_key: str
    duration_seconds: float
    size_bytes: int


class RenderMicroContentVideoUseCase:
    def __init__(
        self,
        store: MicroContentVideoStore,
        file_storage: IFileStorage,
        tts_client: TtsClient | None = None,
        slide_generator: SlideGenerator | None = None,
        video_composer: VideoComposer | None = None,
    ) -> None:
        self.store = store
        self.file_storage = file_storage
        self.tts_client = tts_client or get_default_tts_client()
        self.slide_generator = slide_generator or SlideGenerator()
        self.video_composer = video_composer or VideoComposer()

    async def execute(
        self, request: RenderMicroContentVideoRequest
    ) -> Result[RenderMicroContentVideoResponse, Exception]:
        mc_res = self.store.get_micro_content(request.micro_content_id)
        if mc_res.is_err():
            return Err(mc_res.error)

        mc = mc_res.unwrap()
        script = request.script or mc.ai_script
        if not script or not isinstance(script, dict):
            err = ValueError(f"No valid script found for micro content {request.micro_content_id}")
            self.store.update_video_failed(request.micro_content_id, str(err))
            return Err(err)

        scenes = script.get("scenes") or []
        if not isinstance(scenes, list) or len(scenes) == 0:
            err = ValueError("Script contains no scenes to render")
            self.store.update_video_failed(request.micro_content_id, str(err))
            return Err(err)

        course_id = request.course_id or mc.course_id

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                scene_media_list: list[SceneMedia] = []
                total_scenes = len(scenes)

                for idx, scene in enumerate(scenes):
                    title = scene.get("title", f"Cảnh {idx + 1}")
                    bullets = scene.get("bullets", [])
                    narration = scene.get("narration") or title

                    slide_img_path = os.path.join(temp_dir, f"slide_{idx:02d}.png")
                    audio_path = os.path.join(temp_dir, f"audio_{idx:02d}.mp3")

                    self.slide_generator.generate_slide(
                        title=title,
                        bullets=bullets,
                        output_path=slide_img_path,
                        scene_index=idx + 1,
                        total_scenes=total_scenes,
                    )

                    duration = await self.tts_client.synthesize(
                        text=narration,
                        output_path=audio_path,
                    )

                    scene_media_list.append(
                        SceneMedia(
                            slide_image_path=slide_img_path,
                            audio_path=audio_path,
                            duration_seconds=duration,
                        )
                    )

                final_mp4_path = os.path.join(temp_dir, "output.mp4")
                total_duration = await self.video_composer.compose_video(
                    scenes=scene_media_list,
                    work_dir=os.path.join(temp_dir, "clips"),
                    output_mp4_path=final_mp4_path,
                )

                object_key = f"micro-content/{course_id}/{request.micro_content_id}/video.mp4"
                upload_res = self.file_storage.upload(Path(final_mp4_path), object_key)
                if upload_res.is_err():
                    raise upload_res.error

                size_bytes = os.path.getsize(final_mp4_path)

                store_res = self.store.update_video_ready(
                    micro_content_id=request.micro_content_id,
                    object_key=object_key,
                    duration_seconds=total_duration,
                    size_bytes=size_bytes,
                )
                if store_res.is_err():
                    raise store_res.error

                logger.info(
                    "render_micro_content_video.succeeded",
                    micro_content_id=request.micro_content_id,
                    object_key=object_key,
                    duration=total_duration,
                    size_bytes=size_bytes,
                )

                return Ok(
                    RenderMicroContentVideoResponse(
                        micro_content_id=request.micro_content_id,
                        video_object_key=object_key,
                        duration_seconds=total_duration,
                        size_bytes=size_bytes,
                    )
                )

        except Exception as exc:
            logger.error(
                "render_micro_content_video.failed",
                micro_content_id=request.micro_content_id,
                error=str(exc),
            )
            self.store.update_video_failed(request.micro_content_id, str(exc))
            return Err(exc)
