from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from document_chunk.shared.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SceneMedia:
    slide_image_path: str
    audio_path: str
    duration_seconds: float


class VideoComposer:
    """Ghép ảnh slide và âm thanh đọc thành video MP4 hoàn chỉnh với FFmpeg."""

    def __init__(self, ffmpeg_bin: str = "ffmpeg", ffprobe_bin: str = "ffprobe") -> None:
        self.ffmpeg_bin = ffmpeg_bin
        self.ffprobe_bin = ffprobe_bin

    def _ensure_binaries(self) -> None:
        if not shutil.which(self.ffmpeg_bin):
            raise RuntimeError(f"FFmpeg binary not found at '{self.ffmpeg_bin}'")

    async def render_scene_clip(self, scene: SceneMedia, output_clip_path: str) -> None:
        """Render 1 cảnh đơn lẻ thành clip MP4 từ ảnh tĩnh và audio."""
        os.makedirs(os.path.dirname(output_clip_path), exist_ok=True)
        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-loop",
            "1",
            "-i",
            scene.slide_image_path,
            "-i",
            scene.audio_path,
            "-c:v",
            "libx264",
            "-tune",
            "stillimage",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            output_clip_path,
        ]

        logger.debug("video_composer.rendering_scene", cmd=" ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            err_msg = stderr.decode("utf-8", errors="replace")
            logger.error("video_composer.scene_render_failed", error=err_msg)
            raise RuntimeError(f"FFmpeg scene render failed: {err_msg[:500]}")

    async def concat_clips(self, clip_paths: Sequence[str], final_output_path: str) -> None:
        """Ghép nối nhiều clip cảnh lại với nhau và thêm -movflags +faststart."""
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
        concat_list_path = os.path.join(os.path.dirname(final_output_path), "concat_list.txt")

        with open(concat_list_path, "w", encoding="utf-8") as f:
            for clip in clip_paths:
                # Use absolute path normalized with forward slashes for ffmpeg concat
                safe_path = Path(clip).resolve().as_posix()
                f.write(f"file '{safe_path}'\n")

        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list_path,
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            final_output_path,
        ]

        logger.debug("video_composer.concatenating", cmd=" ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            err_msg = stderr.decode("utf-8", errors="replace")
            logger.error("video_composer.concat_failed", error=err_msg)
            raise RuntimeError(f"FFmpeg concat failed: {err_msg[:500]}")

    async def compose_video(
        self,
        scenes: Sequence[SceneMedia],
        work_dir: str,
        output_mp4_path: str,
    ) -> float:
        """
        Nhận danh sách SceneMedia, render từng clip cảnh và nối thành file MP4 hoàn chỉnh.
        Trả về tổng thời lượng (giây).
        """
        self._ensure_binaries()
        os.makedirs(work_dir, exist_ok=True)
        clip_paths: list[str] = []
        total_duration = 0.0

        for idx, scene in enumerate(scenes):
            clip_path = os.path.join(work_dir, f"scene_{idx:02d}.mp4")
            await self.render_scene_clip(scene, clip_path)
            clip_paths.append(clip_path)
            total_duration += scene.duration_seconds

        await self.concat_clips(clip_paths, output_mp4_path)
        logger.info(
            "video_composer.composed_successfully",
            output=output_mp4_path,
            total_duration=total_duration,
            scene_count=len(scenes),
        )
        return total_duration
