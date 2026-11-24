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
    video_clip_path: str | None = None


@dataclass(frozen=True)
class ComposeResult:
    duration_seconds: float
    scene_durations_seconds: list[float]


class VideoComposer:
    """Ghép ảnh slide và âm thanh đọc thành video MP4 hoàn chỉnh với FFmpeg."""

    def __init__(self, ffmpeg_bin: str = "ffmpeg", ffprobe_bin: str = "ffprobe") -> None:
        self.ffmpeg_bin = ffmpeg_bin
        self.ffprobe_bin = ffprobe_bin

    def _ensure_binaries(self) -> None:
        if not shutil.which(self.ffmpeg_bin):
            raise RuntimeError(f"FFmpeg binary not found at '{self.ffmpeg_bin}'")

    async def render_scene_clip(self, scene: SceneMedia, output_clip_path: str) -> float:
        """
        Render 1 cảnh đơn lẻ thành clip MP4 khớp thời lượng hình và tiếng.
        Hỗ trợ cả:
        - Cảnh Slide tĩnh: loop ảnh slide với giọng đọc TTS.
        - Cảnh Video động (Manim): chuẩn hóa độ phân giải 1280x720 25fps,
          ràng buộc tốc độ trong [0.75x, 1.35x], tpad đóng băng frame cuối nếu thiếu,
          bỏ track audio gốc của clip Manim và thay bằng giọng đọc TTS.
        """
        import math

        os.makedirs(os.path.dirname(output_clip_path), exist_ok=True)
        audio_duration = max(0.5, float(scene.duration_seconds or 0.0))
        frames = max(13, math.ceil((audio_duration + 0.4) * 25))
        clip_duration = frames / 25.0
        dur_str = f"{clip_duration:.2f}"

        # Bộ lọc chuẩn hóa hình ảnh thống nhất 1280x720, SAR=1, 25fps, yuv420p
        normalize_vf = (
            "scale=1280:720:force_original_aspect_ratio=decrease,"
            "pad=1280:720:(ow-iw)/2:(oh-ih)/2:black,"
            "setsar=1,fps=25,format=yuv420p"
        )
        audio_filter = f"apad,atrim=0:{dur_str},asetpts=PTS-STARTPTS"

        if scene.video_clip_path and os.path.exists(scene.video_clip_path):
            # Nhánh Cảnh dùng Video Clip (Manim / Hoạt họa động)
            raw_clip_dur = await self._probe_duration(scene.video_clip_path, audio_duration)
            raw_speed = raw_clip_dur / audio_duration if audio_duration > 0 else 1.0

            # Ràng buộc điều chỉnh tốc độ trong khoảng an toàn [0.75x, 1.35x]
            clamped_speed = max(0.75, min(1.35, raw_speed))
            pts_factor = 1.0 / clamped_speed

            if raw_speed > 1.35:
                logger.warning(
                    "video_composer.clip_longer_than_audio_capped",
                    clip_path=scene.video_clip_path,
                    clip_duration=raw_clip_dur,
                    audio_duration=audio_duration,
                    raw_speed=round(raw_speed, 2),
                    clamped_speed=1.35,
                )
            elif raw_speed < 0.75:
                logger.info(
                    "video_composer.clip_shorter_than_audio_frozen",
                    clip_path=scene.video_clip_path,
                    clip_duration=raw_clip_dur,
                    audio_duration=audio_duration,
                    raw_speed=round(raw_speed, 2),
                    clamped_speed=0.75,
                )

            # Tính thời lượng video sau khi co giãn tốc độ (duration_scaled = raw_clip_dur * pts_factor)
            duration_scaled = raw_clip_dur * pts_factor
            # Khớp tuyệt đối với thời lượng luồng audio sau đệm an toàn (clip_duration) để tránh lệch WebVTT khi concat
            pad_needed = max(0.0, clip_duration - duration_scaled)
            tpad_filter = f",tpad=stop_mode=clone:stop_duration={pad_needed:.3f}" if pad_needed > 0.02 else ""

            # setpts điều chỉnh tốc độ + tpad đóng băng frame cuối nếu thiếu + scale pad chuẩn
            vf = f"setpts={pts_factor:.6f}*PTS{tpad_filter},{normalize_vf}"

            cmd = [
                self.ffmpeg_bin,
                "-y",
                "-i",
                scene.video_clip_path,
                "-i",
                scene.audio_path,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-t",
                dur_str,
                "-vf",
                vf,
                "-af",
                audio_filter,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-r",
                "25",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                "44100",
                "-ac",
                "2",
                output_clip_path,
            ]
        else:
            # Nhánh Cảnh Slide tĩnh (Static Slide Image)
            cmd = [
                self.ffmpeg_bin,
                "-y",
                "-loop",
                "1",
                "-framerate",
                "25",
                "-t",
                dur_str,
                "-i",
                scene.slide_image_path,
                "-i",
                scene.audio_path,
                "-t",
                dur_str,
                "-vf",
                normalize_vf,
                "-af",
                audio_filter,
                "-c:v",
                "libx264",
                "-tune",
                "stillimage",
                "-pix_fmt",
                "yuv420p",
                "-r",
                "25",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                "44100",
                "-ac",
                "2",
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
        return await self._probe_duration(output_clip_path, clip_duration)

    async def _probe_duration(self, media_path: str, fallback: float) -> float:
        """Đo thời lượng thực tế của file media bằng ffprobe."""
        if not shutil.which(self.ffprobe_bin):
            return fallback
        try:
            proc = await asyncio.create_subprocess_exec(
                self.ffprobe_bin,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                media_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode == 0:
                val = float(stdout.decode().strip())
                if val > 0:
                    return round(val, 3)
        except Exception:
            pass
        return fallback

    async def concat_clips(self, clip_paths: Sequence[str], final_output_path: str) -> None:
        """Ghép nối nhiều clip cảnh, encode lại một luồng AAC liên tục và +faststart."""
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
        concat_list_path = os.path.join(os.path.dirname(final_output_path), "concat_list.txt")

        with open(concat_list_path, "w", encoding="utf-8") as f:
            for clip in clip_paths:
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
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
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

    async def compose_video_with_timeline(
        self,
        scenes: Sequence[SceneMedia],
        work_dir: str,
        output_mp4_path: str,
    ) -> ComposeResult:
        """Render từng cảnh, nối thành MP4 và đo thời lượng thực tế từng cảnh lẫn video cuối."""
        self._ensure_binaries()
        os.makedirs(work_dir, exist_ok=True)
        clip_paths: list[str] = []
        scene_durations: list[float] = []

        for idx, scene in enumerate(scenes):
            clip_path = os.path.join(work_dir, f"scene_{idx:02d}.mp4")
            clip_dur = await self.render_scene_clip(scene, clip_path)
            clip_paths.append(clip_path)
            scene_durations.append(clip_dur)

        await self.concat_clips(clip_paths, output_mp4_path)
        fallback_total = round(sum(scene_durations), 3)
        total_duration = await self._probe_duration(output_mp4_path, fallback_total)
        logger.info(
            "video_composer.composed_successfully",
            output=output_mp4_path,
            total_duration=total_duration,
            scene_count=len(scenes),
        )
        return ComposeResult(
            duration_seconds=total_duration,
            scene_durations_seconds=scene_durations,
        )

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
        result = await self.compose_video_with_timeline(scenes, work_dir, output_mp4_path)
        return result.duration_seconds
