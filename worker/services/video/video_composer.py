from __future__ import annotations

import asyncio
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from document_chunk.shared.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SceneMedia:
    slide_image_path: str
    audio_path: str
    duration_seconds: float
    slide_state_paths: tuple[str, ...] = ()
    reveal_times_seconds: tuple[float, ...] = ()
    motion: Literal["none", "zoom_in"] = "zoom_in"
    animation_path: str | None = None


@dataclass(frozen=True)
class ComposeResult:
    duration_seconds: float
    scene_durations_seconds: list[float]


class VideoComposer:
    """Ghép slide hoặc animation Manim với âm thanh đọc thành video MP4 1080p."""

    def __init__(
        self,
        ffmpeg_bin: str = "ffmpeg",
        ffprobe_bin: str = "ffprobe",
        *,
        width: int = 1920,
        height: int = 1080,
        fps: int = 25,
    ) -> None:
        self.ffmpeg_bin = ffmpeg_bin
        self.ffprobe_bin = ffprobe_bin
        self.width = width
        self.height = height
        self.fps = fps

    def _ensure_binaries(self) -> None:
        if not shutil.which(self.ffmpeg_bin):
            raise RuntimeError(f"FFmpeg binary not found at '{self.ffmpeg_bin}'")

    def _visual_inputs(
        self,
        scene: SceneMedia,
        *,
        clip_duration: float,
        frames: int,
    ) -> tuple[list[str], str, int]:
        if scene.animation_path:
            animation_filter = (
                f"[0:v]fps={self.fps},scale={self.width}:{self.height}:"
                "force_original_aspect_ratio=decrease,"
                f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2:color=0x0f172a,setsar=1,"
                "tpad=stop_mode=clone:stop_duration=1,"
                f"trim=duration={clip_duration:.3f},setpts=PTS-STARTPTS,format=yuv420p[video]"
            )
            return ["-i", scene.animation_path], animation_filter, 1

        paths = list(scene.slide_state_paths) or [scene.slide_image_path]
        if len(paths) > 1 and len(scene.reveal_times_seconds) != len(paths) - 1:
            raise ValueError("Số mốc hiện bullet phải nhỏ hơn số trạng thái slide một đơn vị")

        input_args: list[str] = []
        if len(paths) == 1:
            input_args.extend(
                [
                    "-loop",
                    "1",
                    "-framerate",
                    str(self.fps),
                    "-t",
                    f"{clip_duration:.3f}",
                    "-i",
                    paths[0],
                ]
            )
            video_source = "[0:v]"
        else:
            boundaries = [0.0, *scene.reveal_times_seconds, clip_duration]
            for idx, path in enumerate(paths):
                state_duration = max(1.0 / self.fps, boundaries[idx + 1] - boundaries[idx])
                input_args.extend(
                    [
                        "-loop",
                        "1",
                        "-framerate",
                        str(self.fps),
                        "-t",
                        f"{state_duration:.3f}",
                        "-i",
                        path,
                    ]
                )
            sources = "".join(f"[{idx}:v]" for idx in range(len(paths)))
            video_source = "[reveal]"
            reveal_filter = f"{sources}concat=n={len(paths)}:v=1:a=0{video_source}"
        if len(paths) == 1:
            reveal_filter = ""

        normalization = (
            f"fps={self.fps},scale={self.width}:{self.height}:"
            "force_original_aspect_ratio=decrease,"
            f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2:color=0x0f172a,setsar=1"
        )
        if scene.motion == "zoom_in":
            zoom_frames = max(1, frames - 1)
            animated = (
                f"scale={round(self.width * 1.2)}:{round(self.height * 1.2)},"
                f"zoompan=z='min(1+0.025*on/{zoom_frames},1.025)':"
                "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d=1:s={self.width}x{self.height}:fps={self.fps},setsar=1"
            )
        else:
            animated = normalization
        video_filter = f"{video_source}{animated},format=yuv420p[video]"
        if reveal_filter:
            video_filter = f"{reveal_filter};{video_filter}"
        return input_args, video_filter, len(paths)

    async def render_scene_clip(self, scene: SceneMedia, output_clip_path: str) -> float:
        """Render một cảnh, hiện dần bullet nếu có và khớp thời lượng hình với tiếng."""
        os.makedirs(os.path.dirname(output_clip_path), exist_ok=True)
        audio_duration = max(0.5, float(scene.duration_seconds or 0.0))
        frames = max(13, math.ceil((audio_duration + 0.4) * self.fps))
        clip_duration = frames / float(self.fps)
        dur_str = f"{clip_duration:.3f}"
        visual_inputs, video_filter, audio_index = self._visual_inputs(
            scene,
            clip_duration=clip_duration,
            frames=frames,
        )
        cmd = [
            self.ffmpeg_bin,
            "-y",
            *visual_inputs,
            "-i",
            scene.audio_path,
            "-filter_complex",
            video_filter,
            "-map",
            "[video]",
            "-map",
            f"{audio_index}:a",
            "-t",
            dur_str,
            "-af",
            f"apad,atrim=0:{dur_str},asetpts=PTS-STARTPTS",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "19",
            "-tune",
            "animation" if scene.animation_path else "stillimage",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(self.fps),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
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
        _, stderr = await proc.communicate()
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
        """Ghép các cảnh, chuẩn hóa âm lượng toàn video và tối ưu phát trực tuyến."""
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
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-ac",
            "2",
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
        _, stderr = await proc.communicate()
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
        """Nhận danh sách SceneMedia, dựng video và trả về tổng thời lượng giây."""
        result = await self.compose_video_with_timeline(scenes, work_dir, output_mp4_path)
        return result.duration_seconds
