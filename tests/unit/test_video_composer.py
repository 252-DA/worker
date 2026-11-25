from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from worker.services.video_composer import SceneMedia, VideoComposer


def _arg_value(cmd: tuple[str, ...], flag: str) -> str:
    idx = cmd.index(flag)
    return cmd[idx + 1]


@pytest.mark.asyncio
async def test_short_narration_preserves_full_manim_animation(tmp_path: Path) -> None:
    """Khi lời đọc quá ngắn (~2.52s) so với hoạt họa QuickSort (~30.96s), giữ tốc độ 1.0x và không cắt cụt video."""
    clip = tmp_path / "quicksort.mp4"
    clip.write_bytes(b"mp4")
    slide = tmp_path / "slide.png"
    slide.write_bytes(b"png")
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"mp3")
    out = tmp_path / "out.mp4"

    composer = VideoComposer()
    captured_cmds: list[tuple[str, ...]] = []

    async def fake_subproc(*cmd: str, **kwargs):
        captured_cmds.append(cmd)
        proc = MagicMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"", b""))
        return proc

    probe_mock = AsyncMock(side_effect=[30.96, 31.36])
    with (
        patch.object(composer, "_probe_duration", probe_mock),
        patch("worker.services.video_composer.asyncio.create_subprocess_exec", side_effect=fake_subproc),
    ):
        dur = await composer.render_scene_clip(
            SceneMedia(
                slide_image_path=str(slide),
                audio_path=str(audio),
                duration_seconds=2.52,
                video_clip_path=str(clip),
            ),
            str(out),
        )

    assert dur == pytest.approx(31.36)
    cmd = captured_cmds[0]
    assert _arg_value(cmd, "-t") == "31.36"
    assert "setpts=1.000000*PTS" in _arg_value(cmd, "-vf")
    assert "apad,atrim=0:31.36" in _arg_value(cmd, "-af")


@pytest.mark.asyncio
async def test_moderate_narration_caps_speed_at_135_and_extends_duration(tmp_path: Path) -> None:
    """Khi raw_speed = 1.55 <= 2.0, tăng tốc tối đa 1.35x và nới thời lượng cảnh để phát trọn hoạt họa."""
    clip = tmp_path / "quicksort.mp4"
    clip.write_bytes(b"mp4")
    slide = tmp_path / "slide.png"
    slide.write_bytes(b"png")
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"mp3")
    out = tmp_path / "out.mp4"

    composer = VideoComposer()
    captured_cmds: list[tuple[str, ...]] = []

    async def fake_subproc(*cmd: str, **kwargs):
        captured_cmds.append(cmd)
        proc = MagicMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"", b""))
        return proc

    probe_mock = AsyncMock(side_effect=[30.96, 23.36])
    with (
        patch.object(composer, "_probe_duration", probe_mock),
        patch("worker.services.video_composer.asyncio.create_subprocess_exec", side_effect=fake_subproc),
    ):
        dur = await composer.render_scene_clip(
            SceneMedia(
                slide_image_path=str(slide),
                audio_path=str(audio),
                duration_seconds=20.0,
                video_clip_path=str(clip),
            ),
            str(out),
        )

    assert dur == pytest.approx(23.36)
    cmd = captured_cmds[0]
    assert _arg_value(cmd, "-t") == "23.36"
    assert f"setpts={1.0 / 1.35:.6f}*PTS" in _arg_value(cmd, "-vf")
    assert "apad,atrim=0:23.36" in _arg_value(cmd, "-af")


@pytest.mark.asyncio
async def test_matching_narration_scales_smoothly(tmp_path: Path) -> None:
    """Khi audio 18.0s và clip 15.0s (raw_speed=0.833), đồng bộ đúng thời lượng audio + 0.4s."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"mp4")
    slide = tmp_path / "slide.png"
    slide.write_bytes(b"png")
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"mp3")
    out = tmp_path / "out.mp4"

    composer = VideoComposer()
    captured_cmds: list[tuple[str, ...]] = []

    async def fake_subproc(*cmd: str, **kwargs):
        captured_cmds.append(cmd)
        proc = MagicMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"", b""))
        return proc

    probe_mock = AsyncMock(side_effect=[15.0, 18.40])
    with (
        patch.object(composer, "_probe_duration", probe_mock),
        patch("worker.services.video_composer.asyncio.create_subprocess_exec", side_effect=fake_subproc),
    ):
        dur = await composer.render_scene_clip(
            SceneMedia(
                slide_image_path=str(slide),
                audio_path=str(audio),
                duration_seconds=18.0,
                video_clip_path=str(clip),
            ),
            str(out),
        )

    assert dur == pytest.approx(18.40)
    cmd = captured_cmds[0]
    assert _arg_value(cmd, "-t") == "18.40"
    assert f"setpts={18.0 / 15.0:.6f}*PTS" in _arg_value(cmd, "-vf")


@pytest.mark.asyncio
async def test_long_narration_freezes_final_frame_with_tpad(tmp_path: Path) -> None:
    """Khi audio 25.0s dài hơn clip 10.0s, giảm tốc tối thiểu 0.75x và tpad đóng băng khung hình cuối."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"mp4")
    slide = tmp_path / "slide.png"
    slide.write_bytes(b"png")
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"mp3")
    out = tmp_path / "out.mp4"

    composer = VideoComposer()
    captured_cmds: list[tuple[str, ...]] = []

    async def fake_subproc(*cmd: str, **kwargs):
        captured_cmds.append(cmd)
        proc = MagicMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"", b""))
        return proc

    probe_mock = AsyncMock(side_effect=[10.0, 25.40])
    with (
        patch.object(composer, "_probe_duration", probe_mock),
        patch("worker.services.video_composer.asyncio.create_subprocess_exec", side_effect=fake_subproc),
    ):
        dur = await composer.render_scene_clip(
            SceneMedia(
                slide_image_path=str(slide),
                audio_path=str(audio),
                duration_seconds=25.0,
                video_clip_path=str(clip),
            ),
            str(out),
        )

    assert dur == pytest.approx(25.40)
    cmd = captured_cmds[0]
    assert _arg_value(cmd, "-t") == "25.40"
    vf = _arg_value(cmd, "-vf")
    assert f"setpts={1.0 / 0.75:.6f}*PTS" in vf
    assert "tpad=stop_mode=clone:stop_duration=" in vf
    assert "apad,atrim=0:25.40" in _arg_value(cmd, "-af")
