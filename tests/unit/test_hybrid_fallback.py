import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

worker_dir = Path("/app/worker")
packages_ai_dir = Path("/app/packages-ai/src")
ai_sdk_dir = Path("/app/ai-sdk/src")
for p in (worker_dir, packages_ai_dir, ai_sdk_dir):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from document_chunk.shared.result import Ok
from worker.use_cases.render_micro_content_video import (
    RenderMicroContentVideoRequest,
    RenderMicroContentVideoUseCase,
)


class DummyPublisher:
    def __init__(self):
        self.envelopes = []

    async def publish(self, kind, payload, *, result_id, source_event_id=None):
        self.envelopes.append({"kind": kind, "payload": payload, "result_id": result_id})
        return "msg-123"


async def test_fallback_when_manim_clip_missing():
    print("[TEST 1] Kiểm tra cơ chế Fallback khi clip Manim không tồn tại...")

    file_storage = MagicMock()
    file_storage.upload.return_value = Ok("test-video.mp4")
    file_storage.download.return_value = Ok(None)

    tts_client = MagicMock()
    tts_client.synthesize = AsyncMock(return_value=4.0)

    slide_gen = MagicMock()
    slide_gen.generate_slide.return_value = "slide.png"

    async def fake_compose(scenes, work_dir, output_mp4_path):
        Path(output_mp4_path).write_bytes(b"mp4-data")
        return 8.0

    video_composer = MagicMock()
    video_composer.compose_video = AsyncMock(side_effect=fake_compose)
    video_composer.compose_video_with_timeline = AsyncMock(
        return_value=MagicMock(duration_seconds=8.0, scene_durations_seconds=[4.0, 4.0])
    )

    publisher = DummyPublisher()

    use_case = RenderMicroContentVideoUseCase(
        result_publisher=publisher,
        file_storage=file_storage,
        tts_client=tts_client,
        slide_generator=slide_gen,
        video_composer=video_composer,
    )

    # Kịch bản 2 cảnh: Cảnh 1 Slide tĩnh, Cảnh 2 MANIM_CUSTOM nhưng clip không tồn tại
    script = {
        "scenes": [
            {
                "title": "Cảnh 1: Khái niệm",
                "narration": "Nguyên lý chèn trực tiếp.",
                "visual_mode": "SLIDE_STATIC"
            },
            {
                "title": "Cảnh 2: Hoạt họa mảng",
                "narration": "Mô phỏng chèn phần tử.",
                "visual_mode": "MANIM_CUSTOM",
                "manim_clip_key": "non_existent_key_12345.mp4"
            }
        ]
    }

    req = RenderMicroContentVideoRequest(
        micro_content_id="22222222-2222-4222-8222-222222222222",
        course_id="33333333-3333-4333-8333-333333333333",
        asset_id="55555555-5555-4555-8555-555555555555",
        render_id="66666666-6666-4666-8666-666666666666",
        script=script,
    )

    res = await use_case.execute(req)
    assert res.is_ok(), f"Render thất bại: {res.error}"
    data = res.unwrap()

    print(f"  + Render hoàn tất: {data.video_object_key}")
    print(f"  + fallback_used:   {data.fallback_used}")
    print(f"  + fallback_reason: {data.fallback_reason}")
    print(f"  + fallbacks list:  {data.fallbacks}")

    assert data.fallback_used is True, "Phải ghi nhận fallback_used = True"
    assert len(data.fallbacks) == 1, "Phải có đúng 1 cảnh fallback"
    assert data.fallbacks[0]["scene_index"] == 1
    assert data.fallbacks[0]["fallback_mode"] == "SLIDE_STATIC"
    print("  -> PASSED: Cơ chế Fallback hoạt động hoàn hảo, hệ thống an toàn 100%!")


if __name__ == "__main__":
    asyncio.run(test_fallback_when_manim_clip_missing())
