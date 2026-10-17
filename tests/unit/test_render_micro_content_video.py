import pytest
from unittest.mock import AsyncMock, MagicMock

from document_chunk.shared.result import Err, Ok
from worker.runners.micro_content_video_worker import parse_render_job
from worker.services.micro_content_video_store import MicroContentRecord
from worker.services.video_composer import SceneMedia
from worker.use_cases.render_micro_content_video import (
    RenderMicroContentVideoRequest,
    RenderMicroContentVideoUseCase,
)


def test_parse_render_job_success():
    data = {
        "event_type": "MICRO_CONTENT_RENDER_REQUESTED",
        "aggregate_id": "mc-123",
        "payload": {
            "micro_content_id": "mc-123",
            "course_id": "c-456",
            "script": {"scenes": [{"title": "Intro"}]},
        },
    }
    req = parse_render_job(data)
    assert req.micro_content_id == "mc-123"
    assert req.course_id == "c-456"
    assert req.script["scenes"][0]["title"] == "Intro"


def test_parse_render_job_missing_payload():
    with pytest.raises(ValueError, match="outbox payload"):
        parse_render_job({})


@pytest.mark.asyncio
async def test_render_micro_content_video_success():
    store = MagicMock()
    file_storage = MagicMock()
    tts_client = MagicMock()
    slide_gen = MagicMock()
    video_composer = MagicMock()

    script = {
        "scenes": [
            {
                "title": "Cảnh 1: Khái niệm",
                "bullets": ["Ý 1", "Ý 2"],
                "narration": "Đây là khái niệm.",
            },
            {
                "title": "Cảnh 2: Áp dụng",
                "bullets": ["Ví dụ thực tế"],
                "narration": "Đây là ứng dụng.",
            },
        ]
    }
    record = MicroContentRecord(
        micro_content_id="mc-123",
        course_id="c-456",
        title="Tiêu đề",
        ai_script=script,
        video_status="PROCESSING",
    )
    store.get_micro_content.return_value = Ok(record)
    store.update_video_ready.return_value = Ok(True)

    tts_client.synthesize = AsyncMock(side_effect=[3.5, 4.2])
    slide_gen.generate_slide.return_value = "slide.png"
    video_composer.compose_video = AsyncMock(return_value=7.7)
    file_storage.upload.return_value = Ok("micro-content/c-456/mc-123/video.mp4")

    use_case = RenderMicroContentVideoUseCase(
        store=store,
        file_storage=file_storage,
        tts_client=tts_client,
        slide_generator=slide_gen,
        video_composer=video_composer,
    )

    req = RenderMicroContentVideoRequest(
        micro_content_id="mc-123",
        course_id="c-456",
    )

    res = await use_case.execute(req)
    assert res.is_ok
    data = res.unwrap()
    assert data.video_object_key == "micro-content/c-456/mc-123/video.mp4"
    assert data.duration_seconds == 7.7
    assert tts_client.synthesize.await_count == 2
    assert slide_gen.generate_slide.call_count == 2
    store.update_video_ready.assert_called_once()


@pytest.mark.asyncio
async def test_render_micro_content_video_handles_error():
    store = MagicMock()
    file_storage = MagicMock()
    tts_client = MagicMock()
    slide_gen = MagicMock()
    video_composer = MagicMock()

    record = MicroContentRecord(
        micro_content_id="mc-123",
        course_id="c-456",
        title="Tiêu đề",
        ai_script={"scenes": [{"title": "Cảnh 1"}]},
        video_status="PROCESSING",
    )
    store.get_micro_content.return_value = Ok(record)
    tts_client.synthesize = AsyncMock(side_effect=RuntimeError("TTS failure"))

    use_case = RenderMicroContentVideoUseCase(
        store=store,
        file_storage=file_storage,
        tts_client=tts_client,
        slide_generator=slide_gen,
        video_composer=video_composer,
    )

    res = await use_case.execute(RenderMicroContentVideoRequest(micro_content_id="mc-123"))
    assert res.is_err
    assert "TTS failure" in str(res.unwrap_err())
    store.update_video_failed.assert_called_once_with("mc-123", "TTS failure")
