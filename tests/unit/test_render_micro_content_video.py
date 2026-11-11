import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from bullmq.custom_errors import UnrecoverableError

from document_chunk.shared.result import Err, Ok
from tests.support import RecordingResultPublisher
from worker.runners.micro_content_video_worker import parse_render_job, process_video_job
from worker.use_cases.render_micro_content_video import (
    InvalidRenderRequest,
    RenderMicroContentVideoRequest,
    RenderMicroContentVideoUseCase,
)

MC = "22222222-2222-4222-8222-222222222222"
COURSE = "33333333-3333-4333-8333-333333333333"
ASSET = "55555555-5555-4555-8555-555555555555"
RENDER = "66666666-6666-4666-8666-666666666666"
EVENT = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaac1"
KEY = f"micro-content/{COURSE}/{MC}/ai-{RENDER}.mp4"

TWO_SCENES = {
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


def render_job(**payload_overrides):
    payload = {
        "micro_content_id": MC,
        "course_id": COURSE,
        "asset_id": ASSET,
        "render_id": RENDER,
        "script": {"scenes": [{"title": "Intro"}]},
    }
    payload.update(payload_overrides)
    return {
        "event_id": EVENT,
        "event_type": "MICRO_CONTENT_RENDER_REQUESTED",
        "aggregate_type": "micro_content",
        "aggregate_id": MC,
        "payload": {k: v for k, v in payload.items() if v is not None},
    }


def request(script=TWO_SCENES):
    return RenderMicroContentVideoRequest(
        micro_content_id=MC,
        course_id=COURSE,
        asset_id=ASSET,
        render_id=RENDER,
        script=script,
        source_event_id=EVENT,
    )


def make_use_case(publisher=None, tts_side_effect=(3.5, 4.2)):
    file_storage = MagicMock()
    file_storage.upload.return_value = Ok(KEY)
    tts_client = MagicMock()
    tts_client.synthesize = AsyncMock(side_effect=tts_side_effect)
    slide_gen = MagicMock()
    slide_gen.generate_slide.return_value = "slide.png"

    async def fake_compose(scenes, work_dir, output_mp4_path):
        # Use case đọc kích thước file đầu ra sau khi ghép.
        Path(output_mp4_path).write_bytes(b"mp4")
        return 7.7

    video_composer = MagicMock()
    video_composer.compose_video = AsyncMock(side_effect=fake_compose)
    publisher = publisher if publisher is not None else RecordingResultPublisher()
    use_case = RenderMicroContentVideoUseCase(
        result_publisher=publisher,
        file_storage=file_storage,
        tts_client=tts_client,
        slide_generator=slide_gen,
        video_composer=video_composer,
    )
    return SimpleNamespace(
        use_case=use_case,
        publisher=publisher,
        file_storage=file_storage,
        tts_client=tts_client,
        slide_gen=slide_gen,
    )


def bull_job(data, attempts_made=0, attempts=5):
    return SimpleNamespace(data=data, opts={"attempts": attempts}, attemptsMade=attempts_made)


# --- parse_render_job --------------------------------------------------------


def test_parse_render_job_success():
    req = parse_render_job(render_job())
    assert req.micro_content_id == MC
    assert req.course_id == COURSE
    assert req.asset_id == ASSET
    assert req.render_id == RENDER
    assert req.source_event_id == EVENT
    assert req.script["scenes"][0]["title"] == "Intro"


def test_parse_render_job_missing_payload():
    with pytest.raises(ValueError, match="outbox payload"):
        parse_render_job({})


@pytest.mark.parametrize(
    "override",
    [
        {"render_id": None},  # lệnh phát trước khi có ai_render_id
        {"asset_id": None},
        {"course_id": None},
        {"render_id": "render-1"},
        {"asset_id": "55555555555545558555555555555555"},  # không có gạch nối
    ],
)
def test_parse_render_job_rejects_commands_without_valid_ids(override):
    with pytest.raises(ValueError, match="UUID"):
        parse_render_job(render_job(**override))


# --- use case ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_render_micro_content_video_success():
    t = make_use_case()

    res = await t.use_case.execute(request())

    assert res.is_ok()
    data = res.unwrap()
    assert data.video_object_key == KEY
    assert data.duration_seconds == 7.7
    assert data.size_bytes == 3
    assert t.tts_client.synthesize.await_count == 2
    assert t.slide_gen.generate_slide.call_count == 2
    # Mỗi lần render một object riêng, không ghi đè file của lần khác.
    assert t.file_storage.upload.call_args.args[1] == KEY
    # Kết quả đi về core-api đúng một lần, thay cho ghi micro_contents/video_assets.
    assert len(t.publisher.envelopes) == 1
    envelope = t.publisher.envelopes[0]
    assert envelope["kind"] == "MICRO_CONTENT_VIDEO_RESULT"
    assert envelope["result_id"] == f"MICRO_CONTENT_VIDEO_RESULT.{RENDER}.ready"
    assert envelope["source_event_id"] == EVENT
    assert envelope["payload"] == {
        "micro_content_id": MC,
        "asset_id": ASSET,
        "render_id": RENDER,
        "status": "READY",
        "object_key": KEY,
        "duration_ms": 7700,
        "size_bytes": 3,
        "mime_type": "video/mp4",
    }


@pytest.mark.asyncio
async def test_render_micro_content_video_handles_error():
    t = make_use_case(tts_side_effect=RuntimeError("TTS failure"))
    req = request({"scenes": [{"title": "Cảnh 1"}]})

    res = await t.use_case.execute(req)
    assert res.is_err()
    assert "TTS failure" in str(res.error)
    # Chưa báo FAILED: job gốc còn lượt thì BullMQ thử lại, lần sau có thể xong.
    assert t.publisher.envelopes == []

    # Khi người gọi quyết định dừng, lỗi đến tay giảng viên kèm thông báo.
    await t.use_case.report_failure(req, res.error)
    [envelope] = t.publisher.envelopes
    assert envelope["result_id"] == f"MICRO_CONTENT_VIDEO_RESULT.{RENDER}.failed"
    assert envelope["payload"] == {
        "micro_content_id": MC,
        "asset_id": ASSET,
        "render_id": RENDER,
        "status": "FAILED",
        "error": "TTS failure",
    }


@pytest.mark.asyncio
async def test_report_failure_truncates_and_never_sends_an_empty_error():
    t = make_use_case()
    await t.use_case.report_failure(request(), "x" * 5000)
    await t.use_case.report_failure(request(), TimeoutError())
    assert len(t.publisher.envelopes[0]["payload"]["error"]) == 1000
    assert t.publisher.envelopes[1]["payload"]["error"] == "TimeoutError"


@pytest.mark.parametrize("script", [None, {}, {"scenes": []}, {"scenes": "a"}, {"scenes": ["a"]}])
@pytest.mark.asyncio
async def test_script_without_scenes_is_an_invalid_request(script):
    t = make_use_case()

    res = await t.use_case.execute(request(script))

    assert isinstance(res.error, InvalidRenderRequest)
    assert t.tts_client.synthesize.await_count == 0
    assert t.publisher.envelopes == []


@pytest.mark.asyncio
async def test_upload_error_is_not_reported_ready():
    t = make_use_case()
    t.file_storage.upload.return_value = Err(OSError("bucket down"))

    res = await t.use_case.execute(request())

    assert "bucket down" in str(res.error)
    assert t.publisher.envelopes == []


@pytest.mark.asyncio
async def test_publish_error_after_upload_fails_the_render():
    t = make_use_case(publisher=RecordingResultPublisher(fail_with=ConnectionError("redis down")))

    res = await t.use_case.execute(request())

    assert isinstance(res.error, ConnectionError)


def test_video_worker_no_longer_has_a_database_store():
    # Ranh giới dữ liệu: worker video không ghi (cũng không đọc) bảng của core-api.
    assert importlib.util.find_spec("worker.services.micro_content_video_store") is None
    assert "store" not in inspect.signature(RenderMicroContentVideoUseCase).parameters


# --- runner ------------------------------------------------------------------


def container_for(t):
    return SimpleNamespace(render_micro_content_video_use_case=t.use_case)


@pytest.mark.asyncio
async def test_runner_success_publishes_ready_and_completes():
    t = make_use_case()

    out = await process_video_job(container_for(t), bull_job(render_job(script=TWO_SCENES)), "tok")

    assert out["render_id"] == RENDER
    assert out["video_object_key"] == KEY
    assert [e["payload"]["status"] for e in t.publisher.envelopes] == ["READY"]


@pytest.mark.asyncio
async def test_runner_transient_error_with_attempts_left_retries_silently():
    t = make_use_case(tts_side_effect=RuntimeError("TTS failure"))

    with pytest.raises(RuntimeError, match="TTS failure"):
        await process_video_job(container_for(t), bull_job(render_job(), attempts_made=0), "tok")

    assert t.publisher.envelopes == []


@pytest.mark.asyncio
async def test_runner_final_attempt_reports_failed_then_fails_job():
    t = make_use_case(tts_side_effect=RuntimeError("TTS failure"))

    with pytest.raises(RuntimeError, match="TTS failure"):
        await process_video_job(container_for(t), bull_job(render_job(), attempts_made=4), "tok")

    [envelope] = t.publisher.envelopes
    assert envelope["payload"]["status"] == "FAILED"
    assert envelope["payload"]["error"] == "TTS failure"


@pytest.mark.asyncio
async def test_runner_invalid_script_reports_failed_without_retry():
    t = make_use_case()

    with pytest.raises(UnrecoverableError):
        await process_video_job(container_for(t), bull_job(render_job(script={"scenes": []})), "tok")

    assert [e["payload"]["status"] for e in t.publisher.envelopes] == ["FAILED"]


@pytest.mark.asyncio
async def test_runner_failed_publish_error_fails_job_for_retry():
    t = make_use_case(publisher=RecordingResultPublisher(fail_with=ConnectionError("redis down")))

    # Kịch bản hỏng nhưng FAILED chưa gửi được: lỗi thường để BullMQ thử lại,
    # không phải UnrecoverableError (kết quả sẽ mất).
    with pytest.raises(ConnectionError):
        await process_video_job(container_for(t), bull_job(render_job(script=None)), "tok")


@pytest.mark.asyncio
async def test_runner_rejects_command_without_render_id_and_sends_nothing():
    t = make_use_case()

    with pytest.raises(UnrecoverableError, match="render_id"):
        await process_video_job(container_for(t), bull_job(render_job(render_id=None)), "tok")

    assert t.publisher.envelopes == []


@pytest.mark.asyncio
async def test_runner_acknowledges_script_requested_without_publishing():
    t = make_use_case()
    data = {
        "event_id": EVENT,
        "event_type": "MICRO_CONTENT_SCRIPT_REQUESTED",
        "aggregate_type": "micro_content",
        "aggregate_id": MC,
        "payload": {"micro_content_id": MC, "course_id": COURSE, "prompt": "x", "options": None},
    }

    assert await process_video_job(container_for(t), bull_job(data), "tok") == {"status": "ignored"}
    assert t.publisher.envelopes == []


# --- hợp đồng ----------------------------------------------------------------

CONTRACTS = Path(__file__).resolve().parents[3] / "contracts"


@pytest.mark.asyncio
async def test_published_results_and_parsed_command_match_contract_schemas():
    if not CONTRACTS.exists():
        pytest.skip("contracts/ không có trong image này")
    import json

    import jsonschema

    schemas_dir = CONTRACTS / "integration-events" / "v1"
    store = {}
    for path in schemas_dir.glob("*.json"):
        schema = json.loads(path.read_text(encoding="utf-8"))
        store[schema["$id"]] = schema

    def validator(name):
        schema = store[f"https://da-platform.dev/contracts/integration-events/v1/{name}"]
        resolver = jsonschema.RefResolver.from_schema(schema, store=store)
        return jsonschema.Draft7Validator(schema, resolver=resolver)

    t = make_use_case()
    await t.use_case.execute(request())
    await t.use_case.report_failure(request(), "TTS failure")
    result_validator = validator("micro-content-video-result.json")
    for envelope in t.publisher.envelopes:
        assert list(result_validator.iter_errors(envelope)) == []

    # Lệnh mẫu trong fixture đúng là thứ parse_render_job đọc được.
    fixture = json.loads(
        (CONTRACTS / "fixtures" / "v1" / "valid" / "micro-content-render-requested.json").read_text(
            encoding="utf-8"
        )
    )
    assert list(validator("micro-content-render-requested.json").iter_errors(fixture)) == []
    req = parse_render_job(fixture)
    assert req.render_id == fixture["payload"]["render_id"]
    assert req.source_event_id == fixture["event_id"]
