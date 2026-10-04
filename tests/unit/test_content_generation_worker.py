from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from bullmq.custom_errors import UnrecoverableError

from document_chunk.shared.result import Ok
from worker.runners.content_generation_worker import (
    _card_request,
    _parse_job,
    _process_job,
    _quiz_request,
)
from worker.use_cases.generate_curriculum_quiz import GenerateCurriculumQuizResponse
from worker.use_cases.generate_lesson_cards import (
    DEFAULT_CARD_COUNT,
    GenerateLessonCardsResponse,
)


def _job_data():
    return {
        "event_id": "event-1",
        "event_type": "CONTENT_GENERATION_REQUESTED",
        "payload": {
            "request_id": "request-1",
            "course_id": "course-1",
            "type": "quiz",
            "scope": {
                "target_kind": "lo",
                "target_code": "L.O.3.1",
                "count": 3,
                "bloom": "analyze",
            },
        },
    }


def test_parse_job_and_scope():
    job = _parse_job(_job_data())
    request = _quiz_request(job)

    assert job.request_id == "request-1"
    assert request.target_kind == "lo"
    assert request.target_code == "L.O.3.1"
    assert request.count == 3
    assert request.bloom_level == "analyze"


@pytest.mark.asyncio
async def test_process_job_updates_request_lifecycle():
    container = MagicMock()
    container.metadata_store.update_content_generation_request.return_value = Ok(None)
    container.generate_curriculum_quiz_use_case.execute.return_value = Ok(
        GenerateCurriculumQuizResponse(
            course_id="course-1",
            lo_id="lo-1",
            quiz_count=3,
            question_ids=["q1", "q2", "q3"],
        )
    )
    bull_job = SimpleNamespace(data=_job_data())

    result = await _process_job(container, bull_job, None)

    assert result["quiz_count"] == 3
    lifecycle = container.metadata_store.update_content_generation_request.call_args_list
    assert lifecycle[0].kwargs["status"] == "RUNNING"
    assert lifecycle[-1].kwargs["status"] == "SUCCEEDED"
    assert lifecycle[-1].kwargs["generated_count"] == 3


def _card_job_data(**scope):
    data = _job_data()
    data["payload"]["type"] = "card"
    data["payload"]["scope"] = {"target_kind": "lo", "target_code": "lo-uuid", **scope}
    return data


def test_card_request_reads_target_and_defaults_count():
    request = _card_request(_parse_job(_card_job_data(source_document_ids=["doc-1"])))

    assert request.target_kind == "lo"
    assert request.target_code == "lo-uuid"
    assert request.count == DEFAULT_CARD_COUNT
    assert request.source_document_ids == ["doc-1"]


def test_card_request_rejects_count_out_of_range():
    with pytest.raises(ValueError, match="card count"):
        _card_request(_parse_job(_card_job_data(count=50)))


@pytest.mark.asyncio
async def test_card_job_runs_card_use_case_and_records_count():
    container = MagicMock()
    container.metadata_store.update_content_generation_request.return_value = Ok(None)
    container.generate_lesson_cards_use_case.execute.return_value = Ok(
        GenerateLessonCardsResponse(
            course_id="course-1",
            lo_id="lo-1",
            card_count=2,
            card_ids=["c1", "c2"],
        )
    )

    result = await _process_job(container, SimpleNamespace(data=_card_job_data(count=2)), None)

    assert result["card_ids"] == ["c1", "c2"]
    container.generate_curriculum_quiz_use_case.execute.assert_not_called()
    lifecycle = container.metadata_store.update_content_generation_request.call_args_list
    assert lifecycle[-1].kwargs["status"] == "SUCCEEDED"
    assert lifecycle[-1].kwargs["generated_count"] == 2


@pytest.mark.asyncio
async def test_unsupported_content_type_marks_request_failed():
    container = MagicMock()
    container.metadata_store.update_content_generation_request.return_value = Ok(None)
    data = _job_data()
    data["payload"]["type"] = "summary"

    with pytest.raises(UnrecoverableError):
        await _process_job(container, SimpleNamespace(data=data), None)

    update = container.metadata_store.update_content_generation_request
    assert update.call_args.kwargs["request_id"] == "request-1"
    assert update.call_args.kwargs["status"] == "FAILED"


@pytest.mark.asyncio
async def test_unparseable_job_still_marks_request_failed():
    """
    Job hỏng mà không đánh dấu FAILED thì dòng nằm mãi ở QUEUED, và FE chặn
    mọi yêu cầu sinh quiz sau đó của khoá học bằng 409.
    """
    container = MagicMock()
    container.metadata_store.update_content_generation_request.return_value = Ok(None)
    data = _job_data()
    data["payload"]["scope"] = "không-phải-object"

    with pytest.raises(UnrecoverableError):
        await _process_job(container, SimpleNamespace(data=data), None)

    update = container.metadata_store.update_content_generation_request
    assert update.call_args.kwargs["request_id"] == "request-1"
    assert update.call_args.kwargs["status"] == "FAILED"
    assert "scope" in update.call_args.kwargs["last_error"]


@pytest.mark.asyncio
async def test_request_id_recovered_from_aggregate_id_when_payload_is_broken():
    """Outbox đặt aggregate_id = request_id, nên payload hỏng hẳn vẫn cứu được."""
    container = MagicMock()
    container.metadata_store.update_content_generation_request.return_value = Ok(None)
    data = {
        "event_id": "event-1",
        "event_type": "CONTENT_GENERATION_REQUESTED",
        "aggregate_id": "request-42",
        "payload": None,
    }

    with pytest.raises(UnrecoverableError):
        await _process_job(container, SimpleNamespace(data=data), None)

    update = container.metadata_store.update_content_generation_request
    assert update.call_args.kwargs["request_id"] == "request-42"
    assert update.call_args.kwargs["status"] == "FAILED"


@pytest.mark.asyncio
async def test_job_without_any_request_id_does_not_crash():
    container = MagicMock()

    with pytest.raises(UnrecoverableError):
        await _process_job(container, SimpleNamespace(data={"payload": None}), None)

    container.metadata_store.update_content_generation_request.assert_not_called()


@pytest.mark.asyncio
async def test_failing_to_mark_failed_does_not_mask_the_original_error():
    container = MagicMock()
    container.metadata_store.update_content_generation_request.side_effect = RuntimeError("db down")
    data = _job_data()
    data["payload"]["scope"] = 123

    with pytest.raises(UnrecoverableError):
        await _process_job(container, SimpleNamespace(data=data), None)
