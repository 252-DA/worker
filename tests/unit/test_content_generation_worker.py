from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from bullmq.custom_errors import UnrecoverableError

from document_chunk.shared.result import Ok
from worker.runners.content_generation_worker import (
    _parse_job,
    _process_job,
    _quiz_request,
)
from worker.use_cases.generate_curriculum_quiz import GenerateCurriculumQuizResponse


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


@pytest.mark.asyncio
async def test_unsupported_content_type_marks_request_failed():
    container = MagicMock()
    container.metadata_store.update_content_generation_request.return_value = Ok(None)
    data = _job_data()
    data["payload"]["type"] = "card"

    with pytest.raises(UnrecoverableError):
        await _process_job(container, SimpleNamespace(data=data), None)

    update = container.metadata_store.update_content_generation_request
    assert update.call_args.kwargs["request_id"] == "request-1"
    assert update.call_args.kwargs["status"] == "FAILED"
