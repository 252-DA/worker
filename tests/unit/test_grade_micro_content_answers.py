from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from bullmq.custom_errors import UnrecoverableError

from document_chunk.domain.exceptions import ProcessingError
from document_chunk.shared.result import Err, Ok
from tests.support import RecordingResultPublisher
from tests.unit.grading_contract import FakeBullJob, assert_matches_contract, new_id
from worker.runners.answer_grading_worker import parse_micro_content_grading_job, process_grading_job
from worker.use_cases.grade_micro_content_answers import (
    GradeMicroContentAnswersRequest,
    GradeMicroContentAnswersUseCase,
)
from worker.use_cases.shared.llm_role import LlmRole
from worker.use_cases.shared.rubric_grading import (
    LEGACY_COMMAND,
    MODEL_ERROR,
    REVIEW_FEEDBACK,
    AnswerToGrade,
    GradePayload,
    PointVerdict,
)

EVENT, ATTEMPT, COURSE, MICRO, ANSWER_ID = new_id(), new_id(), new_id(), new_id(), new_id()
GRADING_INPUT = {
    "type": "SHORT_ANSWER",
    "question": "Giải thích khái niệm pure function?",
    "correct_answer": {
        "model_answer": "Hàm thuần khiết luôn trả cùng output cho cùng input và không có side effects.",
        "rubric": [
            "Cùng input trả cùng output",
            "Không có side effects / tác dụng phụ",
        ],
    },
}
CHOSEN = "Hàm không làm thay đổi biến bên ngoài và cùng tham số trả cùng giá trị."


def _request() -> GradeMicroContentAnswersRequest:
    return GradeMicroContentAnswersRequest(
        event_id=EVENT,
        attempt_id=ATTEMPT,
        course_id=COURSE,
        micro_content_id=MICRO,
        answers=(AnswerToGrade(ANSWER_ID, GRADING_INPUT, CHOSEN),),
    )


def _job_data(**payload):
    body = {
        "attempt_id": ATTEMPT, "course_id": COURSE, "micro_content_id": MICRO,
        "answer_ids": [ANSWER_ID],
        "answers": [{"answer_id": ANSWER_ID, "grading_input": GRADING_INPUT, "chosen_answer": CHOSEN}],
        **payload,
    }
    return {"event_id": EVENT, "event_type": "MICRO_CONTENT_GRADING_REQUESTED", "aggregate_id": ATTEMPT,
            "payload": body}


def _store():
    # Chỉ được ghi sổ chi phí; mọi hàm đọc/ghi micro_content_answers đã bỏ.
    store = MagicMock(spec=["record_llm_usage"])
    store.record_llm_usage.return_value = Ok(None)
    return store


def test_grades_micro_content_answer_successfully():
    grader = MagicMock()
    grader.client.model_id = "test-model"
    use_case = GradeMicroContentAnswersUseCase(metadata_store=_store(), grader=grader)
    payload = GradePayload(
        points=[
            PointVerdict(index=0, met=True, comment="Ý 1 đúng"),
            PointVerdict(index=1, met=True, comment="Ý 2 đúng"),
        ],
        feedback="Bài làm chính xác.",
    )
    assert len(payload.points) == 2

    use_case._grade = MagicMock(return_value=(100, "Bài làm chính xác."))

    outcome = use_case.execute(_request(), final_attempt=True)

    assert outcome.graded == 1
    assert outcome.needs_review == 0
    [grade] = outcome.grades
    assert grade.to_payload("answer_id") == {
        "answer_id": ANSWER_ID,
        "status": "GRADED",
        "score": 100,
        "is_correct": True,
        "feedback": "Bài làm chính xác.",
        "grader_model": "test-model",
    }


def test_marks_needs_review_on_final_attempt_failure():
    use_case = GradeMicroContentAnswersUseCase(metadata_store=_store(), grader=MagicMock())
    use_case._grade = MagicMock(return_value=None)  # Model error

    outcome = use_case.execute(_request(), final_attempt=True)

    assert outcome.needs_review == 1
    [grade] = outcome.grades
    assert (grade.answer_id, grade.status, grade.reason) == (ANSWER_ID, "NEEDS_REVIEW", MODEL_ERROR)
    assert grade.feedback == REVIEW_FEEDBACK
    # Lỗi model không thành điểm 0.
    assert grade.score is None and "score" not in grade.to_payload("answer_id")


def test_model_error_before_last_attempt_is_retried():
    use_case = GradeMicroContentAnswersUseCase(metadata_store=_store(), grader=MagicMock())
    use_case._grade = MagicMock(return_value=None)

    outcome = use_case.execute(_request(), final_attempt=False)

    assert outcome.retry == 1 and outcome.grades == ()


def test_real_grading_path_uses_the_quiz_prompt_and_logs_usage(mock_llm_client):
    """Đường chấm thật (không mock ``_grade``): cùng prompt và sổ chi phí với tự luận quiz."""
    store = _store()
    mock_llm_client.generate.return_value = Ok('{"points":[{"index":0,"met":true}],"feedback":"Thiếu ý 2."}')
    use_case = GradeMicroContentAnswersUseCase(store, LlmRole(client=mock_llm_client, provider="test"))

    [grade] = use_case.execute(_request()).grades

    assert (grade.status, grade.score, grade.feedback) == ("GRADED", 50, "Thiếu ý 2.")
    assert f"<<<LEARNER_ANSWER\n{CHOSEN}\nLEARNER_ANSWER>>>" in mock_llm_client.generate.call_args.args[0]
    assert store.record_llm_usage.call_args.kwargs["use_case"] == "answer_grading"
    assert store.record_llm_usage.call_args.kwargs["course_id"] == COURSE


def test_wrong_point_index_is_a_model_error_not_a_score(mock_llm_client):
    """Cùng bộ chấm rubric với quiz: index đếm từ 1 không được thành điểm lệch."""
    store = _store()
    mock_llm_client.generate.return_value = Ok(
        '{"points":[{"index":1,"met":true},{"index":2,"met":true}],"feedback":"x"}'
    )
    use_case = GradeMicroContentAnswersUseCase(store, LlmRole(client=mock_llm_client, provider="test"))

    retried = use_case.execute(_request(), final_attempt=False)
    [grade] = use_case.execute(_request(), final_attempt=True).grades

    assert retried.retry == 1 and retried.grades == ()
    assert (grade.status, grade.reason, grade.score) == ("NEEDS_REVIEW", MODEL_ERROR, None)


class TestJob:
    def _container(self, mock_llm_client, publisher=None):
        return SimpleNamespace(
            grade_micro_content_answers_use_case=GradeMicroContentAnswersUseCase(
                _store(), LlmRole(client=mock_llm_client, provider="test"),
            ),
            result_publisher=publisher or RecordingResultPublisher(),
        )

    async def test_publishes_micro_content_result_without_a_total(self, mock_llm_client):
        mock_llm_client.generate.return_value = Ok(
            '{"points":[{"index":0,"met":true},{"index":1,"met":true}],"feedback":"Đủ ý."}'
        )
        container = self._container(mock_llm_client)

        result = await process_grading_job(container, FakeBullJob(_job_data()), "token")

        [envelope] = container.result_publisher.envelopes
        assert envelope["kind"] == "MICRO_CONTENT_GRADING_RESULT"
        assert envelope["result_id"] == f"MICRO_CONTENT_GRADING_RESULT.{EVENT}"
        assert envelope["source_event_id"] == EVENT
        assert envelope["payload"] == {
            "attempt_id": ATTEMPT, "course_id": COURSE, "micro_content_id": MICRO,
            "answers": [{
                "answer_id": ANSWER_ID, "status": "GRADED", "score": 100, "is_correct": True,
                "feedback": "Đủ ý.", "grader_model": "gemini-test",
            }],
        }
        assert_matches_contract(envelope)
        assert result == {"attempt_id": ATTEMPT, "graded": 1, "needs_review": 0,
                          "result_id": f"MICRO_CONTENT_GRADING_RESULT.{EVENT}"}

    async def test_model_error_retries_then_publishes_needs_review(self, mock_llm_client):
        mock_llm_client.generate.return_value = Err(RuntimeError("quota"))
        container = self._container(mock_llm_client)

        with pytest.raises(ProcessingError):
            await process_grading_job(container, FakeBullJob(_job_data()), "token")
        assert container.result_publisher.envelopes == []

        await process_grading_job(container, FakeBullJob(_job_data(), attempts_made=4), "token")
        [envelope] = container.result_publisher.envelopes
        assert envelope["payload"]["answers"][0]["status"] == "NEEDS_REVIEW"
        assert_matches_contract(envelope)

    async def test_publish_failure_fails_the_job(self, mock_llm_client):
        mock_llm_client.generate.return_value = Ok('{"points":[{"index":0,"met":true}],"feedback":""}')
        container = self._container(mock_llm_client, RecordingResultPublisher(fail_with=ConnectionError("down")))

        with pytest.raises(ConnectionError):
            await process_grading_job(container, FakeBullJob(_job_data()), "token")

    async def test_legacy_command_publishes_needs_review(self, mock_llm_client):
        data = _job_data()
        del data["payload"]["answers"]
        container = self._container(mock_llm_client)

        await process_grading_job(container, FakeBullJob(data), "token")

        [envelope] = container.result_publisher.envelopes
        assert [(a["answer_id"], a["reason"]) for a in envelope["payload"]["answers"]] == [(ANSWER_ID, LEGACY_COMMAND)]
        assert_matches_contract(envelope)
        mock_llm_client.generate.assert_not_called()

    async def test_malformed_command_is_unrecoverable(self, mock_llm_client):
        container = self._container(mock_llm_client)
        with pytest.raises(UnrecoverableError):
            await process_grading_job(container, FakeBullJob(_job_data(micro_content_id="")), "token")


def test_parse_reads_the_self_contained_command():
    request = parse_micro_content_grading_job(_job_data())
    assert request.answers == (AnswerToGrade(ANSWER_ID, GRADING_INPUT, CHOSEN),)
    assert request.event_id == EVENT
