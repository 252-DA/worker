from unittest.mock import MagicMock

from document_chunk.domain.ports.metadata_store import StoredAnswerForGrading
from document_chunk.shared.result import Err, Ok

from worker.use_cases.grade_micro_content_answers import (
    GradeMicroContentAnswersRequest,
    GradeMicroContentAnswersUseCase,
    REVIEW_FEEDBACK,
    _GradePayload,
    _PointVerdict,
)


def _answer(status="PENDING") -> StoredAnswerForGrading:
    return StoredAnswerForGrading(
        attempt_id="answer-123",
        quiz_id="question-1",
        course_id="course-1",
        question_type="SHORT_ANSWER",
        question="Giải thích khái niệm pure function?",
        correct_answer={
            "model_answer": "Hàm thuần khiết luôn trả cùng output cho cùng input và không có side effects.",
            "rubric": [
                "Cùng input trả cùng output",
                "Không có side effects / tác dụng phụ",
            ],
        },
        chosen_answer="Hàm không làm thay đổi biến bên ngoài và cùng tham số trả cùng giá trị.",
        grading_status=status,
    )


def test_grades_micro_content_answer_successfully():
    store = MagicMock()
    store.list_answers_for_grading.return_value = Ok([_answer()])
    store.save_answer_grade.return_value = Ok(True)
    store.finalize_attempt.return_value = Ok(True)

    grader = MagicMock()
    grader.call.return_value.profile.model = "test-model"
    grader.call.return_value.profile.max_output_tokens = 1000

    use_case = GradeMicroContentAnswersUseCase(store=store, grader=grader)
    payload = _GradePayload(
        points=[
            _PointVerdict(index=0, met=True, comment="Ý 1 đúng"),
            _PointVerdict(index=1, met=True, comment="Ý 2 đúng"),
        ],
        feedback="Bài làm chính xác.",
    )

    use_case._grade = MagicMock(return_value=(100, "Bài làm chính xác."))

    req = GradeMicroContentAnswersRequest(
        attempt_id="attempt-1",
        course_id="course-1",
        micro_content_id="micro-1",
        answer_ids=["answer-123"],
    )

    res = use_case.execute(req, final_attempt=True)
    assert res.is_ok()
    assert res.unwrap().graded == 1
    assert res.unwrap().needs_review == 0
    store.save_answer_grade.assert_called_once_with(
        answer_id="answer-123",
        score=100,
        is_correct=True,
        feedback="Bài làm chính xác.",
    )
    store.finalize_attempt.assert_called_once_with("attempt-1")


def test_marks_needs_review_on_final_attempt_failure():
    store = MagicMock()
    store.list_answers_for_grading.return_value = Ok([_answer()])
    store.mark_answer_needs_review.return_value = Ok(True)
    store.finalize_attempt.return_value = Ok(True)

    grader = MagicMock()
    use_case = GradeMicroContentAnswersUseCase(store=store, grader=grader)
    use_case._grade = MagicMock(return_value=None)  # Model error

    req = GradeMicroContentAnswersRequest(
        attempt_id="attempt-1",
        course_id="course-1",
        micro_content_id="micro-1",
        answer_ids=["answer-123"],
    )

    res = use_case.execute(req, final_attempt=True)
    assert res.is_ok()
    assert res.unwrap().needs_review == 1
    store.mark_answer_needs_review.assert_called_once_with(
        answer_id="answer-123",
        feedback=REVIEW_FEEDBACK,
    )
    store.finalize_attempt.assert_called_once_with("attempt-1")

