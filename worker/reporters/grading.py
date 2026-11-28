"""
Kết quả chấm bài → core-api:

- QUIZ_GRADING_RESULT: tự luận ngắn của một lượt nộp quiz
  (contracts/integration-events/v1/quiz-grading-result.json)
- MICRO_CONTENT_GRADING_RESULT: câu tự luận của Micro Content
  (micro-content-grading-result.json)
- ESSAY_GRADING_RESULT: bài luận nộp file (essay-grading-result.json). Payload do
  use case dựng (``grade_essay_submission``) vì nó cũng là bản giữ tiến độ trong
  dữ liệu job; reporter chỉ gửi.

``result_id`` = ``<KIND>.<event_id của lệnh>``: job gốc retry sau khi đã gửi sinh
lại đúng id và BullMQ bỏ bản trùng; mỗi lần giảng viên cho chấm lại là một sự
kiện mới nên có id mới.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from worker.adapters.result_publisher import (
    ESSAY_GRADING_RESULT,
    MICRO_CONTENT_GRADING_RESULT,
    QUIZ_GRADING_RESULT,
    make_result_id,
)

if TYPE_CHECKING:
    from worker.use_cases.grade_micro_content_answers import GradeMicroContentAnswersRequest
    from worker.use_cases.grade_short_answers import GradeShortAnswersRequest
    from worker.use_cases.shared.rubric_grading import AnswerGrade


class GradingReporter:
    def __init__(self, publisher: Any) -> None:
        self._publisher = publisher

    async def quiz(self, request: GradeShortAnswersRequest, grades: Sequence[AnswerGrade]) -> str:
        return await self._publish(QUIZ_GRADING_RESULT, quiz_result_payload(request, grades), request.event_id)

    async def micro_content(
        self, request: GradeMicroContentAnswersRequest, grades: Sequence[AnswerGrade]
    ) -> str:
        return await self._publish(
            MICRO_CONTENT_GRADING_RESULT, micro_content_result_payload(request, grades), request.event_id
        )

    async def essay(self, payload: dict[str, Any], source_event_id: str) -> str:
        return await self._publish(ESSAY_GRADING_RESULT, payload, source_event_id)

    async def _publish(self, kind: str, payload: dict[str, Any], source_event_id: str) -> str:
        """Chỉ trả về khi Redis đã nhận; lỗi nổi lên cho runner giữ tiến độ rồi để job thử lại."""
        return await self._publisher.publish(
            kind,
            payload,
            result_id=make_result_id(kind, source_event_id),
            source_event_id=source_event_id,
        )


def quiz_result_payload(request: GradeShortAnswersRequest, grades: Sequence[AnswerGrade]) -> dict[str, Any]:
    """Payload QUIZ_GRADING_RESULT."""
    return {
        "submission_id": request.submission_id,
        "course_id": request.course_id,
        "answers": [grade.to_payload("attempt_id") for grade in grades],
    }


def micro_content_result_payload(
    request: GradeMicroContentAnswersRequest, grades: Sequence[AnswerGrade]
) -> dict[str, Any]:
    """Payload MICRO_CONTENT_GRADING_RESULT."""
    return {
        "attempt_id": request.attempt_id,
        "course_id": request.course_id,
        "micro_content_id": request.micro_content_id,
        "answers": [grade.to_payload("answer_id") for grade in grades],
    }


__all__ = ["GradingReporter", "micro_content_result_payload", "quiz_result_payload"]
