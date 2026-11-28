"""
GradeShortAnswersUseCase — chấm câu tự luận ngắn (SHORT_ANSWER) của một lượt nộp quiz.

core-api ghi bài làm với grading_status PENDING, giữ cả lượt nộp ở ags_status
AWAITING_GRADE và phát ANSWER_GRADING_REQUESTED kèm đề lúc nộp (grading_input)
và bài làm của từng câu (xem core-api src/modules/question/answer-grading.ts). Ở đây mỗi câu
được LLM phán từng ý trong rubric của đề lúc nộp, điểm = số ý đạt / tổng số ý.

Use case chỉ tính, không đọc hay ghi bảng bài làm: runner gửi kết quả
QUIZ_GRADING_RESULT (reporters/grading.py) về core-api qua hàng đợi worker_results, core-api ghi điểm,
tính điểm lượt nộp và mở khoá AGS. Chỉ còn ghi llm_usage_logs (sổ chi phí của AI).

Phần chấm theo rubric nằm ở ``shared/rubric_grading.py``, dùng chung với câu tự
luận của Micro Content.
"""
from collections.abc import Mapping
from dataclasses import dataclass

from document_chunk.shared.logger import get_logger

from worker.use_cases.shared.rubric_grading import (
    AnswerGrade,
    AnswerToGrade,
    GradingOutcome,
    RubricAnswerGrader,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class GradeShortAnswersRequest:
    event_id: str  # event_id outbox của lệnh: nguồn của result_id
    submission_id: str
    course_id: str
    answers: tuple[AnswerToGrade, ...]
    # Lệnh dạng cũ (không có `answers`): chỉ còn id câu, trả NEEDS_REVIEW.
    legacy_attempt_ids: tuple[str, ...] = ()

    @property
    def answer_ids(self) -> set[str]:
        return {a.answer_id for a in self.answers}


class GradeShortAnswersUseCase(RubricAnswerGrader):
    def execute(
        self,
        request: GradeShortAnswersRequest,
        *,
        final_attempt: bool = True,
        prior: Mapping[str, AnswerGrade] | None = None,
    ) -> GradingOutcome:
        outcome = self._grade_all(
            request.answers,
            legacy_ids=request.legacy_attempt_ids,
            course_id=request.course_id,
            final_attempt=final_attempt,
            prior=prior,
            ref=request.submission_id,
        )
        logger.info(
            "grade_short_answers.done",
            submission_id=request.submission_id,
            graded=outcome.graded,
            needs_review=outcome.needs_review,
            retry=outcome.retry,
        )
        return outcome


__all__ = [
    "GradeShortAnswersRequest",
    "GradeShortAnswersUseCase",
]
