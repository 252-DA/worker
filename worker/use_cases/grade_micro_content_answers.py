"""
GradeMicroContentAnswersUseCase — chấm câu tự luận của bài củng cố Micro Content.

core-api phát MICRO_CONTENT_GRADING_REQUESTED kèm đề lúc nộp và bài làm của từng
câu (micro-content-attempt.service.ts). Chấm đúng như tự luận ngắn của quiz
(cùng prompt, cùng model judge, cùng sổ chi phí); runner gửi
MICRO_CONTENT_GRADING_RESULT, core-api ghi điểm câu và tính lại điểm lượt nộp
(submissionScore — không còn round() của Python làm tròn về số chẵn).
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
class GradeMicroContentAnswersRequest:
    event_id: str
    attempt_id: str
    course_id: str
    micro_content_id: str
    answers: tuple[AnswerToGrade, ...]
    legacy_answer_ids: tuple[str, ...] = ()

    @property
    def answer_ids(self) -> set[str]:
        return {a.answer_id for a in self.answers}


class GradeMicroContentAnswersUseCase(RubricAnswerGrader):
    def execute(
        self,
        request: GradeMicroContentAnswersRequest,
        *,
        final_attempt: bool = True,
        prior: Mapping[str, AnswerGrade] | None = None,
    ) -> GradingOutcome:
        outcome = self._grade_all(
            request.answers,
            legacy_ids=request.legacy_answer_ids,
            course_id=request.course_id,
            final_attempt=final_attempt,
            prior=prior,
            ref=request.attempt_id,
        )
        logger.info(
            "grade_micro_content_answers.done",
            attempt_id=request.attempt_id,
            graded=outcome.graded,
            needs_review=outcome.needs_review,
            retry=outcome.retry,
        )
        return outcome


__all__ = [
    "GradeMicroContentAnswersRequest",
    "GradeMicroContentAnswersUseCase",
]
