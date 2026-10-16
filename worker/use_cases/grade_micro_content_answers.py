import time
import uuid
from dataclasses import dataclass

from document_chunk.application.services.structured_output import generate_structured_payload
from document_chunk.domain.exceptions import ProcessingError
from document_chunk.domain.ports.llm_client import LLMUsage
from document_chunk.domain.ports.metadata_store import StoredAnswerForGrading
from document_chunk.shared.logger import get_logger
from document_chunk.shared.result import Err, Ok, Result

from worker.llm_role import LlmRole
from worker.prompts import answer_grading as prompts
from worker.services.micro_content_grading_store import MicroContentGradingStore
from worker.use_cases.grade_short_answers import (
    REVIEW_FEEDBACK,
    _GradePayload,
    grading_key,
    score_from_points,
)

logger = get_logger(__name__)

_USAGE_OK = "OK"
_USAGE_ERROR = "ERROR"
_PENDING = "PENDING"


@dataclass(frozen=True)
class GradeMicroContentAnswersRequest:
    attempt_id: str
    course_id: str
    micro_content_id: str
    answer_ids: list[str]


@dataclass(frozen=True)
class GradeMicroContentAnswersResponse:
    graded: int
    needs_review: int


class GradeMicroContentAnswersUseCase:
    def __init__(
        self,
        store: MicroContentGradingStore,
        grader: LlmRole,
    ) -> None:
        self._store = store
        self._grader = grader

    def execute(
        self,
        request: GradeMicroContentAnswersRequest,
        *,
        final_attempt: bool = True,
    ) -> Result[GradeMicroContentAnswersResponse, Exception]:
        trace_id = str(uuid.uuid4())
        answers_result = self._store.list_answers_for_grading(request.answer_ids)
        if answers_result.is_err():
            return Err(answers_result.error)

        graded = needs_review = retry = 0
        for answer in answers_result.unwrap():
            if answer.grading_status != _PENDING:
                continue

            key = grading_key(answer)
            if key is None:
                logger.warning(
                    "grade_micro_content_answers.no_grading_input",
                    answer_id=answer.attempt_id,
                    question_type=answer.question_type,
                )
                grade = None
            else:
                grade = self._grade(answer, *key, trace_id=trace_id, course_id=request.course_id)
                if grade is None and not final_attempt:
                    retry += 1
                    continue

            if grade is None:
                saved = self._store.mark_answer_needs_review(
                    answer_id=answer.attempt_id,
                    feedback=REVIEW_FEEDBACK,
                )
                if saved.is_err():
                    return Err(saved.error)
                needs_review += 1
                continue

            score, feedback = grade
            saved = self._store.save_answer_grade(
                answer_id=answer.attempt_id,
                score=score,
                is_correct=score == 100,
                feedback=feedback,
            )
            if saved.is_err():
                return Err(saved.error)
            graded += 1

        if retry:
            return Err(
                ProcessingError(
                    f"{retry} câu tự luận micro-content chưa chấm được trong attempt {request.attempt_id}"
                )
            )

        # Tính lại điểm tổng và cập nhật trạng thái của attempt
        finalized = self._store.finalize_attempt(request.attempt_id)
        if finalized.is_err():
            return Err(finalized.error)

        logger.info(
            "grade_micro_content_answers.done",
            attempt_id=request.attempt_id,
            graded=graded,
            needs_review=needs_review,
        )
        return Ok(GradeMicroContentAnswersResponse(graded=graded, needs_review=needs_review))

    def _grade(
        self,
        answer: StoredAnswerForGrading,
        model_answer: str,
        rubric: list[str],
        *,
        trace_id: str,
        course_id: str,
    ) -> tuple[int, str] | None:
        user_prompt = prompts.grading_prompt(
            question=answer.question,
            rubric=rubric,
            model_answer=model_answer,
            student_answer=str(answer.chosen_answer or ""),
        )
        t0 = time.perf_counter()
        call = self._grader.call(course_id=course_id)
        result = generate_structured_payload(
            client=call.client,
            schema=_GradePayload,
            system_prompt=prompts.system_prompt(),
            user_prompt=user_prompt,
            model=call.profile.model,
            temperature=0.0,
            seed=42,
            max_tokens=call.profile.max_output_tokens or 1500,
        )
        duration_ms = int((time.perf_counter() - t0) * 1000)

        if result.is_err():
            logger.warning(
                "grade_micro_content_answers.model_failed",
                answer_id=answer.attempt_id,
                error=str(result.error),
            )
            return None

        payload: _GradePayload = result.unwrap()
        score = score_from_points(len(rubric), payload.points)
        feedback = payload.feedback.strip()
        return score, feedback
