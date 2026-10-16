"""
GradeShortAnswersUseCase — chấm câu tự luận ngắn (SHORT_ANSWER) của một lượt nộp.

core-api ghi bài làm với grading_status PENDING cùng đề lúc nộp (grading_input),
giữ cả lượt nộp ở ags_status AWAITING_GRADE và phát ANSWER_GRADING_REQUESTED (xem
core-api quiz/answer-grading.ts). Ở đây mỗi câu được LLM phán từng ý trong rubric
của đề lúc nộp, điểm = số ý đạt / tổng số ý; chấm xong cả lượt thì mở khoá để
AGS đẩy điểm lên LMS.

Không chấm được thì không cho điểm: câu chuyển NEEDS_REVIEW chờ giảng viên, cả
lượt nộp giữ AWAITING_GRADE. Lỗi của nhà cung cấp model không được thành điểm 0.
"""
import time
import uuid
from dataclasses import dataclass

from pydantic import BaseModel, Field

from document_chunk.application.services.structured_output import generate_structured_payload
from document_chunk.domain.exceptions import ProcessingError
from document_chunk.domain.ports.llm_client import LLMUsage
from document_chunk.domain.ports.metadata_store import IMetadataStore, StoredAnswerForGrading
from document_chunk.shared.logger import get_logger
from document_chunk.shared.result import Err, Ok, Result

from worker.llm_role import LlmRole
from worker.prompts import answer_grading as prompts

logger = get_logger(__name__)

# llm_usage_logs.status CHECK chỉ nhận ba giá trị này, phân biệt hoa thường.
_USAGE_OK = "OK"
_USAGE_ERROR = "ERROR"

_PENDING = "PENDING"

# Người học thấy nhận xét này thay cho điểm câu, để không tưởng mình sai.
REVIEW_FEEDBACK = "AI chưa chấm được câu này. Giảng viên sẽ chấm, điểm cập nhật sau."


class _PointVerdict(BaseModel):
    index: int
    met: bool
    comment: str = ""


class _GradePayload(BaseModel):
    points: list[_PointVerdict] = Field(min_length=1)
    feedback: str = ""


@dataclass(frozen=True)
class GradeShortAnswersRequest:
    submission_id: str
    course_id: str
    attempt_ids: list[str]
    essay_attempt_ids: list[str]


@dataclass(frozen=True)
class GradeShortAnswersResponse:
    graded: int
    needs_review: int
    released: int


def score_from_points(rubric_size: int, points: list[_PointVerdict]) -> int:
    """Điểm 0–100. Ý model không phán (thiếu index) tính là chưa đạt."""
    met = {p.index for p in points if p.met and 0 <= p.index < rubric_size}
    return round(100 * len(met) / rubric_size) if rubric_size else 0


def grading_key(answer: StoredAnswerForGrading) -> tuple[str, list[str]] | None:
    """(bài làm mẫu, rubric) của đề lúc nộp; None nếu bài không lưu đề hợp lệ."""
    if answer.question_type != "SHORT_ANSWER" or not answer.question:
        return None
    key = answer.correct_answer if isinstance(answer.correct_answer, dict) else {}
    rubric = [str(point) for point in key.get("rubric") or [] if str(point).strip()]
    if not rubric:
        return None
    return str(key.get("model_answer") or ""), rubric


class GradeShortAnswersUseCase:
    def __init__(self, metadata_store: IMetadataStore, grader: LlmRole) -> None:
        self._metadata_store = metadata_store
        # Chấm là việc của judge: dùng chung model với bước kiểm tra câu hỏi.
        self._grader = grader

    def execute(
        self,
        request: GradeShortAnswersRequest,
        *,
        final_attempt: bool = True,
    ) -> Result[GradeShortAnswersResponse, Exception]:
        """
        ``final_attempt`` False: câu LLM chấm lỗi được để nguyên PENDING và trả
        Err để hàng đợi thử lại; câu đã chấm không bị chấm lại. True: câu lỗi
        chuyển NEEDS_REVIEW, không ghi điểm, và cả lượt nộp không được mở AGS.
        Câu không có đề lúc nộp hợp lệ chuyển NEEDS_REVIEW ngay vì thử lại
        cũng không chấm được.
        """
        trace_id = str(uuid.uuid4())
        answers_result = self._metadata_store.list_answers_for_grading(request.essay_attempt_ids)
        if answers_result.is_err():
            return Err(answers_result.error)

        graded = needs_review = retry = 0
        for answer in answers_result.unwrap():
            if answer.grading_status != _PENDING:
                continue  # đã chấm ở lượt trước, hoặc đang chờ giảng viên
            key = grading_key(answer)
            if key is None:
                logger.warning("grade_short_answers.no_grading_input", attempt_id=answer.attempt_id,
                               question_type=answer.question_type)
                grade = None
            else:
                grade = self._grade(answer, *key, trace_id=trace_id, course_id=request.course_id)
                if grade is None and not final_attempt:
                    retry += 1
                    continue
            if grade is None:
                saved = self._metadata_store.mark_answer_needs_review(
                    attempt_id=answer.attempt_id, feedback=REVIEW_FEEDBACK,
                )
                if saved.is_err():
                    return Err(saved.error)
                needs_review += 1
                continue
            score, feedback = grade
            saved = self._metadata_store.save_answer_grade(
                attempt_id=answer.attempt_id,
                score=score,
                is_correct=score == 100,
                feedback=feedback,
            )
            if saved.is_err():
                return Err(saved.error)
            graded += 1

        if retry:
            return Err(ProcessingError(
                f"{retry} câu tự luận chưa chấm được trong lượt nộp {request.submission_id}"
            ))

        released = self._metadata_store.release_graded_attempts(request.attempt_ids)
        if released.is_err():
            return Err(released.error)
        logger.info(
            "grade_short_answers.done",
            submission_id=request.submission_id,
            graded=graded,
            needs_review=needs_review,
            released=released.unwrap(),
        )
        return Ok(GradeShortAnswersResponse(
            graded=graded, needs_review=needs_review, released=released.unwrap(),
        ))

    def _grade(
        self,
        answer: StoredAnswerForGrading,
        model_answer: str,
        rubric: list[str],
        *,
        trace_id: str,
        course_id: str,
    ) -> tuple[int, str | None] | None:
        """(điểm, nhận xét); None khi model lỗi hoặc trả JSON sai dạng."""
        prompt = prompts.grading(
            question=answer.question or "",
            model_answer=model_answer,
            rubric=rubric,
            answer=str(answer.chosen_answer or ""),
        )
        started = time.monotonic()
        usage = LLMUsage()

        def on_usage(value: LLMUsage) -> None:
            nonlocal usage
            usage = usage + value

        result = generate_structured_payload(
            llm_client=self._grader.client,
            prompt=prompt.user,
            system=prompt.system,
            schema_model=_GradePayload,
            repair_schema_name="AnswerGrade",
            on_usage=on_usage,
        )
        logged = self._metadata_store.record_llm_usage(
            provider=self._grader.provider,
            model=self._grader.client.model_id,
            use_case=prompt.name,
            status=_USAGE_OK if result.is_ok() else _USAGE_ERROR,
            course_id=course_id,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=usage.cost_usd,
            latency_ms=int((time.monotonic() - started) * 1000),
            trace_id=trace_id,
        )
        if logged.is_err():
            logger.warning("grade_short_answers.usage_log_failed", error=str(logged.error))
        if result.is_err():
            logger.warning("grade_short_answers.llm_failed", attempt_id=answer.attempt_id,
                           error=str(result.error))
            return None
        payload = result.unwrap()
        return score_from_points(len(rubric), payload.points), payload.feedback.strip() or None


__all__ = [
    "REVIEW_FEEDBACK",
    "GradeShortAnswersRequest",
    "GradeShortAnswersResponse",
    "GradeShortAnswersUseCase",
    "grading_key",
    "score_from_points",
]
