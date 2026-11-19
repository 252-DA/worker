"""
GradeShortAnswersUseCase — chấm câu tự luận ngắn (SHORT_ANSWER) của một lượt nộp quiz.

core-api ghi bài làm với grading_status PENDING, giữ cả lượt nộp ở ags_status
AWAITING_GRADE và phát ANSWER_GRADING_REQUESTED kèm đề lúc nộp (grading_input)
và bài làm của từng câu (xem core-api src/modules/question/answer-grading.ts). Ở đây mỗi câu
được LLM phán từng ý trong rubric của đề lúc nộp, điểm = số ý đạt / tổng số ý.

Use case chỉ tính, không đọc hay ghi bảng bài làm: runner gửi kết quả
QUIZ_GRADING_RESULT về core-api qua hàng đợi worker_results, core-api ghi điểm,
tính điểm lượt nộp và mở khoá AGS. Chỉ còn ghi llm_usage_logs (sổ chi phí của AI).

Không chấm được thì không cho điểm: câu thành NEEDS_REVIEW chờ giảng viên. Lỗi của
nhà cung cấp model không được thành điểm 0.

Phần chấm theo rubric (``RubricAnswerGrader``) dùng chung với câu tự luận của
Micro Content (grade_micro_content_answers.py).
"""
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from document_chunk.application.services.structured_output import generate_structured_payload
from document_chunk.domain.ports.llm_client import LLMUsage
from document_chunk.domain.ports.metadata_store import IMetadataStore
from document_chunk.shared.logger import get_logger

from worker.llm_role import LlmRole
from worker.prompts import answer_grading as prompts
from worker.use_cases.grading_verdicts import MalformedGrade, check_verdict_indices

logger = get_logger(__name__)

# llm_usage_logs.status CHECK chỉ nhận ba giá trị này, phân biệt hoa thường.
_USAGE_OK = "OK"
_USAGE_ERROR = "ERROR"

GRADED = "GRADED"
NEEDS_REVIEW = "NEEDS_REVIEW"

# Lý do NEEDS_REVIEW, khớp enum `reason` trong quiz-grading-result.json.
MODEL_ERROR = "MODEL_ERROR"  # model lỗi tới lượt thử cuối
NO_GRADING_INPUT = "NO_GRADING_INPUT"  # đề lúc nộp thiếu hoặc không có rubric: thử lại cũng vô ích
LEGACY_COMMAND = "LEGACY_COMMAND"  # lệnh dạng cũ không kèm dữ liệu chấm
REASONS = (MODEL_ERROR, NO_GRADING_INPUT, LEGACY_COMMAND)

# Người học thấy nhận xét này thay cho điểm câu, để không tưởng mình sai.
REVIEW_FEEDBACK = "AI chưa chấm được câu này. Giảng viên sẽ chấm, điểm cập nhật sau."

# Khớp maxLength trong schema kết quả: core-api từ chối (không retry) chuỗi dài hơn.
MAX_FEEDBACK_CHARS = 4000
MAX_MODEL_CHARS = 200


class _PointVerdict(BaseModel):
    index: int
    met: bool
    comment: str = ""


class _GradePayload(BaseModel):
    points: list[_PointVerdict] = Field(min_length=1)
    feedback: str = ""


@dataclass(frozen=True)
class AnswerToGrade:
    """Một câu cần chấm, lấy nguyên từ lệnh (không đọc DB)."""

    answer_id: str  # quiz: quiz_attempts.attempt_id; micro content: micro_content_answers.answer_id
    grading_input: Any  # {type, question, correct_answer} lúc nộp
    chosen_answer: Any


@dataclass(frozen=True)
class AnswerGrade:
    """Kết quả chấm một câu, đúng dạng một phần tử ``answers`` của kết quả gửi core-api."""

    answer_id: str
    status: str
    score: int | None = None
    is_correct: bool | None = None
    feedback: str | None = None
    grader_model: str | None = None
    reason: str | None = None

    @classmethod
    def graded(cls, answer_id: str, score: int, feedback: str | None, grader_model: str | None):
        return cls(
            answer_id=answer_id, status=GRADED, score=score, is_correct=score == 100,
            feedback=_clip(feedback, MAX_FEEDBACK_CHARS), grader_model=_clip(grader_model, MAX_MODEL_CHARS),
        )

    @classmethod
    def needs_review(cls, answer_id: str, reason: str, grader_model: str | None = None):
        return cls(
            answer_id=answer_id, status=NEEDS_REVIEW, feedback=REVIEW_FEEDBACK,
            grader_model=_clip(grader_model, MAX_MODEL_CHARS), reason=reason,
        )

    def to_payload(self, id_key: str) -> dict[str, Any]:
        """Phần tử ``answers``; NEEDS_REVIEW không có score/is_correct (không phải điểm 0)."""
        if self.status == GRADED:
            return {
                id_key: self.answer_id, "status": GRADED, "score": self.score,
                "is_correct": self.is_correct, "feedback": self.feedback, "grader_model": self.grader_model,
            }
        return {
            id_key: self.answer_id, "status": NEEDS_REVIEW, "reason": self.reason,
            "feedback": self.feedback, "grader_model": self.grader_model,
        }

    def to_progress(self) -> dict[str, Any]:
        return self.to_payload("answer_id")

    @classmethod
    def from_progress(cls, raw: Any) -> "AnswerGrade | None":
        """Đọc lại từ dữ liệu job (lượt thử trước); hỏng thì None để chấm lại câu đó."""
        if not isinstance(raw, Mapping) or not isinstance(raw.get("answer_id"), str):
            return None
        feedback, model = raw.get("feedback"), raw.get("grader_model")
        if not (feedback is None or isinstance(feedback, str)) or not (model is None or isinstance(model, str)):
            return None
        score = raw.get("score")
        if raw.get("status") == GRADED and isinstance(score, int) and not isinstance(score, bool) \
                and 0 <= score <= 100:
            return cls.graded(raw["answer_id"], score, feedback, model)
        if raw.get("status") == NEEDS_REVIEW and raw.get("reason") in REASONS:
            return cls(answer_id=raw["answer_id"], status=NEEDS_REVIEW, feedback=feedback,
                       grader_model=model, reason=raw["reason"])
        return None


@dataclass(frozen=True)
class GradingOutcome:
    """
    ``grades``: mọi câu đã có kết quả, kể cả câu chấm ở lượt thử trước.
    ``retry`` > 0: còn câu model lỗi khi chưa tới lượt thử cuối — chưa được gửi
    kết quả, job phải fail để BullMQ thử lại.
    """

    grades: tuple[AnswerGrade, ...]
    retry: int = 0

    @property
    def graded(self) -> int:
        return sum(1 for g in self.grades if g.status == GRADED)

    @property
    def needs_review(self) -> int:
        return sum(1 for g in self.grades if g.status == NEEDS_REVIEW)


def score_from_points(rubric_size: int, points: list[_PointVerdict]) -> int:
    """
    Điểm 0–100 = số ý đạt / tổng số ý. Ý model không phán (thiếu index) tính là
    chưa đạt. Index ngoài khoảng (đếm từ 1: ý cuối thành index n) hoặc trùng là
    lỗi định dạng: ném ``MalformedGrade`` chứ không bỏ qua, vì bỏ qua một ý "đạt"
    làm điểm thấp hơn thực tế mà không ai biết.
    """
    check_verdict_indices(
        (p.index for p in points), rubric_size, require_all=False, what="ý"
    )
    met = {p.index for p in points if p.met}
    return round(100 * len(met) / rubric_size) if rubric_size else 0


def grading_key(grading_input: Any) -> tuple[str, str, list[str]] | None:
    """(đề, bài làm mẫu, rubric) của đề lúc nộp; None nếu bài không lưu đề hợp lệ."""
    if not isinstance(grading_input, Mapping):
        return None
    question = grading_input.get("question")
    if grading_input.get("type") != "SHORT_ANSWER" or not isinstance(question, str) or not question:
        return None
    key = grading_input.get("correct_answer")
    key = key if isinstance(key, Mapping) else {}
    rubric = [str(point) for point in key.get("rubric") or [] if str(point).strip()]
    if not rubric:
        return None
    return question, str(key.get("model_answer") or ""), rubric


def _clip(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    return value[:limit]


class RubricAnswerGrader:
    """Chấm từng câu tự luận theo rubric của đề lúc nộp; dùng chung cho quiz và Micro Content."""

    def __init__(self, metadata_store: IMetadataStore, grader: LlmRole) -> None:
        # Chỉ dùng để ghi llm_usage_logs; không đọc/ghi bảng bài làm.
        self._metadata_store = metadata_store
        # Chấm là việc của judge: dùng chung model với bước kiểm tra câu hỏi.
        self._grader = grader

    def _grade_all(
        self,
        answers: Sequence[AnswerToGrade],
        *,
        legacy_ids: Sequence[str],
        course_id: str,
        final_attempt: bool,
        prior: Mapping[str, AnswerGrade] | None,
        ref: str,
    ) -> GradingOutcome:
        """
        ``final_attempt`` False: câu model chấm lỗi được để lại (``retry``) cho
        lượt sau; câu đã chấm ở lượt trước (``prior``) không bị chấm lại. True:
        câu lỗi thành NEEDS_REVIEW, không có điểm. Câu không có đề lúc nộp hợp lệ
        thành NEEDS_REVIEW ngay vì thử lại cũng không chấm được.
        """
        prior = prior or {}
        trace_id = str(uuid.uuid4())
        model = self._model_name()
        grades: list[AnswerGrade] = []
        retry = 0
        for answer in answers:
            done = prior.get(answer.answer_id)
            if done is not None:
                grades.append(done)
                continue
            key = grading_key(answer.grading_input)
            if key is None:
                logger.warning("grade_answers.no_grading_input", ref=ref, answer_id=answer.answer_id)
                grades.append(AnswerGrade.needs_review(answer.answer_id, NO_GRADING_INPUT))
                continue
            question, model_answer, rubric = key
            result = self._grade(answer, question, model_answer, rubric, trace_id=trace_id, course_id=course_id)
            if result is None:
                if not final_attempt:
                    retry += 1
                    continue
                grades.append(AnswerGrade.needs_review(answer.answer_id, MODEL_ERROR, model))
                continue
            score, feedback = result
            grades.append(AnswerGrade.graded(answer.answer_id, score, feedback, model))
        for answer_id in legacy_ids:
            grades.append(AnswerGrade.needs_review(answer_id, LEGACY_COMMAND))
        return GradingOutcome(grades=tuple(grades), retry=retry)

    def _model_name(self) -> str | None:
        model = getattr(self._grader.client, "model_id", None)
        return model if isinstance(model, str) and model else None

    def _grade(
        self,
        answer: AnswerToGrade,
        question: str,
        model_answer: str,
        rubric: list[str],
        *,
        trace_id: str,
        course_id: str,
    ) -> tuple[int, str | None] | None:
        """(điểm, nhận xét); None khi model lỗi hoặc trả JSON sai dạng."""
        prompt = prompts.grading(
            question=question,
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
            logger.warning("grade_answers.usage_log_failed", error=str(logged.error))
        if result.is_err():
            logger.warning("grade_answers.llm_failed", answer_id=answer.answer_id, error=str(result.error))
            return None
        payload = result.unwrap()
        try:
            return score_from_points(len(rubric), payload.points), payload.feedback.strip() or None
        except MalformedGrade as exc:
            # Như model lỗi: thử lại, lượt cuối giao giảng viên — không thành điểm.
            logger.warning("grade_answers.malformed_grade", answer_id=answer.answer_id, error=str(exc))
            return None


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


def quiz_result_payload(request: GradeShortAnswersRequest, grades: Sequence[AnswerGrade]) -> dict[str, Any]:
    """Payload QUIZ_GRADING_RESULT (contracts/integration-events/v1/quiz-grading-result.json)."""
    return {
        "submission_id": request.submission_id,
        "course_id": request.course_id,
        "answers": [grade.to_payload("attempt_id") for grade in grades],
    }


__all__ = [
    "GRADED",
    "LEGACY_COMMAND",
    "MODEL_ERROR",
    "NEEDS_REVIEW",
    "NO_GRADING_INPUT",
    "REVIEW_FEEDBACK",
    "AnswerGrade",
    "AnswerToGrade",
    "GradeShortAnswersRequest",
    "GradeShortAnswersUseCase",
    "GradingOutcome",
    "RubricAnswerGrader",
    "grading_key",
    "quiz_result_payload",
    "score_from_points",
]
