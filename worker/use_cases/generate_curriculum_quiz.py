"""
GenerateCurriculumQuizUseCase — generate quiz items aligned to a specific LO/chapter/assessment.

Sibling to RunEnrichmentUseCase (course-scoped, not document-scoped).
Used by the queued content-generation worker and the legacy synchronous adapter.
"""
import re
import time
import uuid
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, Field, field_validator, model_validator

from document_chunk.application.dto.generation_dto import (
    GenerateCurriculumQuizRequest,
    GenerateCurriculumQuizResponse,
)
from document_chunk.application.services.answer_layout import (
    balanced_positions,
    shuffle_choices,
    shuffle_order,
)
from document_chunk.application.services.structured_output import generate_structured_payload
from document_chunk.domain.bloom import bloom_to_level, bloom_to_name
from document_chunk.domain.exceptions import ProcessingError
from document_chunk.domain.ports.llm_client import ILLMClient, LLMUsage
from document_chunk.domain.ports.metadata_store import IMetadataStore, StoredQuizItem
from document_chunk.shared.logger import get_logger
from document_chunk.shared.result import Err, Ok, Result

from worker.llm_role import LlmRole
from worker.prompts import curriculum_quiz as prompts
from worker.use_cases.lo_context import LoContextLoader, QuizContextClient
from worker.use_cases.verification import verify_and_refine

logger = get_logger(__name__)

# Namespace cố định để quiz_id tất định theo (LO, nội dung câu hỏi): worker retry
# sau khi đã lưu sẽ upsert đè lên chính dòng cũ thay vì sinh thêm một bộ trùng.
_QUIZ_ID_NAMESPACE = uuid.UUID("6f9b5f1e-3a4d-5c7b-8e2f-1d0a9c8b7e64")

# llm_usage_logs.status CHECK chỉ nhận ba giá trị này, phân biệt hoa thường.
_USAGE_OK = "OK"
_USAGE_ERROR = "ERROR"

class _UsageAccumulator:
    """Cộng dồn token của mọi lần gọi trong một bước (kể cả lần sửa JSON)."""

    def __init__(self) -> None:
        self.total = LLMUsage()

    def __call__(self, usage: LLMUsage) -> None:
        self.total = self.total + usage


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

# Phương án cố định của câu Đúng/Sai, khớp TRUE_FALSE_OPTIONS ở core-api.
_TRUE_FALSE_OPTIONS = ("Đúng", "Sai")
_BLANK = re.compile(r"_{3,}")


def _texts(values: list[str], *, label: str) -> list[str]:
    normalized = [v.strip() for v in values if v and v.strip()]
    if len(normalized) != len(values):
        raise ValueError(f"{label} must not contain empty items")
    if len({v.casefold() for v in normalized}) != len(normalized):
        raise ValueError(f"{label} must be distinct")
    return normalized


class _QuestionBase(BaseModel):
    question: str
    explanation: str
    difficulty: str = "medium"
    lo_alignment_rationale: str = ""
    source_chunk_ids: list[str] = Field(default_factory=list)

    @field_validator("question", "explanation")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        v = value.strip()
        if not v:
            raise ValueError("value must not be empty")
        return v

    @field_validator("difficulty")
    @classmethod
    def _normalize_difficulty(cls, value: str) -> str:
        v = value.strip().lower()
        if v not in {"easy", "medium", "hard"}:
            return "medium"
        return v


class _CurriculumQuestion(_QuestionBase):
    """MCQ_SINGLE: 4 phương án, một đáp án đúng."""

    choices: list[str] = Field(min_length=4, max_length=4)
    correct_index: int = Field(ge=0, le=3)

    @field_validator("choices")
    @classmethod
    def _normalize_choices(cls, choices: list[str]) -> list[str]:
        normalized = [c.strip() for c in choices if c and c.strip()]
        if len(normalized) != 4:
            raise ValueError("choices must contain exactly 4 non-empty items")
        return normalized


class _MultiAnswerQuestion(_QuestionBase):
    """MCQ_MULTI: 4–6 phương án, chọn tất cả đáp án đúng."""

    choices: list[str] = Field(min_length=4, max_length=6)
    correct_indices: list[int] = Field(min_length=1)

    @field_validator("choices")
    @classmethod
    def _normalize_choices(cls, choices: list[str]) -> list[str]:
        return _texts(choices, label="choices")

    @model_validator(mode="after")
    def _check_indices(self) -> "_MultiAnswerQuestion":
        indices = set(self.correct_indices)
        if len(indices) != len(self.correct_indices):
            raise ValueError("correct_indices must be distinct")
        if not all(0 <= i < len(self.choices) for i in indices):
            raise ValueError("correct_indices must point at choices")
        if len(indices) == len(self.choices):
            raise ValueError("at least one choice must be wrong")
        return self


class _TrueFalseQuestion(_QuestionBase):
    """TRUE_FALSE: ``question`` là một mệnh đề."""

    is_true: bool


class _FillBlankQuestion(_QuestionBase):
    """FILL_BLANK: một chỗ trống ``____``, các cách viết đáp án chấp nhận."""

    accepted_answers: list[str] = Field(min_length=1, max_length=10)

    @field_validator("question")
    @classmethod
    def _one_blank(cls, value: str) -> str:
        if len(_BLANK.findall(value)) != 1:
            raise ValueError("question must contain exactly one blank written as ____")
        return value

    @field_validator("accepted_answers")
    @classmethod
    def _normalize_answers(cls, answers: list[str]) -> list[str]:
        return _texts(answers, label="accepted_answers")


class _ShortAnswerQuestion(_QuestionBase):
    """SHORT_ANSWER: bài mẫu và các ý chấm; worker chấm bài làm bằng LLM."""

    model_answer: str
    rubric: list[str] = Field(min_length=1, max_length=6)

    @field_validator("model_answer")
    @classmethod
    def _model_answer(cls, value: str) -> str:
        v = value.strip()
        if not v:
            raise ValueError("model_answer must not be empty")
        return v

    @field_validator("rubric")
    @classmethod
    def _normalize_rubric(cls, rubric: list[str]) -> list[str]:
        return _texts(rubric, label="rubric")


_QuestionT = TypeVar("_QuestionT", bound=_QuestionBase)


class _CurriculumQuizPayload(BaseModel, Generic[_QuestionT]):
    questions: list[_QuestionT] = Field(min_length=1)


# Kiểu câu → (schema một câu, tên schema cho bước sửa JSON).
_QUESTION_MODELS: dict[str, tuple[type[_QuestionBase], str]] = {
    "MCQ_SINGLE": (_CurriculumQuestion, "CurriculumQuestion"),
    "MCQ_MULTI": (_MultiAnswerQuestion, "MultiAnswerQuestion"),
    "TRUE_FALSE": (_TrueFalseQuestion, "TrueFalseQuestion"),
    "FILL_BLANK": (_FillBlankQuestion, "FillBlankQuestion"),
    "SHORT_ANSWER": (_ShortAnswerQuestion, "ShortAnswerQuestion"),
}


class _QuestionVerdict(BaseModel):
    question_index: int
    verdict: Literal["supported", "contradicted", "not_enough_evidence"]
    reason: str = ""


class _QuizVerificationPayload(BaseModel):
    verdicts: list[_QuestionVerdict] = Field(min_length=1)


def _answer_layout(
    q: _QuestionBase,
    question_type: str,
    *,
    seed: str,
    target_index: int,
) -> tuple[list[str], int, object]:
    """
    (phương án lưu, correct_index, correct_answer) theo dạng của từng kiểu câu.

    ``correct_answer`` None với MCQ_SINGLE: repository lấy ``choices[correct_index]``.
    """
    if isinstance(q, _CurriculumQuestion):
        # G3: xáo phương án, vị trí đáp án đúng phân bố đều trong mẻ.
        choices, correct_index = shuffle_choices(
            list(q.choices), q.correct_index, seed=seed, target_index=target_index
        )
        return choices, correct_index, None
    if isinstance(q, _MultiAnswerQuestion):
        correct = {q.choices[i] for i in q.correct_indices}
        choices = shuffle_order(list(q.choices), seed=seed)
        return choices, 0, [c for c in choices if c in correct]
    if isinstance(q, _TrueFalseQuestion):
        answer = _TRUE_FALSE_OPTIONS[0 if q.is_true else 1]
        return list(_TRUE_FALSE_OPTIONS), _TRUE_FALSE_OPTIONS.index(answer), answer
    if isinstance(q, _FillBlankQuestion):
        return [], 0, list(q.accepted_answers)
    if isinstance(q, _ShortAnswerQuestion):
        return [], 0, {"model_answer": q.model_answer, "rubric": list(q.rubric)}
    raise ProcessingError(f"Unsupported question type: {question_type}")


# ---------------------------------------------------------------------------
# Use case
# ---------------------------------------------------------------------------

class GenerateCurriculumQuizUseCase:
    def __init__(
        self,
        metadata_store: IMetadataStore,
        llm_client: ILLMClient,
        max_context_chars: int = 6000,
        context_client: QuizContextClient | None = None,
        fallback_to_local_context: bool = True,
        llm_provider: str = "unknown",
        llm_profile: str | None = None,
        verifier: LlmRole | None = None,
        max_verification_retries: int = 2,
    ) -> None:
        self._metadata_store = metadata_store
        self._context = LoContextLoader(
            metadata_store=metadata_store,
            context_client=context_client,
            fallback_to_local_context=fallback_to_local_context,
            max_context_chars=max_context_chars,
            log_prefix="generate_curriculum_quiz",
        )
        self._max_verification_retries = max_verification_retries
        # Viết câu và sinh lại câu bị loại đều là việc của writer; chấm là việc
        # của judge. Không truyền judge thì cả hai là một model, như trước đây.
        self._writer = LlmRole(client=llm_client, provider=llm_provider, profile=llm_profile)
        self._verifier = verifier or self._writer

    def execute(
        self, request: GenerateCurriculumQuizRequest
    ) -> Result[GenerateCurriculumQuizResponse, Exception]:
        trace_id = str(uuid.uuid4())
        question_type = request.question_type
        if question_type not in _QUESTION_MODELS:
            return Err(ProcessingError(f"Unsupported question type: {question_type}"))
        question_model, schema_name = _QUESTION_MODELS[question_type]

        # 1. Resolve LO ids from target
        lo_ids = self._resolve_lo_ids(request)
        if lo_ids.is_err():
            return Err(lo_ids.error)

        target_lo_ids = lo_ids.unwrap()
        if not target_lo_ids:
            return Err(ProcessingError(
                f"No LOs found for {request.target_kind}={request.target_code} "
                f"in course {request.course_id}"
            ))

        # For MVP: generate for first LO (or the specific one if lo_code given)
        primary_lo_id = target_lo_ids[0]

        # 2. Fetch curriculum context for the LO
        curriculum_result = self._metadata_store.get_curriculum(request.course_id)
        if curriculum_result.is_err():
            return Err(curriculum_result.error)

        curriculum_data = curriculum_result.unwrap()
        if curriculum_data is None:
            return Err(ProcessingError(f"No curriculum for course {request.course_id}"))

        stored_course, _, stored_los, _ = curriculum_data
        lo_obj = next((lo for lo in stored_los if lo.lo_id == primary_lo_id), None)
        if lo_obj is None:
            return Err(ProcessingError(f"LO {primary_lo_id} not found in curriculum"))

        # D4: bloom_level của LO là INT trong Postgres. str(3) = "3" không khớp
        # bảng tên nên trước đây rơi về mặc định 2 và MỌI câu đều lưu mức 2.
        # Giữ hai dạng tách bạch: tên cho prompt, số cho cột bloom_level.
        bloom_level = (
            bloom_to_level(request.bloom_level)
            or bloom_to_level(lo_obj.bloom_level)
        )
        bloom = bloom_to_name(bloom_level) or "understand"
        if bloom_level is None:
            bloom_level = bloom_to_level(bloom)
            logger.info(
                "generate_curriculum_quiz.bloom_defaulted",
                lo_id=primary_lo_id,
                reason="LO không có Bloom và request không chỉ định",
                bloom=bloom,
            )
        context_result = self._load_context(
            request=request,
            primary_lo_id=primary_lo_id,
            lo_code=lo_obj.code,
            lo_statement=lo_obj.statement_vi,
            bloom=bloom,
        )
        if context_result.is_err():
            return Err(context_result.error)
        context_parts, source_chunk_ids, source_document_id = context_result.unwrap()
        if not context_parts or not source_chunk_ids or not source_document_id:
            return Err(
                ProcessingError(
                    f"No grounded source chunks available for learning outcome {primary_lo_id}"
                )
            )

        # 4. Build prompt
        prompt = prompts.generation(
            course_title=stored_course.title_vi,
            lo_code=lo_obj.code,
            lo_vi=lo_obj.statement_vi,
            lo_en=lo_obj.statement_en,
            bloom=bloom,
            style=request.style,
            count=request.count,
            context_parts=context_parts,
            question_type=question_type,
        )

        # 5. Generate via LLM
        started = time.monotonic()
        usage = _UsageAccumulator()
        payload_result = generate_structured_payload(
            llm_client=self._writer.client,
            prompt=prompt.user,
            system=prompt.system,
            schema_model=_CurriculumQuizPayload[question_model],
            on_usage=usage,
        )
        self._log_usage(
            role=self._writer,
            use_case=prompt.name,
            status=_USAGE_OK if payload_result.is_ok() else _USAGE_ERROR,
            latency_ms=int((time.monotonic() - started) * 1000),
            trace_id=trace_id,
            course_id=request.course_id,
            usage=usage.total,
        )
        if payload_result.is_err():
            return Err(payload_result.error)

        payload = payload_result.unwrap()

        # 5b. Verify each question against the retrieved source excerpts and give
        # the model a bounded number of chances to fix anything unsupported
        # before it ever reaches a human reviewer (doc §4.4/§4.5: CoVe + Reflexion).
        questions, rejected = self._verify_and_refine(
            questions=list(payload.questions[: request.count]),
            context_parts=context_parts,
            trace_id=trace_id,
            course_id=request.course_id,
            question_type=question_type,
            schema=(question_model, schema_name),
        )
        if rejected:
            logger.warning(
                "generate_curriculum_quiz.rejected_not_persisted",
                trace_id=trace_id,
                lo_id=primary_lo_id,
                rejected=len(rejected),
                reasons=[r for _, r in rejected][:3],
            )
        if not questions:
            return Err(ProcessingError(
                f"Tất cả {len(rejected)} câu đều không bám được nguồn sau "
                f"{self._max_verification_retries} lượt sửa; không lưu câu nào "
                f"cho {primary_lo_id}."
            ))

        # 6. Persist quiz_items with lo_id + bloom_level
        question_ids: list[str] = []
        stored_items: list[StoredQuizItem] = []

        primary_chunk_id = source_chunk_ids[0]
        targets = balanced_positions(len(questions), options=4, seed=primary_lo_id)

        for idx, q in enumerate(questions):
            # O3: id tất định theo (LO, nội dung câu hỏi). Trước đây là uuid4 nên
            # một lần retry sau bước lưu sẽ nhân đôi cả bộ câu. Kiểu khác MCQ_SINGLE
            # thêm kiểu vào khoá để không đè lên câu cùng lời của kiểu khác.
            key = q.question.strip()
            if question_type != "MCQ_SINGLE":
                key = f"{question_type}|{key}"
            qid = str(uuid.uuid5(_QUIZ_ID_NAMESPACE, f"{primary_lo_id}|{key}"))
            question_ids.append(qid)
            valid_source_ids = tuple(
                source_id
                for source_id in q.source_chunk_ids
                if source_id in source_chunk_ids
            )
            choices, correct_index, correct_answer = _answer_layout(
                q, question_type, seed=qid, target_index=targets[idx]
            )
            item = StoredQuizItem(
                question_id=qid,
                document_id=source_document_id,
                primary_chunk_id=primary_chunk_id,
                source_chunk_ids=valid_source_ids or tuple(source_chunk_ids[:3]),
                heading_path=(),
                question=q.question,
                choices=tuple(choices),
                correct_index=correct_index,
                explanation=q.explanation,
                difficulty=q.difficulty,
                question_index=idx,
                model_id=self._writer.client.model_id,
                question_type=question_type,
                correct_answer=correct_answer,
            )
            stored_items.append(item)

        persist_result = self._metadata_store.persist_curriculum_quiz_items(
            lo_id=primary_lo_id,
            bloom_level=bloom_level,
            quiz_items=stored_items,
        )
        if persist_result.is_err():
            logger.error(
                "generate_curriculum_quiz.persist_failed",
                lo_id=primary_lo_id,
                error=str(persist_result.error),
            )
            return Err(persist_result.error)

        logger.info(
            "generate_curriculum_quiz.done",
            course_id=request.course_id,
            lo_id=primary_lo_id,
            count=len(question_ids),
        )
        return Ok(GenerateCurriculumQuizResponse(
            course_id=request.course_id,
            lo_id=primary_lo_id,
            quiz_count=len(question_ids),
            question_ids=question_ids,
        ))

    def _load_context(
        self,
        request: GenerateCurriculumQuizRequest,
        primary_lo_id: str,
        lo_code: str,
        lo_statement: str,
        bloom: str,
    ) -> Result[tuple[list[str], list[str], str | None], Exception]:
        return self._context.load(
            course_id=request.course_id,
            lo_id=primary_lo_id,
            lo_code=lo_code,
            lo_statement=lo_statement,
            bloom=bloom,
            style=request.style,
            source_document_ids=request.source_document_ids,
        )

    def _resolve_lo_ids(
        self, request: GenerateCurriculumQuizRequest
    ) -> Result[list[str], Exception]:
        return self._context.resolve_lo_ids(
            request.course_id, request.target_kind, request.target_code
        )

    # ------------------------------------------------------------------
    # Verification + Reflexion
    # ------------------------------------------------------------------

    def _verify_and_refine(
        self,
        *,
        questions: list[_QuestionBase],
        context_parts: list[str],
        trace_id: str,
        course_id: str,
        question_type: str = "MCQ_SINGLE",
        schema: tuple[type[_QuestionBase], str] = _QUESTION_MODELS["MCQ_SINGLE"],
    ) -> tuple[list[_QuestionBase], list[tuple[_QuestionBase, str]]]:
        """
        Đối chiếu từng câu với đoạn nguồn, cho model tối đa
        ``max_verification_retries`` lượt sửa, rồi trả về (giữ lại, bị loại).

        V2: câu vẫn bị loại sau khi hết lượt thì **không lưu**. Trước đây nó vẫn
        được lưu kèm một dòng log, tức là đẩy việc phát hiện sai sang người
        duyệt — ngược với mục đích của vòng kiểm tra này.

        Verifier lỗi (LLM/parse hỏng) vẫn coi là không kết luận được, không phải
        lý do để giữ lại nội dung: người duyệt trong `review.service.ts` vẫn là
        cổng cuối.
        """
        def find_failing(current: list[_QuestionBase]) -> dict[int, str] | None:
            verdicts = self._run_verifier(
                questions=current,
                context_parts=context_parts,
                trace_id=trace_id,
                course_id=course_id,
                question_type=question_type,
            )
            if verdicts is None:
                return None
            return {v.question_index: v.reason for v in verdicts if v.verdict != "supported"}

        return verify_and_refine(
            questions,
            find_failing=find_failing,
            regenerate=lambda original, reason: self._run_regeneration(
                original=original,
                reason=reason,
                context_parts=context_parts,
                trace_id=trace_id,
                course_id=course_id,
                question_type=question_type,
                schema=schema,
            ),
            max_retries=self._max_verification_retries,
            on_exhausted=lambda still_failing: logger.warning(
                "generate_curriculum_quiz.verify.exhausted",
                trace_id=trace_id,
                still_failing=still_failing,
            ),
        )

    def _run_verifier(
        self,
        *,
        questions: list[_QuestionBase],
        context_parts: list[str],
        trace_id: str,
        course_id: str,
        question_type: str = "MCQ_SINGLE",
    ) -> list[_QuestionVerdict] | None:
        prompt = prompts.verification(context_parts, questions, question_type)
        started = time.monotonic()
        usage = _UsageAccumulator()
        result = generate_structured_payload(
            llm_client=self._verifier.client,
            prompt=prompt.user,
            system=prompt.system,
            schema_model=_QuizVerificationPayload,
            repair_schema_name="QuizVerification",
            on_usage=usage,
        )
        self._log_usage(
            role=self._verifier,
            use_case=prompt.name,
            status=_USAGE_OK if result.is_ok() else _USAGE_ERROR,
            latency_ms=int((time.monotonic() - started) * 1000),
            trace_id=trace_id,
            course_id=course_id,
            usage=usage.total,
        )
        if result.is_err():
            logger.warning(
                "generate_curriculum_quiz.verify.failed",
                trace_id=trace_id,
                error=str(result.error),
            )
            return None
        return result.unwrap().verdicts

    def _run_regeneration(
        self,
        *,
        original: _QuestionBase,
        reason: str,
        context_parts: list[str],
        trace_id: str,
        course_id: str,
        question_type: str = "MCQ_SINGLE",
        schema: tuple[type[_QuestionBase], str] = _QUESTION_MODELS["MCQ_SINGLE"],
    ) -> _QuestionBase | None:
        prompt = prompts.regeneration(context_parts, original, reason, question_type)
        schema_model, schema_name = schema
        started = time.monotonic()
        usage = _UsageAccumulator()
        result = generate_structured_payload(
            llm_client=self._writer.client,
            prompt=prompt.user,
            system=prompt.system,
            schema_model=schema_model,
            repair_schema_name=schema_name,
            on_usage=usage,
        )
        self._log_usage(
            role=self._writer,
            use_case=prompt.name,
            status=_USAGE_OK if result.is_ok() else _USAGE_ERROR,
            latency_ms=int((time.monotonic() - started) * 1000),
            trace_id=trace_id,
            course_id=course_id,
            usage=usage.total,
        )
        if result.is_err():
            logger.warning(
                "generate_curriculum_quiz.regenerate.failed",
                trace_id=trace_id,
                error=str(result.error),
            )
            return None
        return result.unwrap()

    def _log_usage(
        self,
        *,
        role: LlmRole,
        use_case: str,
        status: str,
        latency_ms: int,
        trace_id: str,
        course_id: str,
        usage: LLMUsage | None = None,
    ) -> None:
        """
        Ghi chi phí/độ trễ, không bao giờ chặn việc sinh quiz.

        ``status`` phải viết hoa: CHECK của `llm_usage_logs` chỉ nhận
        'OK' | 'RATE_LIMITED' | 'ERROR' và so sánh phân biệt hoa thường, nên
        "ok" làm mọi lần ghi thất bại và chỉ để lại một dòng warning.
        """
        usage = usage or LLMUsage()
        result = self._metadata_store.record_llm_usage(
            provider=role.provider,
            model=role.client.model_id,
            use_case=use_case,
            status=status,
            course_id=course_id,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=usage.cost_usd,
            latency_ms=latency_ms,
            trace_id=trace_id,
        )
        if result.is_err():
            logger.warning(
                "generate_curriculum_quiz.usage_log_failed",
                use_case=use_case,
                profile=role.profile,
                error=str(result.error),
            )
