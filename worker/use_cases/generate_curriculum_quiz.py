"""
GenerateCurriculumQuizUseCase — generate quiz items aligned to a specific LO/chapter/assessment.

Sibling to RunEnrichmentUseCase (course-scoped, not document-scoped).
Used by the queued content-generation worker and the legacy synchronous adapter.
"""
import time
import uuid
from typing import Literal, Protocol

from pydantic import BaseModel, Field, field_validator

from document_chunk.application.dto.generation_dto import (
    GenerateCurriculumQuizRequest,
    GenerateCurriculumQuizResponse,
)
from document_chunk.application.services.answer_layout import (
    balanced_positions,
    shuffle_choices,
)
from document_chunk.application.services.structured_output import generate_structured_payload
from document_chunk.domain.bloom import bloom_to_level, bloom_to_name
from document_chunk.domain.exceptions import ProcessingError
from document_chunk.domain.ports.llm_client import ILLMClient, LLMUsage
from document_chunk.domain.ports.metadata_store import IMetadataStore, StoredQuizItem
from document_chunk.shared.logger import get_logger
from document_chunk.shared.result import Err, Ok, Result

logger = get_logger(__name__)

# Namespace cố định để quiz_id tất định theo (LO, nội dung câu hỏi): worker retry
# sau khi đã lưu sẽ upsert đè lên chính dòng cũ thay vì sinh thêm một bộ trùng.
_QUIZ_ID_NAMESPACE = uuid.UUID("6f9b5f1e-3a4d-5c7b-8e2f-1d0a9c8b7e64")

# llm_usage_logs.status CHECK chỉ nhận ba giá trị này, phân biệt hoa thường.
_USAGE_OK = "OK"
_USAGE_ERROR = "ERROR"

_QUIZ_SYSTEM_PROMPT = (
    "You are an assessment writer for a university course. "
    "Return strict JSON only."
)

_VERIFIER_SYSTEM_PROMPT = (
    "You are a strict fact-checker for a university assessment. "
    "Return strict JSON only."
)

_REGEN_SYSTEM_PROMPT = (
    "You are an assessment writer fixing one rejected question. "
    "Return strict JSON only."
)


class _UsageAccumulator:
    """Cộng dồn token của mọi lần gọi trong một bước (kể cả lần sửa JSON)."""

    def __init__(self) -> None:
        self.total = LLMUsage()

    def __call__(self, usage: LLMUsage) -> None:
        self.total = self.total + usage


class QuizContextClient(Protocol):
    def retrieve_quiz_context(
        self,
        course_id: str,
        lo_code: str,
        query: str | None = None,
        bloom_level: str | None = None,
        assessment_style: str = "quiz",
        top_k: int = 5,
    ):
        ...


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class _CurriculumQuestion(BaseModel):
    question: str
    choices: list[str] = Field(min_length=4, max_length=4)
    correct_index: int = Field(ge=0, le=3)
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

    @field_validator("choices")
    @classmethod
    def _normalize_choices(cls, choices: list[str]) -> list[str]:
        normalized = [c.strip() for c in choices if c and c.strip()]
        if len(normalized) != 4:
            raise ValueError("choices must contain exactly 4 non-empty items")
        return normalized

    @field_validator("difficulty")
    @classmethod
    def _normalize_difficulty(cls, value: str) -> str:
        v = value.strip().lower()
        if v not in {"easy", "medium", "hard"}:
            return "medium"
        return v


class _CurriculumQuizPayload(BaseModel):
    questions: list[_CurriculumQuestion] = Field(min_length=1)


class _QuestionVerdict(BaseModel):
    question_index: int
    verdict: Literal["supported", "contradicted", "not_enough_evidence"]
    reason: str = ""


class _QuizVerificationPayload(BaseModel):
    verdicts: list[_QuestionVerdict] = Field(min_length=1)


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
        max_verification_retries: int = 2,
    ) -> None:
        self._metadata_store = metadata_store
        self._llm_client = llm_client
        self._max_context_chars = max_context_chars
        self._context_client = context_client
        self._fallback_to_local_context = fallback_to_local_context
        self._llm_provider = llm_provider
        self._max_verification_retries = max_verification_retries

    def execute(
        self, request: GenerateCurriculumQuizRequest
    ) -> Result[GenerateCurriculumQuizResponse, Exception]:
        trace_id = str(uuid.uuid4())

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
        prompt = self._build_prompt(
            course_title=stored_course.title_vi,
            lo_code=lo_obj.code,
            lo_vi=lo_obj.statement_vi,
            lo_en=lo_obj.statement_en,
            bloom=bloom,
            style=request.style,
            count=request.count,
            context_parts=context_parts,
        )

        # 5. Generate via LLM
        started = time.monotonic()
        usage = _UsageAccumulator()
        payload_result = generate_structured_payload(
            llm_client=self._llm_client,
            prompt=prompt,
            system=_QUIZ_SYSTEM_PROMPT,
            schema_model=_CurriculumQuizPayload,
            on_usage=usage,
        )
        self._log_usage(
            use_case="quiz_generation",
            status=_USAGE_OK if payload_result.is_ok() else _USAGE_ERROR,
            latency_ms=int((time.monotonic() - started) * 1000),
            trace_id=trace_id,
            course_id=request.course_id,
            usage=usage.total,
        )
        if payload_result.is_err():
            return Err(payload_result.error)

        payload: _CurriculumQuizPayload = payload_result.unwrap()

        # 5b. Verify each question against the retrieved source excerpts and give
        # the model a bounded number of chances to fix anything unsupported
        # before it ever reaches a human reviewer (doc §4.4/§4.5: CoVe + Reflexion).
        questions, rejected = self._verify_and_refine(
            questions=list(payload.questions[: request.count]),
            context_parts=context_parts,
            trace_id=trace_id,
            course_id=request.course_id,
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
            # một lần retry sau bước lưu sẽ nhân đôi cả bộ câu.
            qid = str(uuid.uuid5(_QUIZ_ID_NAMESPACE, f"{primary_lo_id}|{q.question.strip()}"))
            question_ids.append(qid)
            valid_source_ids = tuple(
                source_id
                for source_id in q.source_chunk_ids
                if source_id in source_chunk_ids
            )
            # G3: xáo phương án, vị trí đáp án đúng phân bố đều trong mẻ.
            choices, correct_index = shuffle_choices(
                list(q.choices),
                q.correct_index,
                seed=qid,
                target_index=targets[idx],
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
                model_id=self._llm_client.model_id,
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
        # Module launches must use only the selected chapter's lecture sources.
        # Generic MCP retrieval has no document filter, so use local LO mappings
        # for this explicit scope and never fall back to another chapter.
        if self._context_client is not None and request.source_document_ids is None:
            try:
                context = self._context_client.retrieve_quiz_context(
                    course_id=request.course_id,
                    lo_code=lo_code,
                    query=lo_statement,
                    bloom_level=bloom,
                    assessment_style=request.style,
                    top_k=5,
                )
                parts: list[str] = []
                source_ids: list[str] = []
                source_document_id: str | None = None
                total_chars = 0
                for index, chunk in enumerate(context.chunks, start=1):
                    content = chunk.content.strip()
                    if not content:
                        continue
                    snippet = (
                        f"[{index}] (chunk_id={chunk.chunk_id}, "
                        f"page={chunk.page_number or 'n/a'})\n{content}"
                    )
                    if parts and total_chars + len(snippet) > self._max_context_chars:
                        break
                    parts.append(snippet)
                    source_ids.append(chunk.chunk_id)
                    source_document_id = source_document_id or chunk.document_id
                    total_chars += len(snippet)
                if parts:
                    logger.info(
                        "generate_curriculum_quiz.context.mcp",
                        lo_id=primary_lo_id,
                        chunks=len(parts),
                    )
                    return Ok((parts, source_ids, source_document_id))
            except Exception as exc:
                if not self._fallback_to_local_context:
                    return Err(
                        ProcessingError(
                            "MCP quiz-context retrieval failed",
                            cause=exc,
                        )
                    )
                logger.warning(
                    "generate_curriculum_quiz.context.mcp_failed",
                    lo_id=primary_lo_id,
                    error=str(exc),
                )

        chunks_result = self._metadata_store.list_chunks_for_lo(primary_lo_id)
        if chunks_result.is_err():
            return Err(chunks_result.error)

        context_parts: list[str] = []
        total_chars = 0
        source_chunk_ids: list[str] = []
        local_document_id: str | None = None
        candidates = chunks_result.unwrap()
        if request.source_document_ids is not None:
            # psycopg returns UUID objects even though the metadata port types
            # IDs as strings. Normalize before comparing with JSON job IDs.
            allowed_documents = {str(value) for value in request.source_document_ids}
            candidates = [c for c in candidates if str(c.document_id) in allowed_documents]
        for index, chunk in enumerate(candidates[:5], start=1):
            text = (chunk.content_text or "").strip()
            if not text:
                continue
            snippet = f"[{index}] (chunk_id={chunk.chunk_id})\n{text[:1200]}"
            if context_parts and total_chars + len(snippet) > self._max_context_chars:
                break
            context_parts.append(snippet)
            source_chunk_ids.append(str(chunk.chunk_id))
            local_document_id = local_document_id or str(chunk.document_id)
            total_chars += len(snippet)
        return Ok((context_parts, source_chunk_ids, local_document_id))

    def _resolve_lo_ids(
        self, request: GenerateCurriculumQuizRequest
    ) -> Result[list[str], Exception]:
        kind = request.target_kind
        code = request.target_code
        course_id = request.course_id

        if kind == "lo":
            full_code = code if code.startswith("L.O.") else f"L.O.{code}"
            curriculum_result = self._metadata_store.get_curriculum(course_id)
            if curriculum_result.is_err():
                return Err(curriculum_result.error)
            curriculum = curriculum_result.unwrap()
            if curriculum is None:
                return Ok([])
            _, _, learning_outcomes, _ = curriculum
            matching = [
                lo.lo_id
                for lo in learning_outcomes
                if lo.code == full_code or lo.lo_id == code
            ]
            return Ok(matching)

        if kind == "chapter":
            result = self._metadata_store.list_los_by_chapter(course_id, code)
            if result.is_err():
                return Err(result.error)
            return Ok([lo.lo_id for lo in result.unwrap()])

        if kind == "assessment":
            result = self._metadata_store.list_los_by_assessment(course_id, code)
            if result.is_err():
                return Err(result.error)
            return Ok([lo.lo_id for lo in result.unwrap()])

        return Err(ProcessingError(f"Unknown target_kind: {kind!r}"))

    def _build_prompt(
        self,
        course_title: str,
        lo_code: str,
        lo_vi: str,
        lo_en: str | None,
        bloom: str,
        style: str,
        count: int,
        context_parts: list[str],
    ) -> str:
        style_desc = {
            "quiz": "quick recall / comprehension",
            "midterm": "mixed application and analysis",
            "final": "integrative synthesis across topics",
        }.get(style, style)

        lo_line = f"{lo_code} — {lo_vi}"
        if lo_en:
            lo_line += f" ({lo_en})"

        context_text = "\n\n".join(context_parts) if context_parts else "(no source excerpts available)"

        return (
            f"You are creating an assessment item for course: {course_title}\n"
            f"Target Learning Outcome: {lo_line}\n"
            f"Bloom level required: {bloom}\n"
            f"Assessment style: {style} ({style_desc})\n"
            "Use the language of the source content (Vietnamese unless source is English).\n\n"
            f"Source excerpts (each with chunk_id):\n{context_text}\n\n"
            f"Generate {count} multiple-choice questions strictly aligned to the LO.\n"
            "Each question MUST have exactly 4 choices, one correct_index (0-3), "
            "an explanation, and a lo_alignment_rationale.\n"
            "Return strict JSON:\n"
            '{"questions":[{"question":"...","choices":["...","...","...","..."],'
            '"correct_index":0,"explanation":"...","difficulty":"easy|medium|hard",'
            '"lo_alignment_rationale":"...","source_chunk_ids":["..."]}]}'
        )

    # ------------------------------------------------------------------
    # Verification + Reflexion
    # ------------------------------------------------------------------

    def _verify_and_refine(
        self,
        *,
        questions: list[_CurriculumQuestion],
        context_parts: list[str],
        trace_id: str,
        course_id: str,
    ) -> tuple[list[_CurriculumQuestion], list[tuple[_CurriculumQuestion, str]]]:
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
        current = list(questions)
        if not current:
            return current, []

        for attempt in range(self._max_verification_retries + 1):
            verdicts = self._run_verifier(
                questions=current,
                context_parts=context_parts,
                trace_id=trace_id,
                course_id=course_id,
            )
            if verdicts is None:
                return current, []

            failing = {
                v.question_index: v.reason
                for v in verdicts
                if v.verdict != "supported" and 0 <= v.question_index < len(current)
            }
            if not failing:
                return current, []

            if attempt == self._max_verification_retries:
                logger.warning(
                    "generate_curriculum_quiz.verify.exhausted",
                    trace_id=trace_id,
                    still_failing=sorted(failing.keys()),
                )
                rejected = [(current[i], reason) for i, reason in sorted(failing.items())]
                kept = [q for i, q in enumerate(current) if i not in failing]
                return kept, rejected

            for index, reason in failing.items():
                fixed = self._run_regeneration(
                    original=current[index],
                    reason=reason,
                    context_parts=context_parts,
                    trace_id=trace_id,
                    course_id=course_id,
                )
                if fixed is not None:
                    current[index] = fixed

        return current, []

    def _run_verifier(
        self,
        *,
        questions: list[_CurriculumQuestion],
        context_parts: list[str],
        trace_id: str,
        course_id: str,
    ) -> list[_QuestionVerdict] | None:
        prompt = self._build_verification_prompt(context_parts, questions)
        started = time.monotonic()
        usage = _UsageAccumulator()
        result = generate_structured_payload(
            llm_client=self._llm_client,
            prompt=prompt,
            system=_VERIFIER_SYSTEM_PROMPT,
            schema_model=_QuizVerificationPayload,
            repair_schema_name="QuizVerification",
            on_usage=usage,
        )
        self._log_usage(
            use_case="quiz_verification",
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
        original: _CurriculumQuestion,
        reason: str,
        context_parts: list[str],
        trace_id: str,
        course_id: str,
    ) -> _CurriculumQuestion | None:
        prompt = self._build_regeneration_prompt(context_parts, original, reason)
        started = time.monotonic()
        usage = _UsageAccumulator()
        result = generate_structured_payload(
            llm_client=self._llm_client,
            prompt=prompt,
            system=_REGEN_SYSTEM_PROMPT,
            schema_model=_CurriculumQuestion,
            repair_schema_name="CurriculumQuestion",
            on_usage=usage,
        )
        self._log_usage(
            use_case="quiz_regeneration",
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

    def _build_verification_prompt(
        self,
        context_parts: list[str],
        questions: list[_CurriculumQuestion],
    ) -> str:
        context_text = (
            "\n\n".join(context_parts) if context_parts else "(no source excerpts available)"
        )
        questions_text = "\n\n".join(
            f"[{idx}] Question: {q.question}\n"
            f"Choices: {list(q.choices)}\n"
            "Marked correct: "
            f"{q.choices[q.correct_index] if 0 <= q.correct_index < len(q.choices) else '(invalid index)'}\n"
            f"Explanation: {q.explanation}"
            for idx, q in enumerate(questions)
        )
        return (
            "You are checking a multiple-choice quiz for factual correctness. "
            "Using ONLY the source excerpts below (not outside knowledge), check "
            "each question.\n\n"
            f"Source excerpts:\n{context_text}\n\n"
            f"Questions to check:\n{questions_text}\n\n"
            "For each question index, decide:\n"
            '- "supported": the marked-correct choice and explanation are accurate '
            "per the source, AND the other three choices are clearly wrong (not "
            "arguably correct too).\n"
            '- "contradicted": the marked-correct choice is wrong, the explanation '
            "misstates the source, or another choice is also defensible as correct.\n"
            '- "not_enough_evidence": the source excerpts do not cover this enough '
            "to check.\n"
            "Return strict JSON:\n"
            '{"verdicts":[{"question_index":0,'
            '"verdict":"supported|contradicted|not_enough_evidence","reason":"..."}]}'
        )

    def _build_regeneration_prompt(
        self,
        context_parts: list[str],
        original: _CurriculumQuestion,
        reason: str,
    ) -> str:
        context_text = (
            "\n\n".join(context_parts) if context_parts else "(no source excerpts available)"
        )
        return (
            "A fact-checker rejected the following multiple-choice question because "
            "it is not properly grounded in the source excerpts below. Rewrite it so "
            "the problem is fixed. Keep the same topic and difficulty; you may change "
            "the wording, choices, or correct_index as needed.\n\n"
            f"Source excerpts:\n{context_text}\n\n"
            f"Rejected question: {original.question}\n"
            f"Rejected choices: {list(original.choices)}\n"
            f"Fact-checker feedback: {reason}\n\n"
            "Return strict JSON for ONE corrected question:\n"
            '{"question":"...","choices":["...","...","...","..."],"correct_index":0,'
            '"explanation":"...","difficulty":"easy|medium|hard",'
            '"lo_alignment_rationale":"...","source_chunk_ids":["..."]}'
        )

    def _log_usage(
        self,
        *,
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
            provider=self._llm_provider,
            model=self._llm_client.model_id,
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
                error=str(result.error),
            )
