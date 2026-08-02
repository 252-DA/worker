"""
GenerateCurriculumQuizUseCase — generate quiz items aligned to a specific LO/chapter/assessment.

Sibling to RunEnrichmentUseCase (course-scoped, not document-scoped).
Used by the queued content-generation worker and the legacy synchronous adapter.
"""
import uuid
from typing import Protocol

from pydantic import BaseModel, Field, field_validator

from document_chunk.application.dto.generation_dto import (
    GenerateCurriculumQuizRequest,
    GenerateCurriculumQuizResponse,
)
from document_chunk.application.services.structured_output import generate_structured_payload
from document_chunk.domain.exceptions import ProcessingError
from document_chunk.domain.ports.llm_client import ILLMClient
from document_chunk.domain.ports.metadata_store import IMetadataStore, StoredQuizItem
from document_chunk.shared.logger import get_logger
from document_chunk.shared.result import Err, Ok, Result

logger = get_logger(__name__)

_QUIZ_SYSTEM_PROMPT = (
    "You are an assessment writer for a university course. "
    "Return strict JSON only."
)


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
    ) -> None:
        self._metadata_store = metadata_store
        self._llm_client = llm_client
        self._max_context_chars = max_context_chars
        self._context_client = context_client
        self._fallback_to_local_context = fallback_to_local_context

    def execute(
        self, request: GenerateCurriculumQuizRequest
    ) -> Result[GenerateCurriculumQuizResponse, Exception]:
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

        bloom = str(request.bloom_level or lo_obj.bloom_level or "understand")
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
        payload_result = generate_structured_payload(
            llm_client=self._llm_client,
            prompt=prompt,
            system=_QUIZ_SYSTEM_PROMPT,
            schema_model=_CurriculumQuizPayload,
        )
        if payload_result.is_err():
            return Err(payload_result.error)

        payload: _CurriculumQuizPayload = payload_result.unwrap()

        # 6. Persist quiz_items with lo_id + bloom_level
        question_ids: list[str] = []
        stored_items: list[StoredQuizItem] = []

        primary_chunk_id = source_chunk_ids[0]

        for idx, q in enumerate(payload.questions[: request.count]):
            qid = str(uuid.uuid4())
            question_ids.append(qid)
            valid_source_ids = tuple(
                source_id
                for source_id in q.source_chunk_ids
                if source_id in source_chunk_ids
            )
            item = StoredQuizItem(
                question_id=qid,
                document_id=source_document_id,
                primary_chunk_id=primary_chunk_id,
                source_chunk_ids=valid_source_ids or tuple(source_chunk_ids[:3]),
                heading_path=(),
                question=q.question,
                choices=tuple(q.choices),
                correct_index=q.correct_index,
                explanation=q.explanation,
                difficulty=q.difficulty,
                question_index=idx,
                model_id=self._llm_client.model_id,
            )
            stored_items.append(item)

        persist_result = self._metadata_store.persist_curriculum_quiz_items(
            lo_id=primary_lo_id,
            bloom_level=bloom,
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
        if self._context_client is not None:
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
        for index, chunk in enumerate(chunks_result.unwrap()[:5], start=1):
            text = (chunk.content_text or "").strip()
            if not text:
                continue
            snippet = f"[{index}] (chunk_id={chunk.chunk_id})\n{text[:1200]}"
            if context_parts and total_chars + len(snippet) > self._max_context_chars:
                break
            context_parts.append(snippet)
            source_chunk_ids.append(chunk.chunk_id)
            local_document_id = local_document_id or chunk.document_id
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
