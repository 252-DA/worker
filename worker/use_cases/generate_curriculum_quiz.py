"""
GenerateCurriculumQuizUseCase — generate quiz items aligned to a specific LO/chapter/assessment.

Sibling to RunEnrichmentUseCase (course-scoped, not document-scoped).
Runs synchronously in MVP (called from gRPC handler).
"""
import uuid
from dataclasses import dataclass

from pydantic import BaseModel, Field, ValidationError, field_validator

from document_chunk.domain.exceptions import ProcessingError
from document_chunk.domain.ports.llm_client import ILLMClient
from document_chunk.domain.ports.metadata_store import IMetadataStore, StoredQuizItem
from document_chunk.shared.logger import get_logger
from document_chunk.shared.result import Err, Ok, Result
from worker.use_cases._llm_json import generate_structured_payload

logger = get_logger(__name__)

_QUIZ_SYSTEM_PROMPT = (
    "You are an assessment writer for a university course. "
    "Return strict JSON only."
)

_QUIZ_SCHEMA_HINT = (
    '{"questions":[{"question":"...","choices":["...","...","...","..."],'
    '"correct_index":0,"explanation":"...","difficulty":"easy|medium|hard",'
    '"lo_alignment_rationale":"..."}]}'
)


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
# Request / Response
# ---------------------------------------------------------------------------

@dataclass
class GenerateCurriculumQuizRequest:
    course_id: str
    target_kind: str    # "lo" | "chapter" | "assessment"
    target_code: str
    style: str = "quiz" # "quiz" | "midterm" | "final"
    bloom_level: str | None = None
    count: int = 5


@dataclass
class GenerateCurriculumQuizResponse:
    course_id: str
    lo_id: str
    quiz_count: int
    question_ids: list[str]


# ---------------------------------------------------------------------------
# Use case
# ---------------------------------------------------------------------------

class GenerateCurriculumQuizUseCase:
    def __init__(
        self,
        metadata_store: IMetadataStore,
        llm_client: ILLMClient,
        max_context_chars: int = 6000,
    ) -> None:
        self._metadata_store = metadata_store
        self._llm_client = llm_client
        self._max_context_chars = max_context_chars

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

        # 3. Fetch supporting chunks
        chunks_result = self._metadata_store.list_chunks_for_lo(primary_lo_id)
        if chunks_result.is_err():
            return Err(chunks_result.error)

        chunks = chunks_result.unwrap()

        # 4. Build prompt
        context_parts: list[str] = []
        total_chars = 0
        source_chunk_ids: list[str] = []
        for i, chunk in enumerate(chunks[:5], start=1):
            text = chunk.content_text or ""
            if not text.strip():
                continue
            snippet = f"[{i}] (chunk_id={chunk.chunk_id})\n{text[:1200]}"
            if total_chars + len(snippet) > self._max_context_chars:
                break
            context_parts.append(snippet)
            source_chunk_ids.append(chunk.chunk_id)
            total_chars += len(snippet)

        bloom = request.bloom_level or lo_obj.bloom_level or "understand"
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
            repair_schema_hint=_QUIZ_SCHEMA_HINT,
        )
        if payload_result.is_err():
            return Err(payload_result.error)

        payload: _CurriculumQuizPayload = payload_result.unwrap()

        # 6. Persist quiz_items with lo_id + bloom_level
        question_ids: list[str] = []
        stored_items: list[StoredQuizItem] = []

        dummy_doc_id = f"_curriculum_{request.course_id}"
        dummy_chunk_id = source_chunk_ids[0] if source_chunk_ids else f"_lo_{primary_lo_id}"

        for idx, q in enumerate(payload.questions[: request.count]):
            qid = str(uuid.uuid4())
            question_ids.append(qid)
            item = StoredQuizItem(
                question_id=qid,
                document_id=dummy_doc_id,
                primary_chunk_id=dummy_chunk_id,
                source_chunk_ids=tuple(q.source_chunk_ids or source_chunk_ids[:3]),
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

        # Persist individually (not grouped by document like run_enrichment)
        # Use upsert with lo_id via raw SQL workaround: store in standard quiz_items
        # and rely on extra columns added in schema
        persist_result = self._metadata_store.persist_enrichment_batch(
            document_id=dummy_doc_id,
            lesson_cards=[],
            quiz_items=stored_items,
            concepts=[],
            chunk_concepts=[],
        )
        if persist_result.is_err():
            logger.warning(
                "generate_curriculum_quiz.persist_failed",
                lo_id=primary_lo_id,
                error=str(persist_result.error),
            )

        # Update lo_id column separately (best-effort)
        self._set_lo_id_on_quiz_items(question_ids, primary_lo_id, bloom)

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

    def _resolve_lo_ids(
        self, request: GenerateCurriculumQuizRequest
    ) -> Result[list[str], Exception]:
        kind = request.target_kind
        code = request.target_code
        course_id = request.course_id

        if kind == "lo":
            full_code = code if code.startswith("L.O.") else f"L.O.{code}"
            return Ok([f"{course_id}:{full_code}"])

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

    def _set_lo_id_on_quiz_items(
        self,
        question_ids: list[str],
        lo_id: str,
        bloom_level: str | None,
    ) -> None:
        """Best-effort: update lo_id/bloom_level columns added by ALTER TABLE."""
        try:
            store = self._metadata_store
            # Only works for PostgresMetadataStore — skip gracefully for noop
            with store._connection() as conn:  # type: ignore[attr-defined]
                with conn.cursor() as cur:
                    cur.executemany(
                        "UPDATE quiz_items SET lo_id = %s, bloom_level = %s WHERE id = %s;",
                        [(lo_id, bloom_level, qid) for qid in question_ids],
                    )
                conn.commit()
        except Exception as exc:
            logger.warning("generate_curriculum_quiz.lo_id_update_failed", error=str(exc))
