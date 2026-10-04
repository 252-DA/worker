"""
GenerateLessonCardsUseCase — sinh micro-content card cho một Learning Outcome.

Anh em với GenerateCurriculumQuizUseCase: cùng cách chọn LO, cùng đoạn nguồn,
cùng vòng kiểm chứng trước khi lưu. Thay cho đường sinh card trong enrichment,
vốn chạy theo từng section tài liệu, không biết card thuộc LO nào, và làm mất
card im lặng khi chunk chưa được map LO.

Một card là một ý, đọc trong 1–3 phút, và mọi câu trong đó phải có trong đoạn
nguồn. ``check_question``/``check_answer`` là mặt lật của flashcard.
"""
import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from document_chunk.application.services.structured_output import generate_structured_payload
from document_chunk.domain.bloom import bloom_to_name
from document_chunk.domain.exceptions import ProcessingError
from document_chunk.domain.ports.llm_client import ILLMClient, LLMUsage
from document_chunk.domain.ports.metadata_store import IMetadataStore, StoredLessonCard
from document_chunk.shared.logger import get_logger
from document_chunk.shared.result import Err, Ok, Result

from worker.llm_role import LlmRole
from worker.use_cases.lo_context import LoContextLoader, QuizContextClient
from worker.use_cases.verification import verify_and_refine

logger = get_logger(__name__)

# Namespace cố định để card_id tất định theo (LO, tiêu đề): worker retry sau
# khi đã lưu sẽ ghi đè đúng bản nháp cũ, và hai lần sinh ra cùng một ý cho
# cùng LO không thành hai card.
_CARD_ID_NAMESPACE = uuid.UUID("0c7e2b5d-9f14-5a63-b8d1-4e6f2a9c3b71")

DEFAULT_CARD_COUNT = 4
MAX_CARD_COUNT = 6

# Giới hạn cứng, rộng hơn mức prompt yêu cầu: vượt nghĩa là model đã viết một
# đoạn văn chứ không phải một card, và bước sửa JSON được thấy giới hạn này.
_MAX_TITLE_CHARS = 150
_MAX_SENTENCE_CHARS = 400
_MAX_BULLET_CHARS = 300
_MAX_EXAMPLE_CHARS = 600
_MAX_BULLETS = 4

# Prompt cho "... or null" nên model hay trả chuỗi thay vì null thật.
_NULL_STRINGS = {"", "null", "none", "n/a", "na", "-", "không có"}

# llm_usage_logs.status CHECK chỉ nhận ba giá trị này, phân biệt hoa thường.
_USAGE_OK = "OK"
_USAGE_ERROR = "ERROR"

_CARDS_SYSTEM_PROMPT = (
    "You are an instructional designer writing micro-learning cards for a university course. "
    "Return strict JSON only."
)

_VERIFIER_SYSTEM_PROMPT = (
    "You are a strict fact-checker for university learning material. "
    "Return strict JSON only."
)

_REGEN_SYSTEM_PROMPT = (
    "You are an instructional designer fixing one rejected micro-learning card. "
    "Return strict JSON only."
)


class _UsageAccumulator:
    """Cộng dồn token của mọi lần gọi trong một bước (kể cả lần sửa JSON)."""

    def __init__(self) -> None:
        self.total = LLMUsage()

    def __call__(self, usage: LLMUsage) -> None:
        self.total = self.total + usage


def _required_text(value: str, limit: int) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("value must not be empty")
    if len(normalized) > limit:
        raise ValueError(f"value must be at most {limit} characters")
    return normalized


def _optional_text(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if normalized.lower() in _NULL_STRINGS:
        return None
    if len(normalized) > limit:
        raise ValueError(f"value must be at most {limit} characters")
    return normalized


def _title_key(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip().casefold()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class _LessonCard(BaseModel):
    title: str
    key_insight: str
    bullets: list[str] = Field(min_length=1, max_length=_MAX_BULLETS)
    example: str | None = None
    misconception: str | None = None
    check_question: str
    check_answer: str
    source_chunk_ids: list[str] = Field(default_factory=list)

    @field_validator("title")
    @classmethod
    def _title(cls, value: str) -> str:
        return _required_text(value, _MAX_TITLE_CHARS)

    @field_validator("key_insight", "check_question", "check_answer")
    @classmethod
    def _sentence(cls, value: str) -> str:
        return _required_text(value, _MAX_SENTENCE_CHARS)

    @field_validator("bullets")
    @classmethod
    def _bullets(cls, bullets: list[str]) -> list[str]:
        normalized = [b.strip() for b in bullets if b and b.strip()]
        if not normalized:
            raise ValueError("bullets must contain at least one item")
        for bullet in normalized:
            if len(bullet) > _MAX_BULLET_CHARS:
                raise ValueError(f"each bullet must be at most {_MAX_BULLET_CHARS} characters")
        return normalized

    @field_validator("example")
    @classmethod
    def _example(cls, value: str | None) -> str | None:
        return _optional_text(value, _MAX_EXAMPLE_CHARS)

    @field_validator("misconception")
    @classmethod
    def _misconception(cls, value: str | None) -> str | None:
        return _optional_text(value, _MAX_SENTENCE_CHARS)


class _LessonCardsPayload(BaseModel):
    cards: list[_LessonCard] = Field(min_length=1)


class _CardVerdict(BaseModel):
    card_index: int
    verdict: Literal["supported", "contradicted", "not_enough_evidence"]
    reason: str = ""


class _CardVerificationPayload(BaseModel):
    verdicts: list[_CardVerdict] = Field(min_length=1)


# ---------------------------------------------------------------------------
# Request / response
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenerateLessonCardsRequest:
    course_id: str
    target_kind: str
    target_code: str
    count: int = DEFAULT_CARD_COUNT
    source_document_ids: list[str] | None = None


@dataclass(frozen=True)
class GenerateLessonCardsResponse:
    course_id: str
    lo_id: str
    card_count: int
    card_ids: list[str]


# ---------------------------------------------------------------------------
# Use case
# ---------------------------------------------------------------------------

class GenerateLessonCardsUseCase:
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
            log_prefix="generate_lesson_cards",
        )
        self._max_verification_retries = max_verification_retries
        # Như quiz: người viết card và người chấm nên là hai model khác nhau.
        self._writer = LlmRole(client=llm_client, provider=llm_provider, profile=llm_profile)
        self._verifier = verifier or self._writer

    def execute(
        self, request: GenerateLessonCardsRequest
    ) -> Result[GenerateLessonCardsResponse, Exception]:
        if not 1 <= request.count <= MAX_CARD_COUNT:
            return Err(ProcessingError(
                f"count must be between 1 and {MAX_CARD_COUNT}, got {request.count}"
            ))
        trace_id = str(uuid.uuid4())

        target_result = self._context.resolve_target(
            request.course_id, request.target_kind, request.target_code
        )
        if target_result.is_err():
            return Err(target_result.error)
        target = target_result.unwrap()
        lo = target.lo

        context_result = self._context.load(
            course_id=request.course_id,
            lo_id=lo.lo_id,
            lo_code=lo.code,
            lo_statement=lo.statement_vi,
            bloom=bloom_to_name(lo.bloom_level),
            style="lesson",
            source_document_ids=request.source_document_ids,
        )
        if context_result.is_err():
            return Err(context_result.error)
        context_parts, source_chunk_ids, source_document_id = context_result.unwrap()
        if not context_parts or not source_chunk_ids or not source_document_id:
            return Err(ProcessingError(
                f"No grounded source chunks available for learning outcome {lo.lo_id}"
            ))

        prompt = self._build_prompt(
            course_title=target.course.title_vi,
            lo_code=lo.code,
            lo_vi=lo.statement_vi,
            lo_en=lo.statement_en,
            count=request.count,
            context_parts=context_parts,
        )
        started = time.monotonic()
        usage = _UsageAccumulator()
        payload_result = generate_structured_payload(
            llm_client=self._writer.client,
            prompt=prompt,
            system=_CARDS_SYSTEM_PROMPT,
            schema_model=_LessonCardsPayload,
            repair_schema_name="LessonCards",
            on_usage=usage,
        )
        self._log_usage(
            role=self._writer,
            use_case="card_generation",
            status=_USAGE_OK if payload_result.is_ok() else _USAGE_ERROR,
            latency_ms=int((time.monotonic() - started) * 1000),
            trace_id=trace_id,
            course_id=request.course_id,
            usage=usage.total,
        )
        if payload_result.is_err():
            return Err(payload_result.error)

        cards = self._drop_repeated_titles(payload_result.unwrap().cards[: request.count])

        cards, rejected = verify_and_refine(
            cards,
            find_failing=lambda current: self._find_failing(
                current, context_parts, trace_id, request.course_id
            ),
            regenerate=lambda original, reason: self._run_regeneration(
                original=original,
                reason=reason,
                context_parts=context_parts,
                trace_id=trace_id,
                course_id=request.course_id,
            ),
            max_retries=self._max_verification_retries,
            on_exhausted=lambda still_failing: logger.warning(
                "generate_lesson_cards.verify.exhausted",
                trace_id=trace_id,
                still_failing=still_failing,
            ),
        )
        if rejected:
            logger.warning(
                "generate_lesson_cards.rejected_not_persisted",
                trace_id=trace_id,
                lo_id=lo.lo_id,
                rejected=len(rejected),
                reasons=[reason for _, reason in rejected][:3],
            )
        # Một card sửa lại có thể trùng tiêu đề với card khác trong mẻ.
        cards = self._drop_repeated_titles(cards)
        if not cards:
            return Err(ProcessingError(
                f"Tất cả {len(rejected)} card đều không bám được nguồn sau "
                f"{self._max_verification_retries} lượt sửa; không lưu card nào "
                f"cho {lo.lo_id}."
            ))

        stored = [
            self._to_stored(
                card,
                index=index,
                lo_id=lo.lo_id,
                source_chunk_ids=source_chunk_ids,
                source_document_id=source_document_id,
            )
            for index, card in enumerate(cards)
        ]
        persist_result = self._metadata_store.persist_curriculum_lesson_cards(
            lo_id=lo.lo_id,
            lesson_cards=stored,
        )
        if persist_result.is_err():
            logger.error(
                "generate_lesson_cards.persist_failed",
                lo_id=lo.lo_id,
                error=str(persist_result.error),
            )
            return Err(persist_result.error)

        card_ids = [card.card_id for card in stored]
        logger.info(
            "generate_lesson_cards.done",
            course_id=request.course_id,
            lo_id=lo.lo_id,
            count=len(card_ids),
        )
        return Ok(GenerateLessonCardsResponse(
            course_id=request.course_id,
            lo_id=lo.lo_id,
            card_count=len(card_ids),
            card_ids=card_ids,
        ))

    # ------------------------------------------------------------------

    def _drop_repeated_titles(self, cards: list[_LessonCard]) -> list[_LessonCard]:
        seen: set[str] = set()
        unique: list[_LessonCard] = []
        for card in cards:
            key = _title_key(card.title)
            if key in seen:
                logger.info("generate_lesson_cards.duplicate_title_dropped", title=card.title)
                continue
            seen.add(key)
            unique.append(card)
        return unique

    def _to_stored(
        self,
        card: _LessonCard,
        *,
        index: int,
        lo_id: str,
        source_chunk_ids: list[str],
        source_document_id: str,
    ) -> StoredLessonCard:
        valid_source_ids = tuple(
            chunk_id for chunk_id in card.source_chunk_ids if chunk_id in source_chunk_ids
        ) or tuple(source_chunk_ids[:3])
        return StoredLessonCard(
            card_id=str(uuid.uuid5(_CARD_ID_NAMESPACE, f"{lo_id}|{_title_key(card.title)}")),
            document_id=source_document_id,
            primary_chunk_id=valid_source_ids[0],
            source_chunk_ids=valid_source_ids,
            title=card.title,
            bullets=tuple(card.bullets),
            key_insight=card.key_insight,
            example=card.example,
            misconception=card.misconception,
            check_question=card.check_question,
            check_answer=card.check_answer,
            card_index=index,
            model_id=self._writer.client.model_id,
        )

    def _build_prompt(
        self,
        course_title: str,
        lo_code: str,
        lo_vi: str,
        lo_en: str | None,
        count: int,
        context_parts: list[str],
    ) -> str:
        lo_line = f"{lo_code} — {lo_vi}"
        if lo_en:
            lo_line += f" ({lo_en})"
        context_text = "\n\n".join(context_parts)

        return (
            f"You are writing micro-learning cards for course: {course_title}\n"
            f"Target Learning Outcome: {lo_line}\n"
            "Use the language of the source content (Vietnamese unless source is English).\n\n"
            f"Source excerpts (each with chunk_id):\n{context_text}\n\n"
            f"Write {count} lesson cards that together help a learner reach this LO. "
            "Order them so that earlier cards introduce what later cards build on.\n"
            "Rules:\n"
            "- One idea per card. If a title needs \"and\", split it into two cards.\n"
            "- Cards must not repeat each other.\n"
            "- Every statement must be supported by the source excerpts. Do not add outside facts.\n"
            "- title: a short name for the idea, at most 10 words.\n"
            "- key_insight: the 1-2 sentences a learner must remember if they read nothing else.\n"
            f"- bullets: 2-{_MAX_BULLETS} short points explaining the idea, at most 25 words each.\n"
            "- example: a concrete example taken from the source, or null if the source has none.\n"
            "- misconception: a common misunderstanding, only if the source states or clearly "
            "implies it; otherwise null.\n"
            "- check_question: one short question a learner answers from memory to test this "
            "card. Not a yes/no question.\n"
            "- check_answer: the correct answer in 1-2 sentences.\n"
            "- source_chunk_ids: the chunk_ids this card is based on.\n"
            "Return strict JSON:\n"
            '{"cards":[{"title":"...","key_insight":"...","bullets":["...","..."],'
            '"example":"... or null","misconception":"... or null",'
            '"check_question":"...","check_answer":"...","source_chunk_ids":["..."]}]}'
        )

    # ------------------------------------------------------------------
    # Verification + Reflexion
    # ------------------------------------------------------------------

    def _find_failing(
        self,
        cards: list[_LessonCard],
        context_parts: list[str],
        trace_id: str,
        course_id: str,
    ) -> dict[int, str] | None:
        prompt = self._build_verification_prompt(context_parts, cards)
        started = time.monotonic()
        usage = _UsageAccumulator()
        result = generate_structured_payload(
            llm_client=self._verifier.client,
            prompt=prompt,
            system=_VERIFIER_SYSTEM_PROMPT,
            schema_model=_CardVerificationPayload,
            repair_schema_name="CardVerification",
            on_usage=usage,
        )
        self._log_usage(
            role=self._verifier,
            use_case="card_verification",
            status=_USAGE_OK if result.is_ok() else _USAGE_ERROR,
            latency_ms=int((time.monotonic() - started) * 1000),
            trace_id=trace_id,
            course_id=course_id,
            usage=usage.total,
        )
        if result.is_err():
            logger.warning(
                "generate_lesson_cards.verify.failed",
                trace_id=trace_id,
                error=str(result.error),
            )
            return None
        return {
            v.card_index: v.reason
            for v in result.unwrap().verdicts
            if v.verdict != "supported"
        }

    def _run_regeneration(
        self,
        *,
        original: _LessonCard,
        reason: str,
        context_parts: list[str],
        trace_id: str,
        course_id: str,
    ) -> _LessonCard | None:
        prompt = self._build_regeneration_prompt(context_parts, original, reason)
        started = time.monotonic()
        usage = _UsageAccumulator()
        result = generate_structured_payload(
            llm_client=self._writer.client,
            prompt=prompt,
            system=_REGEN_SYSTEM_PROMPT,
            schema_model=_LessonCard,
            repair_schema_name="LessonCard",
            on_usage=usage,
        )
        self._log_usage(
            role=self._writer,
            use_case="card_regeneration",
            status=_USAGE_OK if result.is_ok() else _USAGE_ERROR,
            latency_ms=int((time.monotonic() - started) * 1000),
            trace_id=trace_id,
            course_id=course_id,
            usage=usage.total,
        )
        if result.is_err():
            logger.warning(
                "generate_lesson_cards.regenerate.failed",
                trace_id=trace_id,
                error=str(result.error),
            )
            return None
        return result.unwrap()

    def _build_verification_prompt(
        self,
        context_parts: list[str],
        cards: list[_LessonCard],
    ) -> str:
        context_text = "\n\n".join(context_parts)
        cards_text = "\n\n".join(
            f"[{idx}] Title: {card.title}\n"
            f"Key insight: {card.key_insight}\n"
            f"Bullets: {card.bullets}\n"
            f"Example: {card.example or '(none)'}\n"
            f"Misconception: {card.misconception or '(none)'}\n"
            f"Check question: {card.check_question}\n"
            f"Check answer: {card.check_answer}"
            for idx, card in enumerate(cards)
        )
        return (
            "You are checking micro-learning cards for factual correctness. "
            "Using ONLY the source excerpts below (not outside knowledge), check "
            "each card.\n\n"
            f"Source excerpts:\n{context_text}\n\n"
            f"Cards to check:\n{cards_text}\n\n"
            "For each card index, decide:\n"
            '- "supported": every statement in the key insight, bullets, example and '
            "misconception is backed by the source, AND the check answer correctly "
            "answers the check question according to the source.\n"
            '- "contradicted": any statement misstates the source, or the check answer '
            "is wrong.\n"
            '- "not_enough_evidence": the card makes claims the source excerpts do not '
            "cover.\n"
            "Return strict JSON:\n"
            '{"verdicts":[{"card_index":0,'
            '"verdict":"supported|contradicted|not_enough_evidence","reason":"..."}]}'
        )

    def _build_regeneration_prompt(
        self,
        context_parts: list[str],
        original: _LessonCard,
        reason: str,
    ) -> str:
        context_text = "\n\n".join(context_parts)
        rejected = json.dumps(original.model_dump(), ensure_ascii=False)
        return (
            "A fact-checker rejected the following micro-learning card because it is "
            "not properly grounded in the source excerpts below. Rewrite it so the "
            "problem is fixed. Keep the same idea; remove anything the source does "
            "not support.\n\n"
            f"Source excerpts:\n{context_text}\n\n"
            f"Rejected card: {rejected}\n"
            f"Fact-checker feedback: {reason}\n\n"
            "Return strict JSON for ONE corrected card:\n"
            '{"title":"...","key_insight":"...","bullets":["...","..."],'
            '"example":"... or null","misconception":"... or null",'
            '"check_question":"...","check_answer":"...","source_chunk_ids":["..."]}'
        )

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
        """Ghi chi phí/độ trễ, không bao giờ chặn việc sinh card."""
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
                "generate_lesson_cards.usage_log_failed",
                use_case=use_case,
                profile=role.profile,
                error=str(result.error),
            )


__all__ = [
    "DEFAULT_CARD_COUNT",
    "MAX_CARD_COUNT",
    "GenerateLessonCardsRequest",
    "GenerateLessonCardsResponse",
    "GenerateLessonCardsUseCase",
]
