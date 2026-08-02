"""
RunEnrichmentUseCase — build heading concepts + generated lesson cards/quiz items.

Enrichment stays asynchronous and retryable:
  DONE -> ENRICHING -> ENRICHED

If enrichment fails, document status falls back to DONE so search remains available.
"""
import re
import unicodedata
import uuid
from dataclasses import dataclass

from pydantic import BaseModel, Field, field_validator

from document_chunk.application.services.structured_output import generate_structured_payload
from document_chunk.domain.exceptions import ProcessingError
from document_chunk.domain.outbox_events import OutboxEventType
from document_chunk.domain.ports.llm_client import ILLMClient
from document_chunk.domain.ports.metadata_store import (
    IMetadataStore,
    IngestionStatus,
    StoredChunkConcept,
    StoredChunkMetadata,
    StoredConcept,
    StoredLessonCard,
    StoredQuizItem,
)
from document_chunk.shared.logger import get_logger
from document_chunk.shared.result import Err, Ok, Result
from document_chunk.shared.tracing import get_tracer

logger = get_logger(__name__)
tracer = get_tracer(__name__)

_CARDS_SYSTEM_PROMPT = (
    "You are a teaching assistant that creates concise micro-learning lesson cards. "
    "Return strict JSON only."
)
_QUIZ_SYSTEM_PROMPT = (
    "You are an assessment writer that creates multiple-choice questions from academic content. "
    "Return strict JSON only."
)


class _GeneratedCard(BaseModel):
    title: str
    bullets: list[str] = Field(min_length=1, max_length=5)
    key_insight: str

    @field_validator("title", "key_insight")
    @classmethod
    def _must_not_be_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @field_validator("bullets")
    @classmethod
    def _normalize_bullets(cls, bullets: list[str]) -> list[str]:
        normalized = [bullet.strip() for bullet in bullets if bullet and bullet.strip()]
        if not normalized:
            raise ValueError("bullets must contain at least one item")
        return normalized


class _GeneratedCardsPayload(BaseModel):
    cards: list[_GeneratedCard] = Field(min_length=1, max_length=3)


class _GeneratedQuestion(BaseModel):
    question: str
    choices: list[str] = Field(min_length=4, max_length=4)
    correct_index: int = Field(ge=0, le=3)
    explanation: str
    difficulty: str = "medium"

    @field_validator("question", "explanation")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @field_validator("choices")
    @classmethod
    def _normalize_choices(cls, choices: list[str]) -> list[str]:
        normalized = [choice.strip() for choice in choices if choice and choice.strip()]
        if len(normalized) != 4:
            raise ValueError("choices must contain exactly 4 non-empty items")
        return normalized

    @field_validator("difficulty")
    @classmethod
    def _normalize_difficulty(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"easy", "medium", "hard"}:
            raise ValueError("difficulty must be easy, medium, or hard")
        return normalized


class _GeneratedQuizPayload(BaseModel):
    questions: list[_GeneratedQuestion] = Field(min_length=1, max_length=5)


@dataclass(frozen=True)
class _SectionContext:
    document_id: str
    heading_path: tuple[str, ...]
    title: str
    primary_chunk_id: str
    source_chunk_ids: tuple[str, ...]
    language: str | None
    content_text: str


@dataclass
class RunEnrichmentRequest:
    document_id: str


@dataclass
class RunEnrichmentResponse:
    document_id: str
    concept_count: int
    mention_count: int
    card_count: int
    quiz_count: int


class RunEnrichmentUseCase:
    def __init__(
        self,
        metadata_store: IMetadataStore,
        llm_client: ILLMClient,
        max_section_chars: int = 6000,
    ) -> None:
        self._metadata_store = metadata_store
        self._llm_client = llm_client
        self._max_section_chars = max_section_chars

    def execute(self, request: RunEnrichmentRequest) -> Result[RunEnrichmentResponse, Exception]:
        document_id = request.document_id

        with tracer.start_as_current_span("run_enrichment") as span:
            span.set_attribute("document.id", document_id)

            context_result = self._metadata_store.get_document_context(document_id)
            if context_result.is_err():
                return Err(context_result.error)

            context = context_result.unwrap()
            if context is None:
                return Err(ProcessingError(f"Document not found for enrichment: {document_id}"))

            status_result = self._metadata_store.update_document_status(
                document_id,
                IngestionStatus.ENRICHING,
            )
            if status_result.is_err():
                return Err(status_result.error)

            try:
                chunks_result = self._metadata_store.list_chunks(document_id)
                if chunks_result.is_err():
                    return self._fail_to_done(document_id, chunks_result.error)

                chunks = chunks_result.unwrap()
                concepts, mentions = self._build_heading_concepts(chunks, context.language)
                sections = self._group_sections(chunks)

                lesson_cards: list[StoredLessonCard] = []
                quiz_items: list[StoredQuizItem] = []

                for section in sections:
                    cards_result = self._generate_cards(section)
                    if cards_result.is_err():
                        return self._fail_to_done(document_id, cards_result.error)
                    lesson_cards.extend(cards_result.unwrap())

                    quiz_result = self._generate_quiz(section)
                    if quiz_result.is_err():
                        return self._fail_to_done(document_id, quiz_result.error)
                    quiz_items.extend(quiz_result.unwrap())

                event_type: OutboxEventType | None = None
                outbox_payload: dict | None = None
                if concepts or mentions:
                    event_type = OutboxEventType.CONCEPT_GRAPH_PROJECT
                    outbox_payload = {
                        "document_id": document_id,
                        "concepts": [
                            {
                                "concept_id": concept.concept_id,
                                "name": concept.name,
                                "canonical_name": concept.canonical_name,
                                "slug": concept.slug,
                                "category": concept.category,
                                "language": concept.language,
                                "domain": concept.domain,
                            }
                            for concept in concepts
                        ],
                        "mentions": [
                            {
                                "chunk_id": mention.chunk_id,
                                "concept_id": mention.concept_id,
                                "confidence": mention.confidence,
                                "source": mention.source,
                            }
                            for mention in mentions
                        ],
                    }

                persist_result = self._metadata_store.persist_enrichment_batch(
                    document_id=document_id,
                    lesson_cards=lesson_cards,
                    quiz_items=quiz_items,
                    concepts=concepts,
                    chunk_concepts=mentions,
                    outbox_event_type=event_type,
                    outbox_payload=outbox_payload,
                )
                if persist_result.is_err():
                    return self._fail_to_done(document_id, persist_result.error)

                appended_event_id = persist_result.unwrap()
                if appended_event_id is None:
                    enriched_result = self._metadata_store.update_document_status(
                        document_id,
                        IngestionStatus.ENRICHED,
                    )
                    if enriched_result.is_err():
                        return self._fail_to_done(document_id, enriched_result.error)

                span.set_attribute("concepts.count", len(concepts))
                span.set_attribute("mentions.count", len(mentions))
                span.set_attribute("cards.count", len(lesson_cards))
                span.set_attribute("quiz.count", len(quiz_items))
                logger.info(
                    "run_enrichment.completed",
                    document_id=document_id,
                    concepts=len(concepts),
                    mentions=len(mentions),
                    cards=len(lesson_cards),
                    quiz_items=len(quiz_items),
                )
                return Ok(
                    RunEnrichmentResponse(
                        document_id=document_id,
                        concept_count=len(concepts),
                        mention_count=len(mentions),
                        card_count=len(lesson_cards),
                        quiz_count=len(quiz_items),
                    )
                )
            except Exception as exc:
                return self._fail_to_done(document_id, exc)

    def _build_heading_concepts(
        self,
        chunks: list[StoredChunkMetadata],
        default_language: str | None,
    ) -> tuple[list[StoredConcept], list[StoredChunkConcept]]:
        concepts_by_id: dict[str, StoredConcept] = {}
        mention_keys: set[tuple[str, str]] = set()
        mentions: list[StoredChunkConcept] = []

        for chunk in chunks:
            for raw_heading in chunk.heading_path:
                name = self._clean_heading(raw_heading)
                if not name:
                    continue

                canonical_name = self._canonicalize_heading(name)
                slug = self._slugify(canonical_name)
                if not slug:
                    continue

                concepts_by_id.setdefault(
                    slug,
                    StoredConcept(
                        concept_id=slug,
                        name=name,
                        canonical_name=canonical_name,
                        slug=slug,
                        category="other",
                        language=chunk.language or default_language,
                    ),
                )

                mention_key = (chunk.chunk_id, slug)
                if mention_key in mention_keys:
                    continue
                mention_keys.add(mention_key)
                mentions.append(
                    StoredChunkConcept(
                        chunk_id=chunk.chunk_id,
                        concept_id=slug,
                        confidence=1.0,
                        source="heading",
                    )
                )

        return list(concepts_by_id.values()), mentions

    def _group_sections(self, chunks: list[StoredChunkMetadata]) -> list[_SectionContext]:
        sections: list[_SectionContext] = []
        current_chunks: list[StoredChunkMetadata] = []
        current_heading_path: tuple[str, ...] | None = None

        for chunk in chunks:
            effective_heading = self._effective_heading_path(chunk)
            if current_chunks and effective_heading != current_heading_path:
                section = self._build_section(current_heading_path or (), current_chunks)
                if section is not None:
                    sections.append(section)
                current_chunks = []

            current_chunks.append(chunk)
            current_heading_path = effective_heading

        if current_chunks:
            section = self._build_section(current_heading_path or (), current_chunks)
            if section is not None:
                sections.append(section)

        return sections

    def _build_section(
        self,
        heading_path: tuple[str, ...],
        chunks: list[StoredChunkMetadata],
    ) -> _SectionContext | None:
        content_parts = [
            content
            for content in (self._chunk_text(chunk) for chunk in chunks)
            if content
        ]
        if not content_parts:
            logger.warning(
                "run_enrichment.section_without_content",
                primary_chunk_id=chunks[0].chunk_id,
                heading_path=list(heading_path),
            )
            return None

        content_text = "\n\n".join(content_parts).strip()
        if len(content_text) > self._max_section_chars:
            content_text = content_text[: self._max_section_chars].rstrip()

        language = next((chunk.language for chunk in chunks if chunk.language), None)
        title = heading_path[-1] if heading_path else f"Section {chunks[0].chunk_index + 1}"
        return _SectionContext(
            document_id=chunks[0].document_id,
            heading_path=heading_path,
            title=title,
            primary_chunk_id=chunks[0].chunk_id,
            source_chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
            language=language,
            content_text=content_text,
        )

    def _generate_cards(
        self,
        section: _SectionContext,
    ) -> Result[list[StoredLessonCard], Exception]:
        prompt = self._build_cards_prompt(section)
        payload_result = generate_structured_payload(
            llm_client=self._llm_client,
            prompt=prompt,
            system=_CARDS_SYSTEM_PROMPT,
            schema_model=_GeneratedCardsPayload,
            repair_schema_name="cards",
        )
        if payload_result.is_err():
            return Err(payload_result.error)

        payload = payload_result.unwrap()
        return Ok(
            [
                StoredLessonCard(
                    card_id=str(uuid.uuid4()),
                    document_id=section.document_id,
                    primary_chunk_id=section.primary_chunk_id,
                    source_chunk_ids=section.source_chunk_ids,
                    heading_path=section.heading_path,
                    title=card.title,
                    bullets=tuple(card.bullets),
                    key_insight=card.key_insight,
                    card_index=index,
                    model_id=self._llm_client.model_id,
                )
                for index, card in enumerate(payload.cards)
            ]
        )

    def _generate_quiz(
        self,
        section: _SectionContext,
    ) -> Result[list[StoredQuizItem], Exception]:
        prompt = self._build_quiz_prompt(section)
        payload_result = generate_structured_payload(
            llm_client=self._llm_client,
            prompt=prompt,
            system=_QUIZ_SYSTEM_PROMPT,
            schema_model=_GeneratedQuizPayload,
            repair_schema_name="quiz",
        )
        if payload_result.is_err():
            return Err(payload_result.error)

        payload = payload_result.unwrap()
        return Ok(
            [
                StoredQuizItem(
                    question_id=str(uuid.uuid4()),
                    document_id=section.document_id,
                    primary_chunk_id=section.primary_chunk_id,
                    source_chunk_ids=section.source_chunk_ids,
                    heading_path=section.heading_path,
                    question=question.question,
                    choices=tuple(question.choices),
                    correct_index=question.correct_index,
                    explanation=question.explanation,
                    difficulty=question.difficulty,
                    question_index=index,
                    model_id=self._llm_client.model_id,
                )
                for index, question in enumerate(payload.questions)
            ]
        )

    def _build_cards_prompt(self, section: _SectionContext) -> str:
        heading_path = " > ".join(section.heading_path)
        language = section.language or "same as source"
        return (
            "Create 2-3 micro-learning lesson cards from the section below.\n"
            "Use the same language as the source content.\n"
            "Return strict JSON only with this shape:\n"
            '{"cards":[{"title":"...","bullets":["..."],"key_insight":"..."}]}\n\n'
            f"Heading path: {heading_path}\n"
            f"Section title: {section.title}\n"
            f"Language: {language}\n\n"
            "Section content:\n"
            f"{section.content_text}"
        )

    def _build_quiz_prompt(self, section: _SectionContext) -> str:
        heading_path = " > ".join(section.heading_path)
        language = section.language or "same as source"
        return (
            "Create 3-5 multiple-choice questions from the section below.\n"
            "Each question must have exactly 4 choices, one correct answer index, and an explanation.\n"
            "Use the same language as the source content.\n"
            "Return strict JSON only with this shape:\n"
            '{"questions":[{"question":"...","choices":["...","...","...","..."],'
            '"correct_index":0,"explanation":"...","difficulty":"easy|medium|hard"}]}\n\n'
            f"Heading path: {heading_path}\n"
            f"Section title: {section.title}\n"
            f"Language: {language}\n\n"
            "Section content:\n"
            f"{section.content_text}"
        )

    def _effective_heading_path(self, chunk: StoredChunkMetadata) -> tuple[str, ...]:
        if chunk.heading_path:
            return chunk.heading_path
        return (f"Section {chunk.chunk_index + 1}",)

    def _chunk_text(self, chunk: StoredChunkMetadata) -> str:
        if chunk.content_text and chunk.content_text.strip():
            return chunk.content_text.strip()
        if chunk.embedding_input and chunk.embedding_input.strip():
            return chunk.embedding_input.strip()
        return ""

    def _fail_to_done(self, document_id: str, error: Exception) -> Result[RunEnrichmentResponse, Exception]:
        self._metadata_store.update_document_status(
            document_id=document_id,
            status=IngestionStatus.DONE,
            error_msg=str(error),
        )
        logger.error("run_enrichment.failed", document_id=document_id, error=str(error))
        return Err(error)

    def _clean_heading(self, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    def _canonicalize_heading(self, value: str) -> str:
        return self._clean_heading(value).lower()

    def _slugify(self, value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value)
        ascii_text = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        ascii_text = ascii_text.replace("đ", "d").replace("Đ", "D")
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text.lower()).strip("-")
        return re.sub(r"-{2,}", "-", slug)
