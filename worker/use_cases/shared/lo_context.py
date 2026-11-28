"""
Ngữ cảnh nguồn theo Learning Outcome, dùng chung cho sinh quiz và sinh card.

Hai use case cùng chọn LO từ một target (lo/chapter/assessment) và cùng lấy
đoạn nguồn cho LO đó. Luật "module theo chương chỉ được dùng slide của chương
đã chọn" phải giống hệt nhau ở cả hai, nên nó nằm ở đây một lần.
"""
from dataclasses import dataclass
from typing import Protocol

from document_chunk.domain.exceptions import ProcessingError
from document_chunk.domain.ports.metadata_store import (
    IMetadataStore,
    StoredCourse,
    StoredLearningOutcome,
)
from document_chunk.shared.logger import get_logger
from document_chunk.shared.result import Err, Ok, Result

logger = get_logger(__name__)


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


@dataclass(frozen=True)
class LoTarget:
    course: StoredCourse
    lo: StoredLearningOutcome


class LoContextLoader:
    def __init__(
        self,
        metadata_store: IMetadataStore,
        context_client: QuizContextClient | None = None,
        fallback_to_local_context: bool = True,
        max_context_chars: int = 6000,
        log_prefix: str = "lo_context",
    ) -> None:
        self._metadata_store = metadata_store
        self._context_client = context_client
        self._fallback_to_local_context = fallback_to_local_context
        self._max_context_chars = max_context_chars
        self._log_prefix = log_prefix

    def resolve_lo_ids(
        self,
        course_id: str,
        target_kind: str,
        target_code: str,
    ) -> Result[list[str], Exception]:
        kind = target_kind
        code = target_code

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

    def resolve_target(
        self,
        course_id: str,
        target_kind: str,
        target_code: str,
    ) -> Result[LoTarget, Exception]:
        """LO đầu tiên của target, kèm khoá học của nó."""
        lo_ids = self.resolve_lo_ids(course_id, target_kind, target_code)
        if lo_ids.is_err():
            return Err(lo_ids.error)
        if not lo_ids.unwrap():
            return Err(ProcessingError(
                f"No LOs found for {target_kind}={target_code} in course {course_id}"
            ))
        primary_lo_id = lo_ids.unwrap()[0]

        curriculum_result = self._metadata_store.get_curriculum(course_id)
        if curriculum_result.is_err():
            return Err(curriculum_result.error)
        curriculum = curriculum_result.unwrap()
        if curriculum is None:
            return Err(ProcessingError(f"No curriculum for course {course_id}"))

        course, _, learning_outcomes, _ = curriculum
        lo = next((lo for lo in learning_outcomes if lo.lo_id == primary_lo_id), None)
        if lo is None:
            return Err(ProcessingError(f"LO {primary_lo_id} not found in curriculum"))
        return Ok(LoTarget(course=course, lo=lo))

    def load(
        self,
        *,
        course_id: str,
        lo_id: str,
        lo_code: str,
        lo_statement: str,
        bloom: str | None,
        style: str,
        source_document_ids: list[str] | None,
    ) -> Result[tuple[list[str], list[str], str | None], Exception]:
        """
        Trả về (đoạn nguồn đã đánh số, chunk_id tương ứng, document_id đầu tiên).
        """
        # Module launches must use only the selected chapter's lecture sources.
        # Generic MCP retrieval has no document filter, so use local LO mappings
        # for this explicit scope and never fall back to another chapter.
        if self._context_client is not None and source_document_ids is None:
            try:
                context = self._context_client.retrieve_quiz_context(
                    course_id=course_id,
                    lo_code=lo_code,
                    query=lo_statement,
                    bloom_level=bloom,
                    assessment_style=style,
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
                        f"{self._log_prefix}.context.mcp",
                        lo_id=lo_id,
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
                    f"{self._log_prefix}.context.mcp_failed",
                    lo_id=lo_id,
                    error=str(exc),
                )

        chunks_result = self._metadata_store.list_chunks_for_lo(lo_id)
        if chunks_result.is_err():
            return Err(chunks_result.error)

        context_parts: list[str] = []
        total_chars = 0
        source_chunk_ids: list[str] = []
        local_document_id: str | None = None
        candidates = chunks_result.unwrap()
        if source_document_ids is not None:
            # psycopg returns UUID objects even though the metadata port types
            # IDs as strings. Normalize before comparing with JSON job IDs.
            allowed_documents = {str(value) for value in source_document_ids}
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


__all__ = ["LoContextLoader", "LoTarget", "QuizContextClient"]
