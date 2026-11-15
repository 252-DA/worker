"""
GenerateMicroOutlineUseCase — tách một bộ slide thành các nội dung micro-content và gợi ý LO chính.

B2 của docs/ke-hoach-video-micro-content.md:
- Truyền toàn bộ danh sách trang 1..N (kể cả trang chỉ có ảnh/không trích được chữ).
- LLM chọn dải trang liên tiếp, tóm tắt, cờ low_text, và gợi ý 1 LO lá theo ngữ nghĩa
  (LO gắn với chương của slide được đánh dấu ưu tiên, nhưng không giới hạn cứng).
- Mã kiểm chặt tính đơn điệu, biên 1..N, không chồng lấn và kiểm lo_id hợp lệ.
- Không đọc/ghi bảng micro_contents; kết quả đi qua CONTENT_GENERATION_RESULT về core-api.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, field_validator

from document_chunk.application.services.structured_output import generate_structured_payload
from document_chunk.domain.exceptions import ProcessingError
from document_chunk.domain.ports.llm_client import ILLMClient, LLMUsage
from document_chunk.domain.ports.metadata_store import (
    IMetadataStore,
    StoredChunkMetadata,
    StoredLearningOutcome,
)
from document_chunk.shared.logger import get_logger
from document_chunk.shared.result import Err, Ok, Result

from worker.llm_role import LlmRole
from worker.prompts import micro_content as prompts

logger = get_logger(__name__)

_USAGE_OK = "OK"
_USAGE_ERROR = "ERROR"
MAX_OUTLINE_SECTIONS = 20
_LOW_TEXT_CHAR_THRESHOLD = 200
_PAGE_SNIPPET_LIMIT = 450


class _UsageAccumulator:
    def __init__(self) -> None:
        self.total = LLMUsage()

    def __call__(self, usage: LLMUsage) -> None:
        self.total = self.total + usage


class _OutlineSectionItem(BaseModel):
    title: str
    summary: str = ""
    page_from: int = Field(ge=1)
    page_to: int = Field(ge=1)
    suggested_lo_id: str | None = None
    lo_rationale: str | None = None
    low_text: bool = False

    @field_validator("title")
    @classmethod
    def _clean_title(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("section title must not be empty")
        return text[:255]

    @field_validator("summary")
    @classmethod
    def _clean_summary(cls, value: str) -> str:
        return value.strip()


class _OutlinePayload(BaseModel):
    sections: list[_OutlineSectionItem] = Field(min_length=1, max_length=MAX_OUTLINE_SECTIONS)


@dataclass(frozen=True)
class OutlineSection:
    title: str
    summary: str
    page_from: int
    page_to: int
    suggested_lo_id: str | None
    low_text: bool
    lo_rationale: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "page_from": self.page_from,
            "page_to": self.page_to,
            "suggested_lo_id": self.suggested_lo_id,
            "low_text": self.low_text,
            "lo_rationale": self.lo_rationale,
        }


@dataclass(frozen=True)
class GenerateMicroOutlineRequest:
    course_id: str
    document_id: str
    document_title: str = ""
    chapter_code: str | None = None
    total_pages: int | None = None
    pages: list[dict[str, Any]] | None = None
    candidate_los: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class GenerateMicroOutlineResponse:
    course_id: str
    document_id: str
    sections: tuple[OutlineSection, ...]

    @property
    def section_count(self) -> int:
        return len(self.sections)


def _leaf_los(los: list[StoredLearningOutcome]) -> list[StoredLearningOutcome]:
    parent_codes = {lo.parent_code for lo in los if lo.parent_code}
    leaves = [lo for lo in los if lo.code not in parent_codes]
    return leaves or los


def _normalize_pages(
    *,
    raw_pages: list[dict[str, Any]] | None,
    chunks: list[StoredChunkMetadata],
    explicit_total_pages: int | None,
) -> tuple[list[dict[str, Any]], int]:
    """Dựng danh sách từng trang 1..N kể cả trang không có text."""
    by_page: dict[int, list[dict[str, Any]]] = {}
    if raw_pages:
        for idx, p in enumerate(raw_pages, start=1):
            pno = int(p.get("page_number") or idx)
            if pno < 1:
                continue
            by_page.setdefault(pno, []).append(
                {
                    "heading": str(p.get("heading") or "").strip(),
                    "text": str(p.get("text") or "").strip(),
                }
            )
    else:
        for idx, c in enumerate(chunks, start=1):
            pno = int(c.page_number) if c.page_number and c.page_number >= 1 else idx
            by_page.setdefault(pno, []).append(
                {
                    "heading": " > ".join(c.heading_path) if c.heading_path else "",
                    "text": (c.content_text or "").strip(),
                }
            )

    max_page = max(
        explicit_total_pages or 0,
        max(by_page.keys(), default=0),
        1,
    )
    summaries: list[dict[str, Any]] = []
    for pno in range(1, max_page + 1):
        entries = by_page.get(pno, [])
        headings = [e["heading"] for e in entries if e["heading"]]
        texts = [e["text"] for e in entries if e["text"]]
        combined = "\n".join(texts).strip()
        snippet = combined[:_PAGE_SNIPPET_LIMIT]
        if len(combined) > _PAGE_SNIPPET_LIMIT:
            snippet += "…"
        summaries.append(
            {
                "page_number": pno,
                "heading": headings[0] if headings else "",
                "text_snippet": snippet,
                "char_count": len(combined),
            }
        )
    return summaries, max_page


def validate_and_normalize_sections(
    raw_sections: list[_OutlineSectionItem],
    *,
    total_pages: int,
    valid_lo_ids: set[str],
    page_char_counts: dict[int, int],
) -> tuple[OutlineSection, ...]:
    """
    Kiểm tra dải trang hợp lệ [1, total_pages], không lộn ngược, không chồng lấn,
    và chỉ giữ suggested_lo_id nằm trong danh sách LO hợp lệ của học phần.
    """
    ordered = sorted(raw_sections, key=lambda s: (s.page_from, s.page_to))
    normalized: list[OutlineSection] = []
    last_end = 0

    for item in ordered[:MAX_OUTLINE_SECTIONS]:
        p_from = max(1, min(item.page_from, total_pages))
        p_to = max(1, min(item.page_to, total_pages))
        if p_to < p_from:
            raise ValueError(f"Invalid page range {item.page_from}..{item.page_to}")
        if p_from <= last_end:
            if p_to > last_end:
                # Lệch biên 1 trang ở giữa 2 nội dung: đẩy page_from lên liền sau mục trước.
                p_from = last_end + 1
            else:
                raise ValueError(
                    f"Overlapping outline section {item.page_from}..{item.page_to} with previous ending at {last_end}"
                )
        last_end = p_to

        lo_id = item.suggested_lo_id.strip() if isinstance(item.suggested_lo_id, str) else None
        if lo_id not in valid_lo_ids:
            lo_id = None
        rationale = item.lo_rationale.strip() if lo_id and isinstance(item.lo_rationale, str) and item.lo_rationale.strip() else None

        section_chars = sum(page_char_counts.get(p, 0) for p in range(p_from, p_to + 1))
        low_text = bool(item.low_text) or section_chars < _LOW_TEXT_CHAR_THRESHOLD

        normalized.append(
            OutlineSection(
                title=item.title,
                summary=item.summary or item.title,
                page_from=p_from,
                page_to=p_to,
                suggested_lo_id=lo_id,
                low_text=low_text,
                lo_rationale=rationale,
            )
        )

    if not normalized:
        raise ValueError("No valid outline sections produced")
    return tuple(normalized)


class GenerateMicroOutlineUseCase:
    def __init__(
        self,
        metadata_store: IMetadataStore,
        llm_client: ILLMClient,
        llm_provider: str = "unknown",
        llm_profile: str | None = None,
    ) -> None:
        self._metadata_store = metadata_store
        self._writer = LlmRole(client=llm_client, provider=llm_provider, profile=llm_profile)

    def execute(
        self, request: GenerateMicroOutlineRequest
    ) -> Result[GenerateMicroOutlineResponse, Exception]:
        trace_id = str(uuid.uuid4())
        course_title = request.course_id
        chapter_code = request.chapter_code
        doc_title = request.document_title or request.document_id

        chunks: list[StoredChunkMetadata] = []
        if not request.pages:
            chunks_res = self._metadata_store.list_chunks(request.document_id)
            if chunks_res.is_err():
                return Err(chunks_res.error)
            chunks = chunks_res.unwrap()

        page_summaries, total_pages = _normalize_pages(
            raw_pages=request.pages,
            chunks=chunks,
            explicit_total_pages=request.total_pages,
        )
        page_char_counts = {p["page_number"]: p["char_count"] for p in page_summaries}
        if not any(page_char_counts.values()) and not request.pages:
            return Err(
                ProcessingError(
                    f"Tài liệu {request.document_id} chưa có trang hoặc chunk nào được lập chỉ mục."
                )
            )

        candidate_los = list(request.candidate_los or [])
        if not candidate_los:
            cur_res = self._metadata_store.get_curriculum(request.course_id)
            if cur_res.is_ok() and cur_res.unwrap() is not None:
                course, chapters, learning_outcomes, _ = cur_res.unwrap()
                course_title = course.title_vi or course.code
                leaves = _leaf_los(learning_outcomes)
                chapter_lo_ids: set[str] = set()
                if chapter_code:
                    ch_res = self._metadata_store.list_los_by_chapter(
                        request.course_id, chapter_code
                    )
                    if ch_res.is_ok():
                        chapter_lo_ids = {lo.lo_id for lo in ch_res.unwrap()}
                candidate_los = [
                    {
                        "lo_id": lo.lo_id,
                        "code": lo.code,
                        "statement_vi": lo.statement_vi,
                        "statement_en": lo.statement_en,
                        "preferred": lo.lo_id in chapter_lo_ids,
                    }
                    for lo in leaves
                ]

        valid_lo_ids = {str(lo["lo_id"]) for lo in candidate_los if lo.get("lo_id")}
        prompt = prompts.outline_suggestion(
            course_title=course_title,
            document_title=doc_title,
            chapter_code=chapter_code,
            total_pages=total_pages,
            page_summaries=page_summaries,
            candidate_los=candidate_los,
        )

        started = time.monotonic()
        usage = _UsageAccumulator()
        payload_res = generate_structured_payload(
            llm_client=self._writer.client,
            prompt=prompt.user,
            system=prompt.system,
            schema_model=_OutlinePayload,
            repair_schema_name="MicroOutlinePayload",
            on_usage=usage,
        )
        self._log_usage(
            status=_USAGE_OK if payload_res.is_ok() else _USAGE_ERROR,
            latency_ms=int((time.monotonic() - started) * 1000),
            trace_id=trace_id,
            course_id=request.course_id,
            usage=usage.total,
        )
        if payload_res.is_err():
            return Err(payload_res.error)

        try:
            sections = validate_and_normalize_sections(
                payload_res.unwrap().sections,
                total_pages=total_pages,
                valid_lo_ids=valid_lo_ids,
                page_char_counts=page_char_counts,
            )
        except ValueError as exc:
            return Err(ProcessingError(f"Invalid micro-outline generated: {exc}"))

        logger.info(
            "generate_micro_outline.done",
            course_id=request.course_id,
            document_id=request.document_id,
            sections=len(sections),
        )
        return Ok(
            GenerateMicroOutlineResponse(
                course_id=request.course_id,
                document_id=request.document_id,
                sections=sections,
            )
        )

    def _log_usage(
        self,
        *,
        status: str,
        latency_ms: int,
        trace_id: str,
        course_id: str,
        usage: LLMUsage,
    ) -> None:
        res = self._metadata_store.record_llm_usage(
            provider=self._writer.provider,
            model=self._writer.client.model_id,
            use_case="micro_outline_suggestion",
            status=status,
            course_id=course_id,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=usage.cost_usd,
            latency_ms=latency_ms,
            trace_id=trace_id,
        )
        if res.is_err():
            logger.warning(
                "generate_micro_outline.usage_log_failed",
                error=str(res.error),
            )
