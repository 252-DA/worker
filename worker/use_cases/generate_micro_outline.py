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

import re
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

from worker.prompts import micro_content as prompts
from worker.use_cases.shared.chunk_text import format_page_chunks_text, resolve_course_title
from worker.use_cases.shared.llm_role import LlmRole

logger = get_logger(__name__)

_USAGE_OK = "OK"
_USAGE_ERROR = "ERROR"
MAX_OUTLINE_SECTIONS = 20
_LOW_TEXT_CHAR_THRESHOLD = 200
_MIN_SECTION_CHAR_THRESHOLD = 550
_MAX_MERGED_SECTION_CHARS = 2600
_MAX_MERGED_SECTION_PAGES = 15
_PAGE_SNIPPET_LIMIT = 450

_COVER_KEYWORDS_RE = re.compile(
    r"\b(instructor|lecturer|professor|giảng\s*viên|phòng\s*thí\s*nghiệm|"
    r"laboratory|lab|department|faculty|university|trường|khoa|agenda|outline|mục\s*lục)\b",
    re.IGNORECASE,
)
_CLOSING_KEYWORDS_RE = re.compile(
    r"\b(thank\s*you|thanks|q\s*&\s*a|questions\??|cảm\s*ơn|hỏi\s*đáp|"
    r"references|tài\s*liệu\s*tham\s*khảo)\b",
    re.IGNORECASE,
)


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
    # Không chặn trần số mục ở đây: bộ slide dài (vd. 101 trang) model hay trả
    # > MAX_OUTLINE_SECTIONS mục, chặn bằng schema thì cả yêu cầu FAILED. Trần
    # của contract được giữ ở validate_and_normalize_sections bằng cách gộp mục.
    sections: list[_OutlineSectionItem] = Field(min_length=1)


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
    chunks_by_page: dict[int, list[StoredChunkMetadata]] = {}
    if not raw_pages:
        for idx, c in enumerate(chunks, start=1):
            pno = int(c.page_number) if c.page_number and c.page_number >= 1 else idx
            chunks_by_page.setdefault(pno, []).append(c)

    max_page = max(
        explicit_total_pages or 0,
        max(by_page.keys(), default=0),
        max(chunks_by_page.keys(), default=0),
        1,
    )
    summaries: list[dict[str, Any]] = []
    for pno in range(1, max_page + 1):
        if raw_pages:
            entries = by_page.get(pno, [])
            headings = [e["heading"] for e in entries if e["heading"]]
            texts = [e["text"] for e in entries if e["text"]]
            combined = "\n".join(texts).strip()
        else:
            p_chunks = chunks_by_page.get(pno, [])
            headings = [
                " > ".join(h for h in c.heading_path if h)
                for c in p_chunks
                if c.heading_path
            ]
            combined = format_page_chunks_text(p_chunks)

        snippet = combined[:_PAGE_SNIPPET_LIMIT]
        if len(combined) > _PAGE_SNIPPET_LIMIT:
            snippet += "…"
        summaries.append(
            {
                "page_number": pno,
                "heading": headings[0] if headings else "",
                "text_snippet": snippet,
                "full_text": combined,
                "char_count": len(combined),
                "is_boilerplate": _is_boilerplate_page(pno, combined, max_page),
            }
        )
    return summaries, max_page


def _is_boilerplate_page(pno: int, text: str, total_pages: int) -> bool:
    """Nhận diện trang bìa đầu tài liệu hoặc trang cảm ơn/hỏi đáp cuối tài liệu."""
    if total_pages < 4:
        return False
    cleaned = (text or "").strip()
    if pno == 1:
        if len(cleaned) < 110:
            return True
        if len(cleaned) < 280 and bool(_COVER_KEYWORDS_RE.search(cleaned)):
            return True
    if pno == total_pages and len(cleaned) < 180 and bool(_CLOSING_KEYWORDS_RE.search(cleaned)):
        return True
    return False


def _section_char_count(page_from: int, page_to: int, page_char_counts: dict[int, int]) -> int:
    return sum(page_char_counts.get(p, 0) for p in range(page_from, page_to + 1))


def _merge_two_sections(
    left: OutlineSection,
    right: OutlineSection,
    page_char_counts: dict[int, int],
) -> OutlineSection:
    left_chars = _section_char_count(left.page_from, left.page_to, page_char_counts)
    right_chars = _section_char_count(right.page_from, right.page_to, page_char_counts)
    dominant, secondary = (left, right) if left_chars >= right_chars else (right, left)
    p_from = min(left.page_from, right.page_from)
    p_to = max(left.page_to, right.page_to)
    combined_chars = _section_char_count(p_from, p_to, page_char_counts)
    lo_id = dominant.suggested_lo_id or secondary.suggested_lo_id
    lo_rationale = dominant.lo_rationale if dominant.suggested_lo_id else secondary.lo_rationale
    title = dominant.title
    summary = dominant.summary or secondary.summary or title
    return OutlineSection(
        title=title[:120],
        summary=summary[:240],
        page_from=p_from,
        page_to=p_to,
        suggested_lo_id=lo_id,
        low_text=combined_chars < _LOW_TEXT_CHAR_THRESHOLD,
        lo_rationale=lo_rationale,
    )


def _merge_sparse_sections(
    sections: list[OutlineSection],
    *,
    total_pages: int,
    page_char_counts: dict[int, int],
) -> list[OutlineSection]:
    """Gộp các mục quá ít chữ liền kề nhau để tránh chia vụn nhiều đoạn dưới ~450–550 ký tự."""
    total_chars = sum(page_char_counts.values())
    if len(sections) <= 1 or total_chars < 600:
        return sections
    min_chars = _MIN_SECTION_CHAR_THRESHOLD if (total_pages >= 6 or total_chars >= 1500) else 250
    merged = list(sections)

    changed = True
    while changed and len(merged) > 1:
        changed = False
        # Ưu tiên gộp mục nhỏ nhất trước
        order_idx = sorted(
            range(len(merged)),
            key=lambda i: _section_char_count(merged[i].page_from, merged[i].page_to, page_char_counts),
        )
        for idx in order_idx:
            cur = merged[idx]
            cur_chars = _section_char_count(cur.page_from, cur.page_to, page_char_counts)
            if cur_chars >= min_chars:
                break
            candidates: list[tuple[int, int, int]] = []
            for nb_idx in (idx - 1, idx + 1):
                if 0 <= nb_idx < len(merged):
                    nb = merged[nb_idx]
                    p_from = min(cur.page_from, nb.page_from)
                    p_to = max(cur.page_to, nb.page_to)
                    span_pages = p_to - p_from + 1
                    comb_chars = _section_char_count(p_from, p_to, page_char_counts)
                    if span_pages <= _MAX_MERGED_SECTION_PAGES and comb_chars <= _MAX_MERGED_SECTION_CHARS:
                        same_lo = 0 if (cur.suggested_lo_id and cur.suggested_lo_id == nb.suggested_lo_id) else 1
                        candidates.append((same_lo, comb_chars, nb_idx))
            if not candidates:
                continue
            candidates.sort()
            _, _, best_nb = candidates[0]
            left_idx, right_idx = min(idx, best_nb), max(idx, best_nb)
            merged[left_idx : right_idx + 1] = [
                _merge_two_sections(merged[left_idx], merged[right_idx], page_char_counts)
            ]
            changed = True
            break
    return merged


def _cap_section_count(
    sections: list[OutlineSection],
    *,
    page_char_counts: dict[int, int],
    max_sections: int = MAX_OUTLINE_SECTIONS,
) -> list[OutlineSection]:
    """
    Ép số mục về <= max_sections (trần của contract content-generation-result và
    core-api). Mỗi lượt gộp cặp mục liền kề có tổng chữ nhỏ nhất (hòa thì cặp ít
    trang hơn) để các mục sau khi gộp vẫn cân đối; không bỏ trang nào.
    """
    merged = list(sections)
    while len(merged) > max_sections:
        best = min(
            range(len(merged) - 1),
            key=lambda i: (
                _section_char_count(merged[i].page_from, merged[i + 1].page_to, page_char_counts),
                merged[i + 1].page_to - merged[i].page_from,
            ),
        )
        merged[best : best + 2] = [_merge_two_sections(merged[best], merged[best + 1], page_char_counts)]
    return merged


def validate_and_normalize_sections(
    raw_sections: list[_OutlineSectionItem],
    *,
    total_pages: int,
    valid_lo_ids: set[str],
    page_char_counts: dict[int, int],
    page_texts: dict[int, str] | None = None,
) -> tuple[OutlineSection, ...]:
    """
    Kiểm tra dải trang hợp lệ [1, total_pages], không lộn ngược, không chồng lấn,
    bỏ trang bìa/kết thúc thuần túy, gộp các mục quá thưa chữ và chỉ giữ suggested_lo_id hợp lệ.
    Model trả nhiều hơn MAX_OUTLINE_SECTIONS mục thì gộp bớt (không cắt đuôi: cắt
    đuôi làm mất các trang cuối của bộ slide).
    """
    ordered = sorted(raw_sections, key=lambda s: (s.page_from, s.page_to))
    normalized: list[OutlineSection] = []
    last_end = 0
    texts = page_texts or {}

    for item in ordered:
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

        if texts and p_from == 1 and _is_boilerplate_page(1, texts.get(1, ""), total_pages):
            if p_to == 1 and len(ordered) > 1:
                last_end = max(last_end, 1)
                continue
            if p_to > 1:
                p_from = 2

        if texts and p_to == total_pages and _is_boilerplate_page(total_pages, texts.get(total_pages, ""), total_pages):
            if p_from == total_pages and len(normalized) >= 1:
                last_end = p_to
                continue
            if p_to > p_from:
                p_to = total_pages - 1

        last_end = max(last_end, p_to)

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
    merged = _merge_sparse_sections(
        normalized,
        total_pages=total_pages,
        page_char_counts=page_char_counts,
    )
    return tuple(_cap_section_count(merged, page_char_counts=page_char_counts))


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
        course_title = resolve_course_title(self._metadata_store, request.course_id)
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
        page_texts = {p["page_number"]: p.get("full_text", "") for p in page_summaries}
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
            max_sections=MAX_OUTLINE_SECTIONS,
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
                page_texts=page_texts,
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
