"""
GenerateVideoScriptUseCase — sinh kịch bản video bám slide nguồn (B3).

Đặc điểm:
- Lấy nguồn chỉ trong [page_from, page_to] của bộ slide gốc.
- Số cảnh 4–8, tối đa 4 bullet/cảnh, lời giảng tiếng Việt giữ thuật ngữ (§8.1).
- Kiểm tra bằng code: lời đọc mỗi cảnh 40–110 từ, slide_page nằm trong dải trang,
  không chứa tham chiếu sang "bài trước"/"phần sau".
- Kiểm chứng CoVe + Reflexion theo chính sách (b) (§8.6): cấm số liệu/tên riêng/
  khẳng định mới ngoài slide. Cảnh hết lượt sửa vẫn không đạt được GIỮ NGUYÊN THỨ TỰ
  với ``check.status = 'FLAGGED'`` (§8.2) để giảng viên xem và xác nhận trước render.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from document_chunk.application.services.structured_output import generate_structured_payload
from document_chunk.domain.exceptions import ProcessingError
from document_chunk.domain.ports.llm_client import ILLMClient, LLMUsage
from document_chunk.domain.ports.metadata_store import IMetadataStore, StoredChunkMetadata
from document_chunk.shared.logger import get_logger
from document_chunk.shared.result import Err, Ok, Result

from worker.llm_role import LlmRole
from worker.prompts import micro_content as prompts
from worker.prompts._common import lo_line
from worker.use_cases._chunk_text import (
    filter_chunks_in_page_range,
    format_page_chunks_text,
    resolve_course_title,
)
from worker.use_cases.generated_content import RejectedDraft
from worker.use_cases.verification import (
    FLAGGED,
    JudgeVerdict,
    collect_verdicts,
    verify_and_refine,
)

logger = get_logger(__name__)

_USAGE_OK = "OK"
_USAGE_ERROR = "ERROR"
MIN_SCENES = 4
MAX_SCENES = 8
MIN_NARRATION_WORDS = 40
MAX_NARRATION_WORDS = 110
_HARD_MAX_NARRATION_WORDS = 135
_CODE_VIOLATION_VERDICT = "style_violation"
MAX_BULLETS_PER_SCENE = 4
DEFAULT_VOICE = "vi-VN-HoaiMyNeural"


@dataclass(frozen=True)
class _ScriptBounds:
    min_scenes: int
    max_scenes: int
    min_words: int
    max_words: int


def _compute_script_bounds(pages: list[dict[str, Any]], *, low_text: bool) -> _ScriptBounds:
    """Co giãn số cảnh và độ dài lời đọc theo mật độ chữ thực tế trên slide."""
    source_words = sum(len((p.get("content") or "").split()) for p in pages)
    if source_words < 90 or (low_text and source_words < 140):
        return _ScriptBounds(
            min_scenes=2 if (len(pages) >= 2 or source_words >= 45) else 1,
            max_scenes=4,
            min_words=25,
            max_words=MAX_NARRATION_WORDS,
        )
    if source_words < 200:
        return _ScriptBounds(
            min_scenes=2,
            max_scenes=6,
            min_words=30,
            max_words=MAX_NARRATION_WORDS,
        )
    return _ScriptBounds(
        min_scenes=3,
        max_scenes=MAX_SCENES,
        min_words=35,
        max_words=MAX_NARRATION_WORDS,
    )

_CROSS_REF_PATTERN = re.compile(
    r"\b(bài trước|video trước|phần trước|chương trước|bài sau|video sau|chương sau|video tiếp theo)\b",
    re.IGNORECASE,
)


class _UsageAccumulator:
    def __init__(self) -> None:
        self.total = LLMUsage()

    def __call__(self, usage: LLMUsage) -> None:
        self.total = self.total + usage


def _strip_markdown(text: str) -> str:
    cleaned = re.sub(r"https?://\S+", "", text or "")
    cleaned = re.sub(r"[*_`#>]+", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


class _GeneratedVisual(BaseModel):
    type: Literal["auto", "source_slide", "bullet_reveal", "manim"] = "auto"
    motion: Literal["none", "zoom_in"] = "zoom_in"
    diagram: Literal["flow", "array", "stack"] | None = None
    items: list[str] | None = Field(default=None, max_length=6)

    @field_validator("items")
    @classmethod
    def _clean_items(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        return [_strip_markdown(value)[:80] for value in values if value and _strip_markdown(value)]

    @model_validator(mode="after")
    def _validate_manim(self) -> "_GeneratedVisual":
        if self.type == "manim":
            if self.diagram is None:
                raise ValueError("manim visual requires a diagram")
            if self.items is None or not 2 <= len(self.items) <= 6:
                raise ValueError("manim visual requires 2 to 6 items")
        return self


class _GeneratedScene(BaseModel):
    title: str
    slide_page: int = Field(ge=1)
    bullets: list[str] = Field(default_factory=list, max_length=6)
    narration: str
    visual: _GeneratedVisual | None = None

    @field_validator("title")
    @classmethod
    def _clean_title(cls, value: str) -> str:
        t = _strip_markdown(value)
        if not t:
            raise ValueError("scene title must not be empty")
        return t[:120]

    @field_validator("narration")
    @classmethod
    def _clean_narration(cls, value: str) -> str:
        n = _strip_markdown(value)
        if not n:
            raise ValueError("scene narration must not be empty")
        return n

    @field_validator("bullets")
    @classmethod
    def _clean_bullets(cls, values: list[str]) -> list[str]:
        out = [_strip_markdown(v) for v in values if v and _strip_markdown(v)]
        return out[:MAX_BULLETS_PER_SCENE]


class _ScriptPayload(BaseModel):
    scenes: list[_GeneratedScene] = Field(min_length=1, max_length=MAX_SCENES)


class _SceneVerdict(BaseModel):
    scene_index: int
    verdict: Literal["supported", "contradicted", "not_enough_evidence"]
    reason: str = ""


class _ScriptVerificationPayload(BaseModel):
    verdicts: list[_SceneVerdict] = Field(min_length=1)


@dataclass
class _IndexedScene:
    index: int
    scene: _GeneratedScene


@dataclass(frozen=True)
class GenerateVideoScriptRequest:
    course_id: str
    micro_content_id: str
    document_id: str | None = None
    page_from: int = 1
    page_to: int = 1
    topic_title: str = ""
    topic_summary: str | None = None
    lo_id: str | None = None
    lo_code: str | None = None
    lo_statement_vi: str | None = None
    lo_statement_en: str | None = None
    voice: str = DEFAULT_VOICE
    neighbor_topics: tuple[str, ...] = ()
    low_text: bool = False
    instructor_prompt: str | None = None
    pages: tuple[dict[str, Any], ...] | None = None


@dataclass(frozen=True)
class GenerateVideoScriptResponse:
    course_id: str
    micro_content_id: str
    script: dict[str, Any]
    rejected: tuple[RejectedDraft, ...] = ()

    @property
    def scene_count(self) -> int:
        return len(self.script.get("scenes") or [])


def _scene_code_problem(
    scene: _GeneratedScene,
    *,
    page_from: int,
    page_to: int,
    min_words: int = MIN_NARRATION_WORDS,
    hard_max_words: int = _HARD_MAX_NARRATION_WORDS,
) -> str | None:
    words = len(scene.narration.split())
    if words < min_words:
        return (
            f"Lời đọc cảnh quá ngắn ({words} từ < {min_words} từ); "
            "cần mở rộng diễn giải sư phạm ngắn gọn bám slide."
        )
    if words > hard_max_words:
        return (
            f"Lời đọc cảnh quá dài ({words} từ > {hard_max_words} từ); "
            "cần rút gọn cho súc tích."
        )
    match = _CROSS_REF_PATTERN.search(scene.narration)
    if match:
        return (
            f"Lời đọc chứa cụm tham chiếu ngoài bài ('{match.group(1)}'); "
            "phải giải thích độc lập không nhắc bài trước/sau."
        )
    if not (page_from <= scene.slide_page <= page_to):
        return (
            f"slide_page={scene.slide_page} nằm ngoài dải nguồn [{page_from}, {page_to}]."
        )
    return None


def _pages_from_chunks(
    chunks: list[StoredChunkMetadata],
    *,
    page_from: int,
    page_to: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    in_range = filter_chunks_in_page_range(chunks, page_from, page_to)

    by_page: dict[int, list[StoredChunkMetadata]] = {}
    all_chunk_ids: list[str] = []
    for c in in_range:
        pno = int(c.page_number) if c.page_number and page_from <= c.page_number <= page_to else page_from
        by_page.setdefault(pno, []).append(c)
        all_chunk_ids.append(str(c.chunk_id))

    pages: list[dict[str, Any]] = []
    for pno in range(page_from, max(page_from, page_to) + 1):
        p_chunks = by_page.get(pno, [])
        content = format_page_chunks_text(p_chunks)
        pages.append(
            {
                "page_number": pno,
                "content": content,
                "chunk_ids": [str(c.chunk_id) for c in p_chunks],
            }
        )
    return pages, all_chunk_ids


class GenerateVideoScriptUseCase:
    def __init__(
        self,
        metadata_store: IMetadataStore,
        llm_client: ILLMClient,
        llm_provider: str = "unknown",
        llm_profile: str | None = None,
        verifier: LlmRole | None = None,
        max_verification_retries: int = 2,
    ) -> None:
        self._metadata_store = metadata_store
        self._writer = LlmRole(client=llm_client, provider=llm_provider, profile=llm_profile)
        self._verifier = verifier or self._writer
        self._max_verification_retries = max_verification_retries

    def execute(
        self, request: GenerateVideoScriptRequest
    ) -> Result[GenerateVideoScriptResponse, Exception]:
        trace_id = str(uuid.uuid4())
        page_from = max(1, int(request.page_from or 1))
        page_to = max(page_from, int(request.page_to or page_from))

        pages: list[dict[str, Any]]
        source_chunk_ids: list[str]
        if request.pages:
            pages = list(request.pages)
            source_chunk_ids = [
                str(cid)
                for p in pages
                for cid in (p.get("chunk_ids") or [])
                if cid
            ]
        elif request.document_id:
            chunks_res = self._metadata_store.list_chunks(request.document_id)
            if chunks_res.is_err():
                return Err(chunks_res.error)
            pages, source_chunk_ids = _pages_from_chunks(
                chunks_res.unwrap(),
                page_from=page_from,
                page_to=page_to,
            )
            if not source_chunk_ids and not (request.instructor_prompt and request.instructor_prompt.strip()):
                return Err(
                    ProcessingError(
                        f"Dải trang {page_from}–{page_to} không có văn bản trích xuất được từ tài liệu nguồn."
                    )
                )
        else:
            return Err(ProcessingError("video_script request requires pages or document_id"))

        lo_text = (
            lo_line(request.lo_code, request.lo_statement_vi, request.lo_statement_en)
            if request.lo_code and request.lo_statement_vi
            else None
        )
        topic_title = request.topic_title.strip() or f"Nội dung trang {page_from}–{page_to}"
        course_title = resolve_course_title(self._metadata_store, request.course_id)
        bounds = _compute_script_bounds(pages, low_text=request.low_text)

        gen_prompt = prompts.video_script_generation(
            course_title=course_title,
            topic_title=topic_title,
            topic_summary=request.topic_summary,
            lo_line_text=lo_text,
            page_from=page_from,
            page_to=page_to,
            pages=pages,
            neighbor_topics=request.neighbor_topics,
            low_text=request.low_text,
            instructor_prompt=request.instructor_prompt,
            min_scenes=bounds.min_scenes,
            max_scenes=bounds.max_scenes,
            min_words=bounds.min_words,
            max_words=bounds.max_words,
        )
        started = time.monotonic()
        usage = _UsageAccumulator()
        gen_res = generate_structured_payload(
            llm_client=self._writer.client,
            prompt=gen_prompt.user,
            system=gen_prompt.system,
            schema_model=_ScriptPayload,
            repair_schema_name="VideoScriptPayload",
            on_usage=usage,
        )
        self._log_usage(
            role=self._writer,
            use_case=gen_prompt.name,
            status=_USAGE_OK if gen_res.is_ok() else _USAGE_ERROR,
            latency_ms=int((time.monotonic() - started) * 1000),
            trace_id=trace_id,
            course_id=request.course_id,
            usage=usage.total,
        )
        if gen_res.is_err():
            return Err(gen_res.error)

        indexed_scenes = [
            _IndexedScene(index=i, scene=sc)
            for i, sc in enumerate(gen_res.unwrap().scenes[:MAX_SCENES])
        ]

        def judge(current: list[_IndexedScene]) -> dict[int, JudgeVerdict] | None:
            raw_scenes = [entry.scene for entry in current]
            code_problems = {
                idx: _scene_code_problem(
                    sc,
                    page_from=page_from,
                    page_to=page_to,
                    min_words=bounds.min_words,
                )
                for idx, sc in enumerate(raw_scenes)
            }
            llm_verdicts = self._run_verifier(
                page_from=page_from,
                page_to=page_to,
                pages=pages,
                scenes=raw_scenes,
                trace_id=trace_id,
                course_id=request.course_id,
            )
            if llm_verdicts is None:
                # Judge LLM lỗi nhưng lỗi code (quá ngắn/quá dài/sai trang) vẫn cần sửa.
                failing_code = {
                    idx: JudgeVerdict(verdict=_CODE_VIOLATION_VERDICT, reason=prob)
                    for idx, prob in code_problems.items()
                    if prob is not None
                }
                return failing_code if failing_code else None

            merged = collect_verdicts(
                (v.scene_index, v.verdict, v.reason) for v in llm_verdicts
            )
            for idx, prob in code_problems.items():
                if prob is not None:
                    merged[idx] = JudgeVerdict(verdict=_CODE_VIOLATION_VERDICT, reason=prob)
            return merged

        def regenerate(entry: _IndexedScene, reason: str) -> _IndexedScene | None:
            fixed = self._run_regeneration(
                topic_title=topic_title,
                page_from=page_from,
                page_to=page_to,
                pages=pages,
                scene=entry.scene,
                reason=reason,
                trace_id=trace_id,
                course_id=request.course_id,
                min_words=bounds.min_words,
                max_words=bounds.max_words,
            )
            if fixed is None:
                return None
            return _IndexedScene(index=entry.index, scene=fixed)

        refined = verify_and_refine(
            indexed_scenes,
            judge=judge,
            regenerate=regenerate,
            max_retries=self._max_verification_retries,
            on_exhausted=lambda still_failing: logger.warning(
                "generate_video_script.verify.exhausted",
                trace_id=trace_id,
                micro_content_id=request.micro_content_id,
                still_failing=still_failing,
            ),
        )

        # §8.2: Giữ MỌI cảnh theo đúng thứ tự ban đầu, cảnh không đạt mang cờ FLAGGED.
        ordered_entries: dict[int, dict[str, Any]] = {}
        rejected_list: list[RejectedDraft] = []
        writer_model = self._writer.client.model_id
        judge_model = self._verifier.client.model_id

        for entry in refined.kept:
            sc = entry.item.scene
            clamped_page = max(page_from, min(page_to, sc.slide_page))
            ordered_entries[entry.item.index] = _scene_to_dict(
                scene=sc,
                slide_page=clamped_page,
                status=entry.verification.status,
                verdict=entry.verification.judge_verdict,
                reason=entry.verification.judge_reason,
                writer_model=writer_model,
                judge_model=judge_model,
                source_chunk_ids=source_chunk_ids,
            )

        for rej in refined.rejected:
            sc = rej.item.scene
            clamped_page = max(page_from, min(page_to, sc.slide_page))
            last_verdict = (
                rej.history[-1].verdict
                if rej.history
                else "not_enough_evidence"
            )
            ordered_entries[rej.item.index] = _scene_to_dict(
                scene=sc,
                slide_page=clamped_page,
                status=FLAGGED,
                verdict=last_verdict,
                reason=rej.reason,
                writer_model=writer_model,
                judge_model=judge_model,
                source_chunk_ids=source_chunk_ids,
            )
            rejected_list.append(
                RejectedDraft(
                    reason=rej.reason,
                    regeneration_attempts=rej.regeneration_attempts,
                )
            )

        ordered_scenes = [ordered_entries[i] for i in sorted(ordered_entries)]
        script_obj: dict[str, Any] = {
            "voice": request.voice or DEFAULT_VOICE,
            "source_document_id": request.document_id,
            "source_page_from": page_from,
            "source_page_to": page_to,
            "scenes": ordered_scenes,
        }
        return Ok(
            GenerateVideoScriptResponse(
                course_id=request.course_id,
                micro_content_id=request.micro_content_id,
                script=script_obj,
                rejected=tuple(rejected_list),
            )
        )

    def _run_verifier(
        self,
        *,
        page_from: int,
        page_to: int,
        pages: list[dict[str, Any]],
        scenes: list[_GeneratedScene],
        trace_id: str,
        course_id: str,
    ) -> list[_SceneVerdict] | None:
        prompt = prompts.video_script_verification(
            page_from=page_from,
            page_to=page_to,
            pages=pages,
            scenes=scenes,
        )
        started = time.monotonic()
        usage = _UsageAccumulator()
        res = generate_structured_payload(
            llm_client=self._verifier.client,
            prompt=prompt.user,
            system=prompt.system,
            schema_model=_ScriptVerificationPayload,
            repair_schema_name="VideoScriptVerification",
            on_usage=usage,
        )
        self._log_usage(
            role=self._verifier,
            use_case=prompt.name,
            status=_USAGE_OK if res.is_ok() else _USAGE_ERROR,
            latency_ms=int((time.monotonic() - started) * 1000),
            trace_id=trace_id,
            course_id=course_id,
            usage=usage.total,
        )
        if res.is_err():
            logger.warning(
                "generate_video_script.verify.failed",
                trace_id=trace_id,
                error=str(res.error),
            )
            return None
        return res.unwrap().verdicts

    def _run_regeneration(
        self,
        *,
        topic_title: str,
        page_from: int,
        page_to: int,
        pages: list[dict[str, Any]],
        scene: _GeneratedScene,
        reason: str,
        trace_id: str,
        course_id: str,
        min_words: int = MIN_NARRATION_WORDS,
        max_words: int = MAX_NARRATION_WORDS,
    ) -> _GeneratedScene | None:
        prompt = prompts.video_script_regeneration(
            topic_title=topic_title,
            page_from=page_from,
            page_to=page_to,
            pages=pages,
            scene=scene,
            reason=reason,
            min_words=min_words,
            max_words=max_words,
        )
        started = time.monotonic()
        usage = _UsageAccumulator()
        res = generate_structured_payload(
            llm_client=self._writer.client,
            prompt=prompt.user,
            system=prompt.system,
            schema_model=_GeneratedScene,
            repair_schema_name="VideoScriptScene",
            on_usage=usage,
        )
        self._log_usage(
            role=self._writer,
            use_case=prompt.name,
            status=_USAGE_OK if res.is_ok() else _USAGE_ERROR,
            latency_ms=int((time.monotonic() - started) * 1000),
            trace_id=trace_id,
            course_id=course_id,
            usage=usage.total,
        )
        if res.is_err():
            logger.warning(
                "generate_video_script.regenerate.failed",
                trace_id=trace_id,
                error=str(res.error),
            )
            return None
        return res.unwrap()

    def _log_usage(
        self,
        *,
        role: LlmRole,
        use_case: str,
        status: str,
        latency_ms: int,
        trace_id: str,
        course_id: str,
        usage: LLMUsage,
    ) -> None:
        res = self._metadata_store.record_llm_usage(
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
        if res.is_err():
            logger.warning(
                "generate_video_script.usage_log_failed",
                use_case=use_case,
                error=str(res.error),
            )


def _scene_to_dict(
    *,
    scene: _GeneratedScene,
    slide_page: int,
    status: str,
    verdict: str | None,
    reason: str | None,
    writer_model: str | None,
    judge_model: str | None,
    source_chunk_ids: list[str],
) -> dict[str, Any]:
    word_count = len(scene.narration.split())
    estimated_sec = max(12, round(word_count / 2.6))
    result = {
        "title": scene.title,
        "slide_page": slide_page,
        "bullets": list(scene.bullets),
        "narration": scene.narration,
        "duration_sec": estimated_sec,
        "check": {
            "status": status,
            "verdict": verdict,
            "reason": reason,
            "writer_model": writer_model,
            "judge_model": judge_model,
            "source_chunk_ids": list(source_chunk_ids[:20]),
        },
    }
    if scene.visual is not None:
        result["visual"] = scene.visual.model_dump(exclude_none=True)
    return result
