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

from pydantic import BaseModel, Field, field_validator

from document_chunk.application.services.structured_output import generate_structured_payload
from document_chunk.domain.exceptions import ProcessingError
from document_chunk.domain.ports.llm_client import ILLMClient, LLMUsage
from document_chunk.domain.ports.metadata_store import IMetadataStore, StoredChunkMetadata
from document_chunk.shared.logger import get_logger
from document_chunk.shared.result import Err, Ok, Result

from worker.llm_role import LlmRole
from worker.prompts import micro_content as prompts
from worker.prompts._common import lo_line
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
MAX_BULLETS_PER_SCENE = 4
DEFAULT_VOICE = "vi-VN-HoaiMyNeural"

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


class _GeneratedScene(BaseModel):
    title: str
    slide_page: int = Field(ge=1)
    bullets: list[str] = Field(default_factory=list, max_length=6)
    narration: str

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
    scenes: list[_GeneratedScene] = Field(min_length=MIN_SCENES, max_length=MAX_SCENES)


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
) -> str | None:
    words = len(scene.narration.split())
    if words < MIN_NARRATION_WORDS:
        return (
            f"Lời đọc cảnh quá ngắn ({words} từ < {MIN_NARRATION_WORDS} từ); "
            "cần mở rộng diễn giải sư phạm ngắn gọn bám slide."
        )
    if words > MAX_NARRATION_WORDS:
        return (
            f"Lời đọc cảnh quá dài ({words} từ > {MAX_NARRATION_WORDS} từ); "
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
    in_range = [
        c for c in chunks
        if c.page_number is not None and page_from <= c.page_number <= page_to
    ]
    if not in_range:
        # Slide có thể chưa đánh số trang hoặc là slide ảnh; dùng chunk toàn tài liệu làm ngữ cảnh phụ nếu có.
        in_range = [c for c in chunks if c.page_number is None] or chunks[:8]

    by_page: dict[int, list[StoredChunkMetadata]] = {}
    all_chunk_ids: list[str] = []
    for c in in_range:
        pno = int(c.page_number) if c.page_number and page_from <= c.page_number <= page_to else page_from
        by_page.setdefault(pno, []).append(c)
        all_chunk_ids.append(str(c.chunk_id))

    pages: list[dict[str, Any]] = []
    for pno in range(page_from, max(page_from, page_to) + 1):
        p_chunks = by_page.get(pno, [])
        content = "\n".join((c.content_text or "").strip() for c in p_chunks if c.content_text)
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
        else:
            return Err(ProcessingError("video_script request requires pages or document_id"))

        lo_text = (
            lo_line(request.lo_code, request.lo_statement_vi, request.lo_statement_en)
            if request.lo_code and request.lo_statement_vi
            else None
        )
        topic_title = request.topic_title.strip() or f"Nội dung trang {page_from}–{page_to}"

        gen_prompt = prompts.video_script_generation(
            course_title=request.course_id,
            topic_title=topic_title,
            topic_summary=request.topic_summary,
            lo_line_text=lo_text,
            page_from=page_from,
            page_to=page_to,
            pages=pages,
            neighbor_topics=request.neighbor_topics,
            low_text=request.low_text,
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
                idx: _scene_code_problem(sc, page_from=page_from, page_to=page_to)
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
                    idx: JudgeVerdict(verdict="contradicted", reason=prob)
                    for idx, prob in code_problems.items()
                    if prob is not None
                }
                return failing_code if failing_code else None

            merged = collect_verdicts(
                (v.scene_index, v.verdict, v.reason) for v in llm_verdicts
            )
            for idx, prob in code_problems.items():
                if prob is not None:
                    merged[idx] = JudgeVerdict(verdict="contradicted", reason=prob)
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
            ordered_entries[rej.item.index] = _scene_to_dict(
                scene=sc,
                slide_page=clamped_page,
                status=FLAGGED,
                verdict="not_enough_evidence",
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
    ) -> _GeneratedScene | None:
        prompt = prompts.video_script_regeneration(
            topic_title=topic_title,
            page_from=page_from,
            page_to=page_to,
            pages=pages,
            scene=scene,
            reason=reason,
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
    return {
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
