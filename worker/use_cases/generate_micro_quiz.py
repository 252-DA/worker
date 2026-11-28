"""
GenerateMicroQuizUseCase — sinh 2–4 câu hỏi củng cố cho một micro-content (B5).

Tái sử dụng hoàn toàn prompt sinh/chấm/sửa câu hỏi, bảng schema theo kiểu câu,
``answer_layout`` / ``answer_key_problem`` và vòng ``verify_and_refine`` của
``GenerateCurriculumQuizUseCase``, nhưng thu hẹp nguồn vào đúng dải trang
[source_page_from, source_page_to] (kèm kịch bản video nếu đã có).
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

from document_chunk.application.services.answer_layout import balanced_positions
from document_chunk.application.services.structured_output import generate_structured_payload
from document_chunk.domain.exceptions import ProcessingError
from document_chunk.domain.ports.llm_client import ILLMClient, LLMUsage
from document_chunk.domain.ports.metadata_store import IMetadataStore, StoredChunkMetadata
from document_chunk.shared.logger import get_logger
from document_chunk.shared.result import Err, Ok, Result

from worker.prompts import micro_content as micro_prompts
from worker.use_cases.shared.chunk_text import (
    filter_chunks_in_page_range,
    format_chunk_with_heading,
    resolve_course_title,
)
from worker.use_cases.shared.generated_content import GeneratedQuizItem, RejectedDraft
from worker.use_cases.shared.llm_role import LlmRole
from worker.use_cases.shared.quiz_schema import (
    QUESTION_MODELS,
    CurriculumQuizPayload,
    QuestionBase,
    QuizVerificationPayload,
    answer_key_problem,
    answer_layout,
)
from worker.use_cases.shared.verification import (
    JudgeVerdict,
    RefineResult,
    collect_verdicts,
    verify_and_refine,
)

logger = get_logger(__name__)

_MICRO_QUIZ_ID_NAMESPACE = uuid.UUID("72a4f5e0-89ab-5c11-9e1d-3a4b5c6d7e88")
_USAGE_OK = "OK"
_USAGE_ERROR = "ERROR"
DEFAULT_MICRO_QUIZ_COUNT = 3
DEFAULT_MICRO_QUIZ_TYPES = ("MCQ_SINGLE", "TRUE_FALSE")

_LEADING_CHUNK_SOURCE_RE = re.compile(
    r"^\s*(?:Theo\s+)?Nguồn\s*\([^)]*chunk[^)]*\)\s*(?:nêu\s*(?:rằng)?|cho\s*biết\s*(?:rằng)?|chỉ\s*ra\s*(?:rằng)?|:)?[,:\s]*",
    re.IGNORECASE,
)
_INLINE_CHUNK_PAREN_RE = re.compile(r"\s*\([^)]*\bchunk(?:_id)?\b[^)]*\)", re.IGNORECASE)
_INLINE_CHUNK_TOKEN_RE = re.compile(r"\bchunk(?:_id)?\s*[:=#]?\s*[0-9a-zA-Z_-]{4,}\b", re.IGNORECASE)
_BRACKET_REF_RE = re.compile(r"\s*\[\d+\]")
_BLOOM_META_RE = re.compile(
    r"(?:Đây\s+là\s+)?mức\s+độ\s+(?:nhớ|hiểu|vận\s+dụng|phân\s+tích|đánh\s+giá|sáng\s+tạo)"
    r"(?:\s*\(\s*(?:remember|understand|apply|analyze|evaluate|create)\s*\))?"
    r"(?:\s*(?:vì|do|bởi\s+vì|nhằm|:))?\s*",
    re.IGNORECASE,
)
_BLOOM_PAREN_RE = re.compile(
    r"\s*\(\s*(?:remember|understand|apply|analyze|evaluate|create)\s*\)",
    re.IGNORECASE,
)


def _sanitize_explanation(text: str) -> str:
    """Loại bỏ mã chunk nội bộ, ký hiệu [1] và nhãn Bloom khỏi giải thích cho người học."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    cleaned = _LEADING_CHUNK_SOURCE_RE.sub("Theo nội dung bài giảng, ", cleaned)
    cleaned = _INLINE_CHUNK_PAREN_RE.sub("", cleaned)
    cleaned = _INLINE_CHUNK_TOKEN_RE.sub("", cleaned)
    cleaned = _BRACKET_REF_RE.sub("", cleaned)
    cleaned = _BLOOM_META_RE.sub("", cleaned)
    cleaned = _BLOOM_PAREN_RE.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,;")
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


class _UsageAccumulator:
    def __init__(self) -> None:
        self.total = LLMUsage()

    def __call__(self, usage: LLMUsage) -> None:
        self.total = self.total + usage


@dataclass(frozen=True)
class GenerateMicroQuizRequest:
    course_id: str
    micro_content_id: str
    topic_title: str = ""
    topic_summary: str | None = None
    lo_id: str | None = None
    lo_code: str | None = None
    lo_statement_vi: str | None = None
    lo_statement_en: str | None = None
    document_id: str | None = None
    page_from: int = 1
    page_to: int = 1
    script: dict[str, Any] | None = None
    count: int = DEFAULT_MICRO_QUIZ_COUNT
    question_types: tuple[str, ...] = DEFAULT_MICRO_QUIZ_TYPES
    source_chunks: tuple[dict[str, Any], ...] | None = None


@dataclass(frozen=True)
class GenerateMicroQuizResponse:
    course_id: str
    micro_content_id: str
    lo_id: str | None = None
    items: tuple[GeneratedQuizItem, ...] = ()
    rejected: tuple[RejectedDraft, ...] = ()

    @property
    def quiz_count(self) -> int:
        return len(self.items)

    @property
    def question_ids(self) -> list[str]:
        return [item.question_id for item in self.items]


def _allocate_counts(total: int, question_types: tuple[str, ...]) -> list[tuple[str, int]]:
    valid_types = [t for t in question_types if t in QUESTION_MODELS]
    if not valid_types:
        valid_types = list(DEFAULT_MICRO_QUIZ_TYPES)
    n = len(valid_types)
    base, rem = divmod(total, n)
    alloc: list[tuple[str, int]] = []
    for idx, qtype in enumerate(valid_types):
        cnt = base + (1 if idx < rem else 0)
        if cnt > 0:
            alloc.append((qtype, cnt))
    return alloc


class GenerateMicroQuizUseCase:
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
        self, request: GenerateMicroQuizRequest
    ) -> Result[GenerateMicroQuizResponse, Exception]:
        trace_id = str(uuid.uuid4())
        count = max(1, min(10, int(request.count or DEFAULT_MICRO_QUIZ_COUNT)))
        page_from = max(1, int(request.page_from or 1))
        page_to = max(page_from, int(request.page_to or page_from))

        course_title, lo_code, lo_vi, lo_en = self._resolve_curriculum_labels(request)
        context_parts, source_chunk_ids = self._build_context(request, page_from=page_from, page_to=page_to)
        if not context_parts or not source_chunk_ids:
            return Err(
                ProcessingError(
                    f"Không tìm thấy đoạn nguồn trong trang {page_from}–{page_to} hay kịch bản cho micro-content {request.micro_content_id}"
                )
            )

        all_items: list[GeneratedQuizItem] = []
        all_rejected: list[RejectedDraft] = []
        seen_ids: set[str] = set()
        writer_model = self._writer.client.model_id
        judge_model = self._verifier.client.model_id

        for question_type, sub_count in _allocate_counts(count, request.question_types):
            question_model, schema_name = QUESTION_MODELS[question_type]
            payload_model = CurriculumQuizPayload[question_model]  # type: ignore[valid-type]

            gen_prompt = micro_prompts.micro_quiz_generation(
                course_title=course_title,
                topic_title=request.topic_title or f"Trang {page_from}–{page_to}",
                topic_summary=request.topic_summary,
                page_from=page_from,
                page_to=page_to,
                lo_code=lo_code,
                lo_vi=lo_vi,
                lo_en=lo_en,
                count=sub_count,
                context_parts=context_parts,
                question_type=question_type,
            )
            started = time.monotonic()
            usage = _UsageAccumulator()
            gen_res = generate_structured_payload(
                llm_client=self._writer.client,
                prompt=gen_prompt.user,
                system=gen_prompt.system,
                schema_model=payload_model,
                repair_schema_name=f"MicroQuiz{schema_name}s",
                on_usage=usage,
            )
            self._log_usage(
                role=self._writer,
                use_case=f"micro_quiz_{question_type.lower()}",
                status=_USAGE_OK if gen_res.is_ok() else _USAGE_ERROR,
                latency_ms=int((time.monotonic() - started) * 1000),
                trace_id=trace_id,
                course_id=request.course_id,
                usage=usage.total,
            )
            if gen_res.is_err():
                return Err(gen_res.error)

            raw_questions: list[QuestionBase] = list(gen_res.unwrap().questions[:sub_count])
            refined = self._verify_and_refine(
                questions=raw_questions,
                context_parts=context_parts,
                page_from=page_from,
                page_to=page_to,
                trace_id=trace_id,
                course_id=request.course_id,
                question_type=question_type,
                schema=(question_model, schema_name),
            )
            for r in refined.rejected:
                all_rejected.append(
                    RejectedDraft(reason=r.reason, regeneration_attempts=r.regeneration_attempts)
                )

            targets = balanced_positions(
                len(refined.kept),
                seed=f"{request.micro_content_id}|{question_type}",
            )
            for idx, kept in enumerate(refined.kept):
                q = kept.item
                qid = str(
                    uuid.uuid5(
                        _MICRO_QUIZ_ID_NAMESPACE,
                        f"{request.micro_content_id}|{question_type}|{q.question.strip()}",
                    )
                )
                if qid in seen_ids:
                    continue
                valid_cids = tuple(c for c in q.source_chunk_ids if c in source_chunk_ids)
                choices, correct_index, correct_answer = answer_layout(
                    q, question_type, seed=qid, target_index=targets[idx]
                )
                if correct_answer is None and 0 <= correct_index < len(choices):
                    correct_answer = choices[correct_index]
                problem = answer_key_problem(question_type, q.question, choices, correct_answer)
                if problem is not None:
                    all_rejected.append(
                        RejectedDraft(
                            reason=problem,
                            regeneration_attempts=kept.verification.regeneration_attempts,
                        )
                    )
                    continue
                seen_ids.add(qid)
                all_items.append(
                    GeneratedQuizItem(
                        question_id=qid,
                        question_type=question_type,
                        question=q.question,
                        choices=tuple(choices),
                        correct_index=correct_index,
                        correct_answer=correct_answer,
                        explanation=_sanitize_explanation(q.explanation),
                        bloom_level=2,
                        source_chunk_ids=valid_cids or tuple(source_chunk_ids[:3]),
                        model_id=writer_model,
                        judge_model=judge_model,
                        verification=kept.verification,
                        difficulty=q.difficulty,
                    )
                )

        if not all_items:
            return Err(
                ProcessingError(
                    f"Không còn câu hỏi củng cố hợp lệ nào cho micro-content {request.micro_content_id}"
                )
            )

        logger.info(
            "generate_micro_quiz.done",
            course_id=request.course_id,
            micro_content_id=request.micro_content_id,
            count=len(all_items),
            rejected=len(all_rejected),
        )
        return Ok(
            GenerateMicroQuizResponse(
                course_id=request.course_id,
                micro_content_id=request.micro_content_id,
                lo_id=request.lo_id,
                items=tuple(all_items),
                rejected=tuple(all_rejected),
            )
        )

    def _resolve_curriculum_labels(
        self, request: GenerateMicroQuizRequest
    ) -> tuple[str, str, str, str | None]:
        course_title = resolve_course_title(self._metadata_store, request.course_id)
        lo_code = request.lo_code or "MICRO"
        lo_vi = request.lo_statement_vi or request.topic_title or "Nội dung bài giảng"
        lo_en = request.lo_statement_en

        if request.lo_id and not request.lo_statement_vi:
            cur = self._metadata_store.get_curriculum(request.course_id)
            if cur.is_ok() and cur.unwrap() is not None:
                _, _, los, _ = cur.unwrap()
                match = next((lo for lo in los if lo.lo_id == request.lo_id), None)
                if match is not None:
                    lo_code = match.code
                    lo_vi = match.statement_vi
                    lo_en = match.statement_en
        return course_title, lo_code, lo_vi, lo_en

    def _build_context(
        self,
        request: GenerateMicroQuizRequest,
        *,
        page_from: int,
        page_to: int,
    ) -> tuple[list[str], list[str]]:
        parts: list[str] = []
        chunk_ids: list[str] = []

        raw_chunks: list[tuple[str, int | None, str]] = []
        if request.source_chunks:
            for c in request.source_chunks:
                text = str(c.get("content") or "").strip()
                if text:
                    raw_chunks.append((str(c["chunk_id"]), c.get("page_number"), text))
        elif request.document_id:
            res = self._metadata_store.list_chunks(request.document_id)
            if res.is_ok():
                all_chunks: list[StoredChunkMetadata] = res.unwrap()
                chosen = filter_chunks_in_page_range(all_chunks, page_from, page_to)
                for c in chosen:
                    formatted = format_chunk_with_heading(c)
                    if formatted:
                        raw_chunks.append((str(c.chunk_id), c.page_number, formatted))

        for idx, (cid, pno, content) in enumerate(raw_chunks, start=1):
            parts.append(f"[{idx}] (chunk_id={cid}, page={pno or 'n/a'})\n{content}")
            chunk_ids.append(cid)

        # Thêm lời giảng đã duyệt của kịch bản để câu hỏi bám mạch video, giữ chunk_id thật cho citation.
        scenes = (request.script or {}).get("scenes") if isinstance(request.script, dict) else None
        if isinstance(scenes, list) and scenes:
            script_lines: list[str] = []
            for idx, sc in enumerate(scenes, start=1):
                if not isinstance(sc, dict):
                    continue
                title = str(sc.get("title") or f"Cảnh {idx}").strip()
                narration = str(sc.get("narration") or "").strip()
                pno = sc.get("slide_page")
                if narration:
                    page_tag = f"trang {pno}" if pno else "slide"
                    script_lines.append(f"- [{title} ({page_tag})]: {narration}")
            if script_lines:
                if not chunk_ids:
                    fallback_cid = str(
                        uuid.uuid5(
                            _MICRO_QUIZ_ID_NAMESPACE,
                            f"script:{request.micro_content_id}",
                        )
                    )
                    chunk_ids.append(fallback_cid)
                    parts.append(
                        f"[1] (chunk_id={fallback_cid}, page={page_from})\n"
                        + "\n".join(script_lines)
                    )
                else:
                    parts.append(
                        f"[Kịch bản video đã soạn — chunk_id={chunk_ids[0]}]\n"
                        + "\n".join(script_lines)
                    )

        return parts, chunk_ids

    def _verify_and_refine(
        self,
        *,
        questions: list[QuestionBase],
        context_parts: list[str],
        page_from: int = 1,
        page_to: int = 1,
        trace_id: str,
        course_id: str,
        question_type: str,
        schema: tuple[type[QuestionBase], str],
    ) -> RefineResult[QuestionBase]:
        def judge(current: list[QuestionBase]) -> dict[int, JudgeVerdict] | None:
            prompt = micro_prompts.micro_quiz_verification(
                page_from=page_from,
                page_to=page_to,
                context_parts=context_parts,
                questions=current,
                question_type=question_type,
            )
            started = time.monotonic()
            usage = _UsageAccumulator()
            res = generate_structured_payload(
                llm_client=self._verifier.client,
                prompt=prompt.user,
                system=prompt.system,
                schema_model=QuizVerificationPayload,
                repair_schema_name="MicroQuizVerification",
                on_usage=usage,
            )
            self._log_usage(
                role=self._verifier,
                use_case=f"micro_quiz_verify_{question_type.lower()}",
                status=_USAGE_OK if res.is_ok() else _USAGE_ERROR,
                latency_ms=int((time.monotonic() - started) * 1000),
                trace_id=trace_id,
                course_id=course_id,
                usage=usage.total,
            )
            if res.is_err():
                return None
            return collect_verdicts(
                (v.question_index, v.verdict, v.reason) for v in res.unwrap().verdicts
            )

        def regenerate(original: QuestionBase, reason: str) -> QuestionBase | None:
            prompt = micro_prompts.micro_quiz_regeneration(
                page_from=page_from,
                page_to=page_to,
                context_parts=context_parts,
                original=original,
                reason=reason,
                question_type=question_type,
            )
            schema_model, schema_name = schema
            started = time.monotonic()
            usage = _UsageAccumulator()
            res = generate_structured_payload(
                llm_client=self._writer.client,
                prompt=prompt.user,
                system=prompt.system,
                schema_model=schema_model,
                repair_schema_name=schema_name,
                on_usage=usage,
            )
            self._log_usage(
                role=self._writer,
                use_case=f"micro_quiz_regen_{question_type.lower()}",
                status=_USAGE_OK if res.is_ok() else _USAGE_ERROR,
                latency_ms=int((time.monotonic() - started) * 1000),
                trace_id=trace_id,
                course_id=course_id,
                usage=usage.total,
            )
            return None if res.is_err() else res.unwrap()

        return verify_and_refine(
            questions,
            judge=judge,
            regenerate=regenerate,
            max_retries=self._max_verification_retries,
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
            logger.warning("generate_micro_quiz.usage_log_failed", error=str(res.error))
