"""
GradeEssaySubmissionUseCase — chấm bài luận sinh viên nộp file vào một
Assignment Canvas có gắn công cụ làm Asset Processor.

core-api đã tải file về MinIO và phát ESSAY_GRADING_REQUESTED. Ở đây: đọc chữ
từ từng file, cho LLM chấm từng tiêu chí trong rubric của giảng viên, kẹp điểm
vào khoảng hợp lệ, cộng lại, ghi essay_submissions.result. core-api gửi kết
quả về SpeedGrader như một báo cáo — điểm gợi ý, giảng viên chốt điểm cuối.

Không chấm được thì không bịa điểm: dòng chuyển FAILED kèm mã lỗi, core-api
báo cho giảng viên chấm tay.
"""
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from document_chunk.application.services.structured_output import generate_structured_payload
from document_chunk.domain.exceptions import ProcessingError
from document_chunk.domain.ports.file_storage import IFileStorage
from document_chunk.domain.ports.llm_client import LLMUsage
from document_chunk.domain.ports.metadata_store import IMetadataStore
from document_chunk.shared.logger import get_logger
from document_chunk.shared.result import Err, Ok, Result

from worker.llm_role import LlmRole
from worker.prompts import essay_grading as prompts
from worker.services.essay_submission_store import QUEUED, EssayJob, EssaySubmissionStore
from worker.services.essay_text import UnsupportedAsset, extract_text

logger = get_logger(__name__)

# Mã lỗi gửi về Canvas: hai mã đầu là mã chuẩn của Asset Report Service.
UNSUPPORTED = "UNSUPPORTED_ASSET_TYPE"
PROCESSING_FAILED = "PROCESSING_FAILED"


class _CriterionVerdict(BaseModel):
    index: int
    points: float
    comment: str = ""


class _EssayGrade(BaseModel):
    criteria: list[_CriterionVerdict] = Field(min_length=1)
    feedback: str = ""


@dataclass(frozen=True)
class GradeEssayResponse:
    status: str  # GRADED | FAILED | SKIPPED
    score: float | None = None


def score_criteria(rubric: list[dict], verdicts: list[_CriterionVerdict]) -> list[dict]:
    """Điểm từng tiêu chí, kẹp vào [0, max]. Tiêu chí model bỏ sót tính 0 và nói rõ."""
    by_index = {v.index: v for v in verdicts}
    out = []
    for idx, criterion in enumerate(rubric):
        max_points = float(criterion.get("points") or 0)
        verdict = by_index.get(idx)
        points = min(max(verdict.points, 0.0), max_points) if verdict else 0.0
        out.append({
            "criterion": str(criterion.get("criterion", "")),
            "points": round(points, 2),
            "max_points": max_points,
            "comment": verdict.comment.strip() if verdict else "AI không chấm tiêu chí này.",
        })
    return out


class GradeEssaySubmissionUseCase:
    def __init__(
        self,
        store: EssaySubmissionStore,
        file_storage: IFileStorage,
        metadata_store: IMetadataStore,
        grader: LlmRole,
    ) -> None:
        self._store = store
        self._files = file_storage
        self._metadata_store = metadata_store
        # Chấm là việc của judge, như chấm tự luận ngắn và kiểm chứng câu hỏi.
        self._grader = grader

    def execute(
        self, essay_submission_id: str, *, final_attempt: bool = True
    ) -> Result[GradeEssayResponse, Exception]:
        """
        ``final_attempt`` False: lỗi tạm thời (MinIO, model) trả Err để hàng đợi
        thử lại. True: chuyển FAILED để core-api báo giảng viên chấm tay.
        """
        job = self._store.get(essay_submission_id)
        if job is None:
            return Err(ProcessingError(f"Không có bài nộp {essay_submission_id}"))
        if job.status != QUEUED:
            return Ok(GradeEssayResponse(status="SKIPPED"))

        try:
            essay = self._essay_text(job)
        except UnsupportedAsset as exc:
            self._store.mark_failed(essay_submission_id, UNSUPPORTED, str(exc))
            logger.warning("grade_essay.unsupported", essay_submission_id=essay_submission_id, error=str(exc))
            return Ok(GradeEssayResponse(status="FAILED"))
        except Exception as exc:  # noqa: BLE001 -- MinIO/IO: thử lại được
            return self._fail_or_retry(job, f"Không đọc được file từ kho lưu trữ: {exc}", final_attempt)

        prompt, truncated = prompts.grading(
            title=job.title, instructions=job.instructions, rubric=job.rubric, essay=essay,
        )
        usage = LLMUsage()

        def on_usage(value: LLMUsage) -> None:
            nonlocal usage
            usage = usage + value

        started = time.monotonic()
        result = generate_structured_payload(
            llm_client=self._grader.client,
            prompt=prompt.user,
            system=prompt.system,
            schema_model=_EssayGrade,
            repair_schema_name="EssayGrade",
            on_usage=on_usage,
        )
        self._record_usage(job, prompt.name, result.is_ok(), usage, started)
        if result.is_err():
            return self._fail_or_retry(job, f"AI không trả kết quả hợp lệ: {result.error}", final_attempt)

        grade = result.unwrap()
        criteria = score_criteria(job.rubric, grade.criteria)
        score = round(sum(c["points"] for c in criteria), 2)
        saved = self._store.save_result(essay_submission_id, {
            "score": score,
            "max_score": job.max_score,
            "criteria": criteria,
            "feedback": grade.feedback.strip(),
            "truncated": truncated,
            "model": self._grader.model_id,
        })
        logger.info(
            "grade_essay.done",
            essay_submission_id=essay_submission_id,
            score=score,
            max_score=job.max_score,
            saved=saved,
        )
        return Ok(GradeEssayResponse(status="GRADED", score=score))

    def _essay_text(self, job: EssayJob) -> str:
        parts: list[tuple[str, str]] = []
        with tempfile.TemporaryDirectory(prefix="essay-") as tmp:
            for asset in job.assets:
                key = asset.get("storage_key")
                if not key:
                    raise UnsupportedAsset(f"File {asset.get('title') or asset.get('asset_id')} chưa được tải về")
                local = Path(tmp) / Path(key).name
                downloaded = self._files.download(key, local)
                if downloaded.is_err():
                    raise downloaded.error
                parts.append((asset.get("title") or local.name, extract_text(local, asset.get("content_type"))))
        if len(parts) == 1:
            return parts[0][1]
        return "\n\n".join(f"=== {title} ===\n{text}" for title, text in parts)

    def _fail_or_retry(self, job: EssayJob, error: str, final_attempt: bool):
        if not final_attempt:
            return Err(ProcessingError(error))
        self._store.mark_failed(job.essay_submission_id, PROCESSING_FAILED, error)
        logger.warning("grade_essay.failed", essay_submission_id=job.essay_submission_id, error=error)
        return Ok(GradeEssayResponse(status="FAILED"))

    def _record_usage(self, job: EssayJob, use_case: str, ok: bool, usage: LLMUsage, started: float) -> None:
        logged = self._metadata_store.record_llm_usage(
            provider=self._grader.provider,
            model=self._grader.model_id,
            use_case=use_case,
            status="OK" if ok else "ERROR",
            course_id=job.course_id,
            user_id=job.user_id,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=usage.cost_usd,
            latency_ms=int((time.monotonic() - started) * 1000),
            trace_id=str(uuid.uuid4()),
        )
        if logged.is_err():
            logger.warning("grade_essay.usage_log_failed", error=str(logged.error))


__all__ = ["GradeEssayResponse", "GradeEssaySubmissionUseCase", "score_criteria"]
