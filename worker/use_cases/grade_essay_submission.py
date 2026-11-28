"""
GradeEssaySubmissionUseCase — chấm bài luận sinh viên nộp file vào một
Assignment Canvas có gắn công cụ làm Asset Processor.

core-api đã tải file về MinIO và phát ESSAY_GRADING_REQUESTED kèm đề, rubric,
điểm tối đa và khoá MinIO của từng file. Ở đây: đọc chữ từ từng file (chỉ đọc
object storage, không đọc bảng nào), cho LLM chấm từng tiêu chí trong rubric
của giảng viên, kẹp điểm vào khoảng hợp lệ, cộng lại. Runner gửi
ESSAY_GRADING_RESULT; core-api ghi essay_submissions rồi gửi kết quả về
SpeedGrader như một báo cáo — điểm gợi ý, giảng viên chốt điểm cuối.

Không chấm được thì không bịa điểm: kết quả FAILED kèm mã lỗi, core-api báo
cho giảng viên chấm tay.
"""
import math
import tempfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from document_chunk.application.services.structured_output import generate_structured_payload
from document_chunk.domain.exceptions import ProcessingError
from document_chunk.domain.ports.file_storage import IFileStorage
from document_chunk.domain.ports.llm_client import LLMUsage
from document_chunk.domain.ports.metadata_store import IMetadataStore
from document_chunk.shared.logger import get_logger
from document_chunk.shared.result import Err, Ok, Result

from worker.prompts import essay_grading as prompts
from worker.services.essay_text import UnsupportedAsset, extract_text
from worker.use_cases.shared.grading_verdicts import MalformedGrade, check_verdict_indices
from worker.use_cases.shared.llm_role import LlmRole

logger = get_logger(__name__)

# Mã lỗi gửi về Canvas: mã chuẩn của Asset Report Service.
UNSUPPORTED = "UNSUPPORTED_ASSET_TYPE"
PROCESSING_FAILED = "PROCESSING_FAILED"

GRADED = "GRADED"
FAILED = "FAILED"

# Khớp maxLength trong essay-grading-result.json: core-api từ chối chuỗi dài hơn.
MAX_ERROR_CHARS = 2000
MAX_FEEDBACK_CHARS = 8000
MAX_COMMENT_CHARS = 4000
MAX_MODEL_CHARS = 200

# Giảng viên đọc câu này trong báo cáo lỗi trên SpeedGrader.
LEGACY_ERROR = (
    "Lệnh chấm được tạo trước bản cập nhật, không kèm đề và file cần chấm; "
    "AI không chấm lại tự động bài này."
)


class _CriterionVerdict(BaseModel):
    index: int
    points: float
    comment: str = ""


class _EssayGrade(BaseModel):
    criteria: list[_CriterionVerdict] = Field(min_length=1)
    feedback: str = ""


@dataclass(frozen=True)
class EssayGradingRequest:
    """Lệnh ESSAY_GRADING_REQUESTED, đủ để chấm mà không đọc DB."""

    event_id: str
    essay_submission_id: str
    course_id: str
    user_id: str | None = None
    title: str = ""
    instructions: str = ""
    rubric: tuple[Mapping[str, Any], ...] = ()
    max_score: float = 0.0
    assets: tuple[Mapping[str, Any], ...] = ()
    # Lệnh dạng cũ chỉ có essay_submission_id: không có gì để chấm.
    legacy: bool = False


@dataclass(frozen=True)
class GradeEssayResponse:
    status: str  # GRADED | FAILED
    payload: dict[str, Any]  # payload ESSAY_GRADING_RESULT

    @property
    def score(self) -> float | None:
        result = self.payload.get("result")
        return result["score"] if isinstance(result, dict) else None


def score_criteria(rubric, verdicts: list[_CriterionVerdict]) -> list[dict]:
    """
    Điểm từng tiêu chí, kẹp vào [0, max].

    Model phải chấm đủ và đúng một lần từng tiêu chí (index 0..n-1). Index ngoài
    khoảng (đếm từ 1), trùng, thiếu tiêu chí hay điểm không phải số hữu hạn là lỗi
    định dạng: ném ``MalformedGrade`` chứ không tính tiêu chí thiếu là 0 — điểm đó
    sẽ là điểm bịa do lỗi của model.
    """
    check_verdict_indices(
        (v.index for v in verdicts), len(rubric), require_all=True, what="tiêu chí"
    )
    if not all(math.isfinite(v.points) for v in verdicts):
        raise MalformedGrade("AI trả điểm không phải số hữu hạn")
    by_index = {v.index: v for v in verdicts}
    out = []
    for idx, criterion in enumerate(rubric):
        max_points = float(criterion.get("points") or 0)
        verdict = by_index[idx]
        points = min(max(verdict.points, 0.0), max_points)
        out.append({
            "criterion": str(criterion.get("criterion", "")),
            "points": round(points, 2),
            "max_points": max_points,
            "comment": verdict.comment.strip()[:MAX_COMMENT_CHARS],
        })
    return out


def failed_payload(essay_submission_id: str, error_code: str, error: str) -> dict[str, Any]:
    return {
        "essay_submission_id": essay_submission_id,
        "status": FAILED,
        "error_code": error_code,
        "error": (error or error_code)[:MAX_ERROR_CHARS],
    }


class GradeEssaySubmissionUseCase:
    def __init__(
        self,
        file_storage: IFileStorage,
        metadata_store: IMetadataStore,
        grader: LlmRole,
    ) -> None:
        self._files = file_storage
        # Chỉ để ghi llm_usage_logs.
        self._metadata_store = metadata_store
        # Chấm là việc của judge, như chấm tự luận ngắn và kiểm chứng câu hỏi.
        self._grader = grader

    def execute(
        self, request: EssayGradingRequest, *, final_attempt: bool = True
    ) -> Result[GradeEssayResponse, Exception]:
        """
        ``final_attempt`` False: lỗi tạm thời (MinIO, model) trả Err để hàng đợi
        thử lại. True: kết quả FAILED để core-api báo giảng viên chấm tay.
        """
        sid = request.essay_submission_id
        if request.legacy:
            logger.warning("grade_essay.legacy_command", essay_submission_id=sid)
            return Ok(GradeEssayResponse(FAILED, failed_payload(sid, PROCESSING_FAILED, LEGACY_ERROR)))

        try:
            essay = self._essay_text(request)
        except UnsupportedAsset as exc:
            logger.warning("grade_essay.unsupported", essay_submission_id=sid, error=str(exc))
            return Ok(GradeEssayResponse(FAILED, failed_payload(sid, UNSUPPORTED, str(exc))))
        except Exception as exc:  # noqa: BLE001 -- MinIO/IO: thử lại được
            return self._fail_or_retry(request, f"Không đọc được file từ kho lưu trữ: {exc}", final_attempt)

        prompt, truncated = prompts.grading(
            title=request.title, instructions=request.instructions, rubric=list(request.rubric), essay=essay,
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
        self._record_usage(request, prompt.name, result.is_ok(), usage, started)
        if result.is_err():
            return self._fail_or_retry(request, f"AI không trả kết quả hợp lệ: {result.error}", final_attempt)

        grade = result.unwrap()
        try:
            criteria = score_criteria(request.rubric, grade.criteria)
        except MalformedGrade as exc:
            # Lỗi của model chứ không phải của bài: đi đường thử lại như lỗi model,
            # lượt cuối giao giảng viên chấm tay — không bao giờ thành điểm.
            logger.warning("grade_essay.malformed_grade", essay_submission_id=sid, error=str(exc))
            return self._fail_or_retry(request, str(exc), final_attempt)
        # Tổng của các điểm đã kẹp: không bao giờ vượt max_score (chỉ lệch làm tròn).
        score = round(sum(c["points"] for c in criteria), 2)
        model = self._grader.model_id
        payload = {
            "essay_submission_id": sid,
            "status": GRADED,
            "result": {
                "score": score,
                "max_score": request.max_score,
                "criteria": criteria,
                "feedback": grade.feedback.strip()[:MAX_FEEDBACK_CHARS],
                "truncated": truncated,
                "model": model[:MAX_MODEL_CHARS] if isinstance(model, str) and model else None,
            },
        }
        logger.info("grade_essay.done", essay_submission_id=sid, score=score, max_score=request.max_score)
        return Ok(GradeEssayResponse(GRADED, payload))

    def _essay_text(self, request: EssayGradingRequest) -> str:
        parts: list[tuple[str, str]] = []
        with tempfile.TemporaryDirectory(prefix="essay-") as tmp:
            for asset in request.assets:
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

    def _fail_or_retry(self, request: EssayGradingRequest, error: str, final_attempt: bool):
        if not final_attempt:
            return Err(ProcessingError(error))
        logger.warning("grade_essay.failed", essay_submission_id=request.essay_submission_id, error=error)
        return Ok(GradeEssayResponse(
            FAILED, failed_payload(request.essay_submission_id, PROCESSING_FAILED, error),
        ))

    def _record_usage(
        self, request: EssayGradingRequest, use_case: str, ok: bool, usage: LLMUsage, started: float,
    ) -> None:
        logged = self._metadata_store.record_llm_usage(
            provider=self._grader.provider,
            model=self._grader.model_id,
            use_case=use_case,
            status="OK" if ok else "ERROR",
            course_id=request.course_id,
            user_id=request.user_id,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=usage.cost_usd,
            latency_ms=int((time.monotonic() - started) * 1000),
            trace_id=str(uuid.uuid4()),
        )
        if logged.is_err():
            logger.warning("grade_essay.usage_log_failed", error=str(logged.error))


__all__ = [
    "EssayGradingRequest",
    "GradeEssayResponse",
    "GradeEssaySubmissionUseCase",
    "LEGACY_ERROR",
    "PROCESSING_FAILED",
    "UNSUPPORTED",
    "failed_payload",
    "score_criteria",
]
