"""
BullMQ worker — chấm bài của người học, hai hàng đợi trong một process:

- ``answer_grading``: tự luận ngắn trong một lượt nộp quiz
  (ANSWER_GRADING_REQUESTED) và câu tự luận của Micro Content
  (MICRO_CONTENT_GRADING_REQUESTED)
- ``essay_grading``: bài luận nộp file vào Assignment Canvas (Asset Processor),
  core-api đã tải file về MinIO (essay-processor/essay-report.publisher.ts)

Worker tính, core-api ghi: lệnh mang đủ dữ liệu cần chấm, worker không đọc hay
ghi bảng bài làm; chấm xong thì gửi QUIZ_GRADING_RESULT /
MICRO_CONTENT_GRADING_RESULT / ESSAY_GRADING_RESULT về core-api qua hàng đợi
``worker_results`` và chỉ hoàn tất job gốc sau khi Redis đã nhận kết quả.
result_id = ``<KIND>.<event_id của lệnh>``: job gốc retry sau khi đã gửi sinh lại
đúng id đó và BullMQ bỏ bản trùng; mỗi lần giảng viên cho chấm lại là một sự
kiện mới nên có id mới.

Process riêng, không chung với content-generation worker: sinh quiz gọi model
hàng phút, còn người học đang chờ điểm. Mỗi job chạy trong thread để event loop
còn rảnh gia hạn lock BullMQ và để ``concurrency`` thật sự chấm song song.

Run:
    python -m worker.runners.answer_grading_worker
"""
import asyncio
import re
from collections.abc import Mapping
from typing import Any

from ai_runtime.catalog import QUIZ_VERIFICATION
from bullmq.custom_errors import UnrecoverableError

from document_chunk.domain.exceptions import ProcessingError
from document_chunk.domain.ports.job_queue import ANSWER_GRADING_QUEUE_NAME
from document_chunk.shared.logger import get_logger

from worker.config import get_worker_settings
from worker.container import build_answer_grading_container
from worker.runtime import (
    build_redis_options,
    drain_bullmq_worker,
    install_signal_handlers,
    maybe_start_health_server,
    setup_worker_runtime,
    shutdown_runtime,
)
from worker.services.result_publisher import (
    ESSAY_GRADING_RESULT,
    MICRO_CONTENT_GRADING_RESULT,
    QUIZ_GRADING_RESULT,
    make_result_id,
)
from worker.use_cases.grade_essay_submission import EssayGradingRequest
from worker.use_cases.grade_micro_content_answers import (
    GradeMicroContentAnswersRequest,
    micro_content_result_payload,
)
from worker.use_cases.grade_short_answers import (
    AnswerGrade,
    AnswerToGrade,
    GradeShortAnswersRequest,
    quiz_result_payload,
)

logger = get_logger(__name__)

MICRO_CONTENT_GRADING_REQUESTED = "MICRO_CONTENT_GRADING_REQUESTED"
# Khớp ESSAY_GRADING_QUEUE ở core-api essay-processor/essay-grading.ts.
ESSAY_GRADING_QUEUE_NAME = "essay_grading"
# Bài luận dài, mỗi bài một lời gọi model lớn: chấm lần lượt cho đỡ đụng rate limit.
ESSAY_CONCURRENCY = 1

# Khoá trong dữ liệu job (Redis) giữ kết quả đã tính ở lượt thử trước: lượt sau
# không gọi lại model cho câu đã chấm, kể cả khi lượt trước chỉ hỏng lúc gửi kết quả.
PROGRESS_KEY = "grading_progress"

_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
# event_id thành một phần của result_id (make_result_id): kiểm ngay lúc nhận lệnh.
_EVENT_ID = re.compile(r"[A-Za-z0-9_-]{1,150}")


def _payload(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict) or not isinstance(data.get("payload"), dict):
        raise ValueError("grading job must carry an outbox payload")
    return data["payload"]


def _uuid(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _UUID.fullmatch(value):
        raise ValueError(f"{name} must be a uuid")
    return value


def _ids(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty list of ids")
    ids = tuple(_uuid(i, key) for i in value)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{key} has duplicate ids")
    return ids


def _answers(payload: Mapping[str, Any], id_key: str) -> tuple[AnswerToGrade, ...]:
    value = payload.get("answers")
    if not isinstance(value, list) or not value:
        raise ValueError("answers must be a non-empty list")
    answers = []
    for item in value:
        if not isinstance(item, dict) or "grading_input" not in item or "chosen_answer" not in item:
            raise ValueError(f"each answer needs {id_key}, grading_input and chosen_answer")
        answers.append(AnswerToGrade(
            answer_id=_uuid(item.get(id_key), id_key),
            # Đề hỏng không làm hỏng cả lệnh: use case trả NEEDS_REVIEW cho câu đó.
            grading_input=item["grading_input"],
            chosen_answer=item["chosen_answer"],
        ))
    if len({a.answer_id for a in answers}) != len(answers):
        raise ValueError(f"answers have duplicate {id_key}")
    return tuple(answers)


def _event_id(data: Any, job_id: Any = None) -> str:
    """event_id outbox của lệnh (= jobId do OutboxService đặt)."""
    event_id = data.get("event_id") if isinstance(data, dict) else None
    event_id = event_id or job_id
    if not isinstance(event_id, str) or not _EVENT_ID.fullmatch(event_id):
        raise ValueError("grading job needs the outbox event_id")
    return event_id


def parse_grading_job(data: Any, job_id: Any = None) -> GradeShortAnswersRequest:
    """
    Dữ liệu job là bản ghi outbox: ``{event_id, event_type, aggregate_id, payload: {...}}``.
    Lệnh dạng cũ (không có ``answers``) vẫn nhận: các câu trong
    essay_attempt_ids trả NEEDS_REVIEW để giảng viên chấm tay hoặc cho AI chấm
    lại — lần chấm lại phát lệnh dạng mới. Không đọc DB để bù.
    """
    payload = _payload(data)
    submission_id = _uuid(payload.get("submission_id") or data.get("aggregate_id"), "submission_id")
    course_id = _uuid(payload.get("course_id"), "course_id")
    event_id = _event_id(data, job_id)
    if "answers" not in payload:
        return GradeShortAnswersRequest(
            event_id=event_id, submission_id=submission_id, course_id=course_id,
            answers=(), legacy_attempt_ids=_ids(payload, "essay_attempt_ids"),
        )
    return GradeShortAnswersRequest(
        event_id=event_id, submission_id=submission_id, course_id=course_id,
        answers=_answers(payload, "attempt_id"),
    )


def parse_micro_content_grading_job(data: Any, job_id: Any = None) -> GradeMicroContentAnswersRequest:
    payload = _payload(data)
    attempt_id = _uuid(payload.get("attempt_id") or data.get("aggregate_id"), "attempt_id")
    course_id = _uuid(payload.get("course_id"), "course_id")
    micro_content_id = _uuid(payload.get("micro_content_id"), "micro_content_id")
    event_id = _event_id(data, job_id)
    if "answers" not in payload:
        return GradeMicroContentAnswersRequest(
            event_id=event_id, attempt_id=attempt_id, course_id=course_id,
            micro_content_id=micro_content_id, answers=(),
            legacy_answer_ids=_ids(payload, "answer_ids"),
        )
    return GradeMicroContentAnswersRequest(
        event_id=event_id, attempt_id=attempt_id, course_id=course_id,
        micro_content_id=micro_content_id, answers=_answers(payload, "answer_id"),
    )


def parse_essay_job(data: Any, job_id: Any = None) -> EssayGradingRequest:
    """
    Dữ liệu job là bản ghi outbox ESSAY_GRADING_REQUESTED. Lệnh dạng cũ (không có
    ``assets``) thành request ``legacy``: use case trả FAILED để giảng viên chấm tay.
    """
    payload = _payload(data)
    essay_submission_id = _uuid(
        payload.get("essay_submission_id") or data.get("aggregate_id"), "essay_submission_id",
    )
    course_id = _uuid(payload.get("course_id"), "course_id")
    event_id = _event_id(data, job_id)
    if "assets" not in payload:
        return EssayGradingRequest(
            event_id=event_id, essay_submission_id=essay_submission_id, course_id=course_id, legacy=True,
        )
    assets, rubric = payload.get("assets"), payload.get("rubric")
    if not isinstance(assets, list) or not assets or not all(isinstance(a, dict) for a in assets):
        raise ValueError("assets must be a non-empty list of objects")
    if not isinstance(rubric, list) or not rubric or not all(isinstance(c, dict) for c in rubric):
        raise ValueError("rubric must be a non-empty list of criteria")
    max_score = payload.get("max_score")
    if isinstance(max_score, bool) or not isinstance(max_score, (int, float)) or max_score <= 0:
        raise ValueError("max_score must be a positive number")
    user_id = payload.get("user_id")
    return EssayGradingRequest(
        event_id=event_id,
        essay_submission_id=essay_submission_id,
        course_id=course_id,
        user_id=_uuid(user_id, "user_id") if user_id is not None else None,
        title=str(payload.get("title") or ""),
        instructions=str(payload.get("instructions") or ""),
        rubric=tuple(rubric),
        max_score=float(max_score),
        assets=tuple(assets),
    )


def is_final_attempt(bull_job: Any) -> bool:
    """``attemptsMade`` đếm số lần đã thất bại, nên lần đang chạy là lần thứ +1."""
    opts = getattr(bull_job, "opts", None) or {}
    attempts = int(opts.get("attempts") or 1)
    return int(getattr(bull_job, "attemptsMade", 0) or 0) + 1 >= attempts


def load_answer_progress(bull_job: Any, answer_ids: set[str]) -> dict[str, AnswerGrade]:
    """Kết quả từng câu đã tính ở lượt thử trước; mục hỏng hay lạ thì bỏ (chấm lại câu đó)."""
    data = getattr(bull_job, "data", None)
    raw = data.get(PROGRESS_KEY) if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return {}
    grades = (AnswerGrade.from_progress(item) for item in raw)
    return {g.answer_id: g for g in grades if g is not None and g.answer_id in answer_ids}


async def save_progress(bull_job: Any, progress: Any) -> None:
    """
    Ghi kết quả đã tính vào dữ liệu job trước khi để job fail. Ghi không được
    thì lượt sau chấm lại (tốn thêm lời gọi model), không sai kết quả.
    """
    update = getattr(bull_job, "updateData", None)
    data = getattr(bull_job, "data", None)
    if not callable(update) or not isinstance(data, dict):
        return
    try:
        await update({**data, PROGRESS_KEY: progress})
    except Exception as exc:  # noqa: BLE001 -- chỉ là bộ nhớ đệm
        logger.warning("answer_grading_worker.save_progress_failed", job_id=getattr(bull_job, "id", None),
                       error=str(exc))


async def _publish_or_keep(container, bull_job, kind: str, payload: dict, *, source_event_id: str,
                           progress: Any) -> str:
    """Gửi kết quả; Redis không nhận thì giữ kết quả trong job rồi ném lỗi để BullMQ thử lại."""
    try:
        return await container.result_publisher.publish(
            kind, payload,
            result_id=make_result_id(kind, source_event_id),
            source_event_id=source_event_id,
        )
    except Exception:
        await save_progress(bull_job, progress)
        raise


async def process_grading_job(container, bull_job, job_token):
    del job_token
    data = bull_job.data
    event_type = data.get("event_type") if isinstance(data, dict) else None
    job_id = getattr(bull_job, "id", None)
    final = is_final_attempt(bull_job)

    if event_type == MICRO_CONTENT_GRADING_REQUESTED:
        kind, label = MICRO_CONTENT_GRADING_RESULT, "micro-content"
        try:
            request = parse_micro_content_grading_job(data, job_id)
        except (KeyError, TypeError, ValueError) as exc:
            raise UnrecoverableError(f"invalid micro-content grading job: {exc}") from exc
        use_case = container.grade_micro_content_answers_use_case
        build_payload = micro_content_result_payload
        ref = {"attempt_id": request.attempt_id}
    else:
        kind, label = QUIZ_GRADING_RESULT, "answer-grading"
        try:
            request = parse_grading_job(data, job_id)
        except (KeyError, TypeError, ValueError) as exc:
            raise UnrecoverableError(f"invalid answer-grading job: {exc}") from exc
        use_case = container.grade_short_answers_use_case
        build_payload = quiz_result_payload
        ref = {"submission_id": request.submission_id}

    # Use case gọi LLM đồng bộ: chạy ngoài event loop.
    outcome = await asyncio.to_thread(
        use_case.execute, request,
        final_attempt=final,
        prior=load_answer_progress(bull_job, request.answer_ids),
    )
    progress = [g.to_progress() for g in outcome.grades]
    if outcome.retry:
        # Chưa gửi gì: câu đã chấm nằm trong job, lượt sau chỉ chấm lại câu lỗi.
        await save_progress(bull_job, progress)
        raise ProcessingError(f"{label}: {outcome.retry} câu tự luận chưa chấm được ({ref})")

    result_id = await _publish_or_keep(
        container, bull_job, kind, build_payload(request, outcome.grades),
        source_event_id=request.event_id, progress=progress,
    )
    return {**ref, "graded": outcome.graded, "needs_review": outcome.needs_review, "result_id": result_id}


def _essay_progress(bull_job: Any, essay_submission_id: str) -> dict | None:
    data = getattr(bull_job, "data", None)
    raw = data.get(PROGRESS_KEY) if isinstance(data, dict) else None
    if isinstance(raw, dict) and raw.get("essay_submission_id") == essay_submission_id \
            and raw.get("status") in ("GRADED", "FAILED"):
        return raw
    return None


async def process_essay_job(container, bull_job, job_token):
    del job_token
    try:
        request = parse_essay_job(bull_job.data, getattr(bull_job, "id", None))
    except (KeyError, TypeError, ValueError) as exc:
        raise UnrecoverableError(f"invalid essay-grading job: {exc}") from exc

    payload = _essay_progress(bull_job, request.essay_submission_id)
    if payload is None:
        result = await asyncio.to_thread(
            container.grade_essay_submission_use_case.execute,
            request,
            final_attempt=is_final_attempt(bull_job),
        )
        if result.is_err():
            raise result.error
        payload = result.unwrap().payload

    result_id = await _publish_or_keep(
        container, bull_job, ESSAY_GRADING_RESULT, payload,
        source_event_id=request.event_id, progress=payload,
    )
    graded = payload.get("result") if payload["status"] == "GRADED" else None
    return {
        "essay_submission_id": request.essay_submission_id,
        "status": payload["status"],
        "score": graded["score"] if graded else None,
        "result_id": result_id,
    }


async def _main() -> None:
    settings = get_worker_settings("answer_grading")
    health_state = setup_worker_runtime(settings)
    container = build_answer_grading_container(settings)
    health_server = maybe_start_health_server(settings, health_state)
    stop_event = asyncio.Event()
    install_signal_handlers(stop_event)

    from bullmq import Worker

    worker = Worker(
        ANSWER_GRADING_QUEUE_NAME,
        lambda job, job_token: process_grading_job(container, job, job_token),
        {
            "connection": build_redis_options(settings),
            "concurrency": settings.worker.concurrency,
        },
    )
    essay_worker = Worker(
        ESSAY_GRADING_QUEUE_NAME,
        lambda job, job_token: process_essay_job(container, job, job_token),
        {
            "connection": build_redis_options(settings),
            "concurrency": ESSAY_CONCURRENCY,
        },
    )
    logger.info(
        "answer_grading_worker.started",
        queues=[ANSWER_GRADING_QUEUE_NAME, ESSAY_GRADING_QUEUE_NAME],
        concurrency=settings.worker.concurrency,
        # Model chấm được chọn lại mỗi job (admin đổi trên web); đây là lựa chọn lúc khởi động.
        grader=container.roles.role(QUIZ_VERIFICATION).profile,
    )

    try:
        await stop_event.wait()
    finally:
        try:
            await drain_bullmq_worker(essay_worker, timeout_seconds=settings.worker.shutdown_grace_seconds)
        except Exception as exc:  # noqa: BLE001 -- vẫn phải đóng worker còn lại
            logger.warning("answer_grading_worker.essay_close_failed", error=str(exc))
        await shutdown_runtime(
            stop_event=stop_event,
            container=container,
            health_server=health_server,
            bullmq_worker=worker,
            timeout_seconds=settings.worker.shutdown_grace_seconds,
        )


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
