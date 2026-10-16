"""
BullMQ worker — chấm bài của người học, hai hàng đợi trong một process:

- ``answer_grading``: tự luận ngắn trong một lượt nộp quiz
- ``essay_grading``: bài luận nộp file vào Assignment Canvas (Asset Processor),
  core-api đã tải file về MinIO (essay-processor/essay-report.publisher.ts)

Process riêng, không chung với content-generation worker: sinh quiz gọi model
hàng phút, còn người học đang chờ điểm. Mỗi job chạy trong thread để event loop
còn rảnh gia hạn lock BullMQ và để ``concurrency`` thật sự chấm song song.

Run:
    python -m worker.runners.answer_grading_worker
"""
import asyncio
from typing import Any

from ai_runtime.catalog import QUIZ_VERIFICATION
from bullmq.custom_errors import UnrecoverableError

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
from worker.use_cases.grade_short_answers import GradeShortAnswersRequest

logger = get_logger(__name__)

# Khớp ESSAY_GRADING_QUEUE ở core-api essay-processor/essay-grading.ts.
ESSAY_GRADING_QUEUE_NAME = "essay_grading"
# Bài luận dài, mỗi bài một lời gọi model lớn: chấm lần lượt cho đỡ đụng rate limit.
ESSAY_CONCURRENCY = 1


def _ids(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(i, str) and i for i in value):
        raise ValueError(f"{key} must be a list of ids")
    return value


def parse_grading_job(data: Any) -> GradeShortAnswersRequest:
    """Dữ liệu job là bản ghi outbox: ``{event_type, aggregate_id, payload: {...}}``."""
    if not isinstance(data, dict) or not isinstance(data.get("payload"), dict):
        raise ValueError("grading job must carry an outbox payload")
    payload = data["payload"]
    submission_id = payload.get("submission_id") or data.get("aggregate_id")
    course_id = payload.get("course_id")
    if not isinstance(submission_id, str) or not isinstance(course_id, str):
        raise ValueError("grading job needs submission_id and course_id")
    essay_ids = _ids(payload, "essay_attempt_ids")
    attempt_ids = _ids(payload, "attempt_ids")
    if not set(essay_ids) <= set(attempt_ids):
        raise ValueError("essay_attempt_ids must be part of attempt_ids")
    return GradeShortAnswersRequest(
        submission_id=submission_id,
        course_id=course_id,
        attempt_ids=attempt_ids,
        essay_attempt_ids=essay_ids,
    )


def is_final_attempt(bull_job: Any) -> bool:
    """``attemptsMade`` đếm số lần đã thất bại, nên lần đang chạy là lần thứ +1."""
    opts = getattr(bull_job, "opts", None) or {}
    attempts = int(opts.get("attempts") or 1)
    return int(getattr(bull_job, "attemptsMade", 0) or 0) + 1 >= attempts


async def process_grading_job(container, bull_job, job_token):
    del job_token
    try:
        request = parse_grading_job(bull_job.data)
    except (KeyError, TypeError, ValueError) as exc:
        raise UnrecoverableError(f"invalid answer-grading job: {exc}") from exc

    # Use case gọi LLM và psycopg đồng bộ: chạy ngoài event loop.
    result = await asyncio.to_thread(
        container.grade_short_answers_use_case.execute,
        request,
        final_attempt=is_final_attempt(bull_job),
    )
    if result.is_err():
        raise result.error
    response = result.unwrap()
    return {
        "submission_id": request.submission_id,
        "graded": response.graded,
        "needs_review": response.needs_review,
        "released": response.released,
    }


def parse_essay_job(data: Any) -> str:
    """Dữ liệu job là bản ghi outbox ESSAY_GRADING_REQUESTED; trả essay_submission_id."""
    if not isinstance(data, dict) or not isinstance(data.get("payload"), dict):
        raise ValueError("essay job must carry an outbox payload")
    essay_submission_id = data["payload"].get("essay_submission_id") or data.get("aggregate_id")
    if not isinstance(essay_submission_id, str) or not essay_submission_id:
        raise ValueError("essay job needs essay_submission_id")
    return essay_submission_id


async def process_essay_job(container, bull_job, job_token):
    del job_token
    try:
        essay_submission_id = parse_essay_job(bull_job.data)
    except (KeyError, TypeError, ValueError) as exc:
        raise UnrecoverableError(f"invalid essay-grading job: {exc}") from exc

    result = await asyncio.to_thread(
        container.grade_essay_submission_use_case.execute,
        essay_submission_id,
        final_attempt=is_final_attempt(bull_job),
    )
    if result.is_err():
        raise result.error
    response = result.unwrap()
    return {
        "essay_submission_id": essay_submission_id,
        "status": response.status,
        "score": response.score,
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
