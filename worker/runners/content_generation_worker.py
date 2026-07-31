"""BullMQ worker for grounded curriculum quiz-generation requests."""

import asyncio
from dataclasses import dataclass
from typing import Any

from bullmq.custom_errors import UnrecoverableError

from document_chunk.domain.ports.job_queue import CONTENT_GENERATION_QUEUE_NAME
from document_chunk.shared.logger import get_logger

from worker.config import get_worker_settings
from worker.container import build_curriculum_quiz_container
from worker.runtime import (
    build_redis_options,
    install_signal_handlers,
    maybe_start_health_server,
    setup_worker_runtime,
    shutdown_runtime,
)
from worker.use_cases.generate_curriculum_quiz import GenerateCurriculumQuizRequest

logger = get_logger(__name__)


@dataclass(frozen=True)
class ContentGenerationJob:
    request_id: str
    course_id: str
    content_type: str
    scope: dict[str, Any]


def _parse_job(data: Any) -> ContentGenerationJob:
    if not isinstance(data, dict):
        raise ValueError("job data must be an object")
    event_payload = data.get("payload")
    if not isinstance(event_payload, dict):
        raise ValueError("job payload must be an object")
    scope = event_payload.get("scope")
    if not isinstance(scope, dict):
        raise ValueError("content-generation scope must be an object")
    return ContentGenerationJob(
        request_id=str(event_payload["request_id"]),
        course_id=str(event_payload["course_id"]),
        content_type=str(event_payload["type"]),
        scope=scope,
    )


def _quiz_request(job: ContentGenerationJob) -> GenerateCurriculumQuizRequest:
    if job.content_type != "quiz":
        raise ValueError(
            f"content-generation worker currently supports type='quiz', got "
            f"{job.content_type!r}"
        )

    scope = job.scope
    target_kind = str(scope.get("target_kind") or scope.get("targetKind") or "")
    target_code = str(scope.get("target_code") or scope.get("targetCode") or "")

    if not target_kind:
        for kind, keys in (
            ("lo", ("lo_code", "loCode", "lo_id", "loId")),
            ("chapter", ("chapter_code", "chapterCode")),
            ("assessment", ("assessment_code", "assessmentCode")),
        ):
            value = next((scope.get(key) for key in keys if scope.get(key)), None)
            if value is not None:
                target_kind = kind
                target_code = str(value)
                break

    if target_kind == "lo" and ":" in target_code:
        target_code = target_code.rsplit(":", 1)[-1]
    if not target_kind or not target_code:
        raise ValueError(
            "scope must include target_kind/target_code or an LO/chapter/assessment code"
        )

    return GenerateCurriculumQuizRequest(
        course_id=job.course_id,
        target_kind=target_kind,
        target_code=target_code,
        style=str(scope.get("style") or "quiz"),
        bloom_level=scope.get("bloom_level") or scope.get("bloomLevel") or scope.get("bloom"),
        count=int(scope.get("count") or 5),
    )


async def _process_job(container, bull_job, job_token):
    del job_token
    try:
        job = _parse_job(bull_job.data)
    except (KeyError, TypeError, ValueError) as exc:
        raise UnrecoverableError(f"invalid content-generation job: {exc}") from exc

    try:
        request = _quiz_request(job)
    except (TypeError, ValueError) as exc:
        container.metadata_store.update_content_generation_request(
            request_id=job.request_id,
            status="FAILED",
            last_error=str(exc),
        )
        raise UnrecoverableError(f"invalid content-generation request: {exc}") from exc

    running_result = container.metadata_store.update_content_generation_request(
        request_id=job.request_id,
        status="RUNNING",
        last_error=None,
    )
    if running_result.is_err():
        raise running_result.error

    result = container.generate_curriculum_quiz_use_case.execute(request)
    if result.is_err():
        container.metadata_store.update_content_generation_request(
            request_id=job.request_id,
            status="FAILED",
            last_error=str(result.error),
        )
        raise result.error

    response = result.unwrap()
    completed_result = container.metadata_store.update_content_generation_request(
        request_id=job.request_id,
        status="SUCCEEDED",
        generated_count=response.quiz_count,
        last_error=None,
    )
    if completed_result.is_err():
        raise completed_result.error

    logger.info(
        "content_generation_worker.job.completed",
        request_id=job.request_id,
        course_id=job.course_id,
        lo_id=response.lo_id,
        quiz_count=response.quiz_count,
    )
    return {
        "request_id": job.request_id,
        "course_id": response.course_id,
        "lo_id": response.lo_id,
        "quiz_count": response.quiz_count,
        "question_ids": response.question_ids,
    }


async def _main() -> None:
    settings = get_worker_settings("content_generation")
    health_state = setup_worker_runtime(settings)
    container = build_curriculum_quiz_container(settings)
    health_server = maybe_start_health_server(settings, health_state)
    stop_event = asyncio.Event()
    install_signal_handlers(stop_event)

    from bullmq import Worker

    worker = Worker(
        CONTENT_GENERATION_QUEUE_NAME,
        lambda job, job_token: _process_job(container, job, job_token),
        {
            "connection": build_redis_options(settings),
            "concurrency": settings.worker.concurrency,
        },
    )
    logger.info(
        "content_generation_worker.started",
        queue=CONTENT_GENERATION_QUEUE_NAME,
        concurrency=settings.worker.concurrency,
        mcp_enabled=settings.learning_context_mcp.enabled,
    )

    try:
        await stop_event.wait()
    finally:
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
