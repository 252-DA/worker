"""
BullMQ worker for grounded, LO-scoped quiz and lesson-card generation requests.

Worker tính, core-api ghi: mỗi bước của vòng đời request (bắt đầu chạy, xong
kèm câu/card đã kiểm chứng, thất bại) được gửi về core-api thành một kết quả
CONTENT_GENERATION_RESULT trên hàng đợi ``worker_results``
(contracts/integration-events/v1/content-generation-result.json). core-api là
nơi duy nhất ghi ``content_generation_requests``, ``lessons``, ``quiz_items``,
``lesson_cards`` (src/modules/content-generation/handlers/content-generation-result.handler.ts).

Đọc lệnh: ``content_generation_jobs.py``; dựng kết quả: ``reporters/content_generation.py``.
"""

import asyncio
from typing import Any

from bullmq.custom_errors import UnrecoverableError

from document_chunk.domain.ports.job_queue import CONTENT_GENERATION_QUEUE_NAME
from document_chunk.shared.logger import get_logger

from worker.config import get_worker_settings
from worker.container.content_generation import build_curriculum_quiz_container
from worker.reporters.content_generation import ContentGenerationReporter, result_source
from worker.runners.bullmq_jobs import is_final_attempt
from worker.runners.content_generation_jobs import (
    ContentGenerationJob,
    card_request,
    micro_outline_request,
    micro_quiz_request,
    parse_job,
    payload_text,
    quiz_request,
    relations_request,
    request_id_from_job_data,
    video_script_request,
)
from worker.runtime import (
    build_redis_options,
    install_signal_handlers,
    maybe_start_health_server,
    setup_worker_runtime,
    shutdown_runtime,
)

logger = get_logger(__name__)


def _reporter(
    container, bull_job: Any, request_id: str, course_id: str | None, content_type: str | None
) -> ContentGenerationReporter | None:
    """Reporter của job; None nếu không dựng được result_id từ lệnh."""
    data = getattr(bull_job, "data", None)
    event_id = data.get("event_id") if isinstance(data, dict) else None
    source = result_source(event_id, getattr(bull_job, "id", None), request_id)
    if source is None:
        return None
    return ContentGenerationReporter(
        publisher=container.result_publisher,
        source=source,
        request_id=request_id,
        course_id=course_id,
        content_type=content_type,
    )


async def _fail_unparsed_request(container, bull_job: Any, exc: Exception) -> None:
    """
    Gửi kết quả FAILED cho một job hỏng để dòng request không treo ở QUEUED.

    Lỗi gửi được ném ra cho chỗ gọi: job phải retry để còn gửi lại được.
    """
    data = getattr(bull_job, "data", None)
    request_id = request_id_from_job_data(data)
    reporter = (
        _reporter(container, bull_job, request_id, payload_text(data, "course_id"), payload_text(data, "type"))
        if request_id
        else None
    )
    if reporter is None:
        logger.error(
            "content_generation_worker.job.unidentifiable",
            error=str(exc),
            reason="không lấy được request_id; dòng QUEUED phải dọn tay",
        )
        return
    await reporter.failed(f"invalid content-generation job: {exc}")


def _plan(container, job: ContentGenerationJob):
    """(use case, request) cho loại nội dung của job; ValueError nếu không hỗ trợ."""
    if job.content_type == "quiz":
        return container.generate_curriculum_quiz_use_case, quiz_request(job)
    if job.content_type == "card":
        return container.generate_lesson_cards_use_case, card_request(job)
    if job.content_type == "relations":
        return container.extract_relations_use_case, relations_request(job)
    if job.content_type == "micro_outline":
        return container.generate_micro_outline_use_case, micro_outline_request(job)
    if job.content_type == "video_script":
        return container.generate_video_script_use_case, video_script_request(job)
    if job.content_type == "micro_quiz":
        return container.generate_micro_quiz_use_case, micro_quiz_request(job)
    raise ValueError(
        "content-generation worker supports type='quiz', 'card', 'relations', "
        f"'micro_outline', 'video_script' or 'micro_quiz', got {job.content_type!r}"
    )


def _summary(job: ContentGenerationJob, response) -> dict[str, Any]:
    """Giá trị trả về của job gốc (BullMQ giữ làm returnvalue): nhỏ, để tra cứu."""
    if job.content_type == "relations":
        generated: dict[str, Any] = {
            "relations_written": response.relations_written,
            "relations_dropped": response.relations_dropped,
            "chunks_processed": response.chunks_processed,
            "failed_batches": response.failed_batches,
        }
    elif job.content_type == "micro_outline":
        generated = {
            "document_id": response.document_id,
            "section_count": response.section_count,
        }
    elif job.content_type == "video_script":
        generated = {
            "micro_content_id": response.micro_content_id,
            "scene_count": response.scene_count,
            "rejected_count": len(response.rejected),
        }
    elif job.content_type == "card":
        generated = {
            "card_count": response.card_count,
            "card_ids": response.card_ids,
            "rejected_count": len(response.rejected),
        }
    elif job.content_type == "micro_quiz":
        generated = {
            "micro_content_id": response.micro_content_id,
            "quiz_count": response.quiz_count,
            "question_ids": response.question_ids,
            "rejected_count": len(response.rejected),
        }
    else:
        generated = {
            "quiz_count": response.quiz_count,
            "question_ids": response.question_ids,
            "rejected_count": len(response.rejected),
        }
    return {
        "request_id": job.request_id,
        "course_id": response.course_id,
        "lo_id": getattr(response, "lo_id", None),
        **generated,
    }


def _run_job_sync(use_case, request):
    """Chạy use case (đồng bộ, trong thread); Err thành exception cho chỗ gọi."""
    result = use_case.execute(request)
    if result.is_err():
        error = result.error
        raise error if isinstance(error, BaseException) else RuntimeError(str(error))
    return result.unwrap()


async def _process_job(container, bull_job, job_token):
    del job_token
    try:
        job = parse_job(bull_job.data)
    except (KeyError, TypeError, ValueError) as exc:
        try:
            await _fail_unparsed_request(container, bull_job, exc)
        except Exception as publish_exc:
            # FAILED chưa tới core-api: lỗi thường (không Unrecoverable) để
            # BullMQ retry job và lần sau gửi lại, thay vì để dòng treo QUEUED.
            logger.error(
                "content_generation_worker.job.fail_publish_failed",
                error=str(exc),
                publish_error=str(publish_exc),
            )
            raise RuntimeError(
                f"invalid content-generation job ({exc}); "
                f"publishing the FAILED result failed: {publish_exc}"
            ) from publish_exc
        raise UnrecoverableError(f"invalid content-generation job: {exc}") from exc

    reporter = _reporter(container, bull_job, job.request_id, job.course_id, job.content_type)
    if reporter is None:
        logger.error(
            "content_generation_worker.job.no_result_id",
            request_id=job.request_id,
            reason="event_id, job id và request_id đều không dùng làm result_id được",
        )
        raise UnrecoverableError(f"cannot derive a result_id for request {job.request_id!r}")

    try:
        # Dựng use case có thể đọc llm_model_bindings (đồng bộ): cũng chạy
        # ngoài event loop như trước.
        use_case, request = await asyncio.to_thread(_plan, container, job)
    except (TypeError, ValueError) as exc:
        await reporter.failed(exc)
        raise UnrecoverableError(f"invalid content-generation request: {exc}") from exc

    await reporter.started()

    try:
        # Use case gọi LLM, MCP và psycopg đồng bộ: chạy ngoài event loop để
        # không chặn heartbeat gia hạn lock BullMQ và HTTP healthcheck.
        response = await asyncio.to_thread(_run_job_sync, use_case, request)
    except Exception as exc:
        # Chỉ báo FAILED ở lần thử cuối: FAILED là trạng thái kết thúc, core-api
        # không cho lần retry sau kéo request về RUNNING/SUCCEEDED. Giữa các lần
        # thử, request vẫn RUNNING.
        if is_final_attempt(bull_job) or isinstance(exc, UnrecoverableError):
            await reporter.failed(exc)
        else:
            logger.warning(
                "content_generation_worker.job.will_retry",
                request_id=job.request_id,
                attempts_made=getattr(bull_job, "attemptsMade", None),
                error=str(exc),
            )
        raise

    await reporter.succeeded(response)
    summary = _summary(job, response)
    logger.info(
        "content_generation_worker.job.completed",
        request_id=job.request_id,
        course_id=job.course_id,
        content_type=job.content_type,
        lo_id=summary["lo_id"],
        rejected=summary.get("rejected_count"),
    )
    return summary


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
