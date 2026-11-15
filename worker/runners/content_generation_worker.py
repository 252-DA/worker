"""
BullMQ worker for grounded, LO-scoped quiz and lesson-card generation requests.

Worker tính, core-api ghi: mỗi bước của vòng đời request (bắt đầu chạy, xong
kèm câu/card đã kiểm chứng, thất bại) được gửi về core-api thành một kết quả
CONTENT_GENERATION_RESULT trên hàng đợi ``worker_results``
(contracts/integration-events/v1/content-generation-result.json). core-api là
nơi duy nhất ghi ``content_generation_requests``, ``lessons``, ``quiz_items``,
``lesson_cards`` (src/content-generation/content-generation-result.handler.ts).
"""

import asyncio
import re
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
from worker.services.result_publisher import CONTENT_GENERATION_RESULT, make_result_id
from worker.use_cases.extract_relations import ExtractRelationsRequest
from worker.use_cases.generate_curriculum_quiz import GenerateCurriculumQuizRequest
from worker.use_cases.generate_lesson_cards import (
    DEFAULT_CARD_COUNT,
    MAX_CARD_COUNT,
    GenerateLessonCardsRequest,
)
from worker.use_cases.generate_micro_outline import GenerateMicroOutlineRequest
from worker.use_cases.generate_micro_quiz import (
    DEFAULT_MICRO_QUIZ_COUNT,
    DEFAULT_MICRO_QUIZ_TYPES,
    GenerateMicroQuizRequest,
)
from worker.use_cases.generate_video_script import (
    DEFAULT_VOICE,
    GenerateVideoScriptRequest,
)

logger = get_logger(__name__)


# Trạng thái đích của content_generation_requests mà mỗi kết quả mang theo; cũng
# là phần cuối của result_id, nên mỗi bước có đúng một id cho một lệnh: job gốc
# retry gửi lại cùng id và BullMQ bỏ bản trùng.
STATUS_RUNNING = "RUNNING"
STATUS_SUCCEEDED = "SUCCEEDED"
STATUS_FAILED = "FAILED"

_CONTENT_TYPES = frozenset(
    {"quiz", "card", "relations", "micro_outline", "video_script", "micro_quiz"}
)
# last_error là TEXT; chặn để một thông báo lỗi dài không đi nguyên vào payload.
_MAX_ERROR_CHARS = 4000
# Khớp _ID_PART của result_publisher: một phần của result_id.
_RESULT_ID_PART = re.compile(r"[A-Za-z0-9_-]+")
_MAX_SOURCE_EVENT_ID = 200


@dataclass(frozen=True)
class ContentGenerationJob:
    request_id: str
    course_id: str
    content_type: str
    scope: dict[str, Any]


@dataclass(frozen=True)
class ResultSource:
    """Lệnh đã sinh ra việc này: gốc của result_id và source_event_id."""

    source_id: str
    source_event_id: str | None


def _result_source(bull_job: Any, request_id: str | None) -> ResultSource | None:
    """
    ``event_id`` outbox của lệnh là gốc tốt nhất (mỗi lần yêu cầu là một sự
    kiện mới); job id (outbox đặt = event_id) rồi request_id là dự phòng.
    """
    data = getattr(bull_job, "data", None)
    event_id = data.get("event_id") if isinstance(data, dict) else None
    if not isinstance(event_id, str) or not event_id.strip():
        event_id = None
    candidates = (event_id, getattr(bull_job, "id", None), request_id)
    source_id = next(
        (c for c in candidates if isinstance(c, str) and _RESULT_ID_PART.fullmatch(c)),
        None,
    )
    if source_id is None:
        return None
    if event_id is not None and len(event_id) > _MAX_SOURCE_EVENT_ID:
        event_id = None
    return ResultSource(source_id=source_id, source_event_id=event_id)


def _is_final_attempt(bull_job: Any) -> bool:
    """``attemptsMade`` đếm số lần đã thất bại, nên lần đang chạy là lần thứ +1."""
    opts = getattr(bull_job, "opts", None) or {}
    attempts = int(opts.get("attempts") or 1)
    return int(getattr(bull_job, "attemptsMade", 0) or 0) + 1 >= attempts


def _error_text(error: object) -> str:
    text = str(error).strip() or type(error).__name__
    return text if len(text) <= _MAX_ERROR_CHARS else text[: _MAX_ERROR_CHARS - 1] + "…"


def _base_payload(
    request_id: str, course_id: str | None, content_type: str | None, status: str
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "course_id": course_id,
        # Loại lạ (job hỏng) gửi null: core-api chỉ đối chiếu khi có giá trị.
        "content_type": content_type if content_type in _CONTENT_TYPES else None,
        "status": status,
    }


def _started_payload(job: ContentGenerationJob) -> dict[str, Any]:
    return _base_payload(job.request_id, job.course_id, job.content_type, STATUS_RUNNING)


def _failed_payload(
    request_id: str, course_id: str | None, content_type: str | None, error: object
) -> dict[str, Any]:
    return {
        **_base_payload(request_id, course_id, content_type, STATUS_FAILED),
        "error": _error_text(error),
    }


def _succeeded_payload(job: ContentGenerationJob, response) -> dict[str, Any]:
    payload = _base_payload(job.request_id, job.course_id, job.content_type, STATUS_SUCCEEDED)
    if job.content_type == "relations":
        # knowledge_relations vẫn do worker ghi (chưa thuộc đợt này); core-api
        # chỉ cập nhật trạng thái request.
        return {**payload, "generated_count": response.relations_written}
    if job.content_type == "micro_outline":
        return {
            **payload,
            "document_id": response.document_id,
            "generated_count": response.section_count,
            "sections": [sec.to_payload() for sec in response.sections],
        }
    rejected = [r.to_payload() for r in response.rejected]
    if job.content_type == "video_script":
        return {
            **payload,
            "micro_content_id": response.micro_content_id,
            "generated_count": response.scene_count,
            "script": response.script,
            "rejected": rejected,
        }
    if job.content_type == "micro_quiz":
        return {
            **payload,
            "micro_content_id": response.micro_content_id,
            "lo_id": response.lo_id,
            "generated_count": response.quiz_count,
            "quiz_items": [item.to_payload() for item in response.items],
            "rejected": rejected,
        }
    if job.content_type == "card":
        return {
            **payload,
            "lo_id": response.lo_id,
            "generated_count": response.card_count,
            "lesson_cards": [card.to_payload() for card in response.cards],
            "rejected": rejected,
        }
    return {
        **payload,
        "lo_id": response.lo_id,
        "generated_count": response.quiz_count,
        "quiz_items": [item.to_payload() for item in response.items],
        "rejected": rejected,
    }


async def _publish(container, source: ResultSource, payload: dict[str, Any]) -> str:
    """Chỉ trả về khi Redis đã nhận; lỗi nổi lên để job gốc fail và BullMQ retry."""
    return await container.result_publisher.publish(
        CONTENT_GENERATION_RESULT,
        payload,
        result_id=make_result_id(CONTENT_GENERATION_RESULT, source.source_id, payload["status"]),
        source_event_id=source.source_event_id,
    )


def _request_id_from_job_data(data: Any) -> str | None:
    """
    Lấy lại ``request_id`` từ một job mà ``_parse_job`` không đọc được.

    Không parse được thì không có ``ContentGenerationJob``, nhưng dòng
    ``content_generation_requests`` vẫn đang nằm ở QUEUED. FE chặn mọi yêu cầu
    mới của khoá học khi còn một dòng QUEUED/RUNNING, mà không có endpoint huỷ
    và không job nào dọn dòng treo — nên bỏ qua ở đây là khoá cứng việc sinh
    quiz của cả khoá học cho tới khi ai đó sửa tay trong DB.

    Outbox đặt ``aggregate_id = request_id`` (`outbox.service.ts`), nên vẫn lấy
    lại được id kể cả khi chính ``payload`` mới là phần hỏng.
    """
    if not isinstance(data, dict):
        return None
    payload = data.get("payload")
    if isinstance(payload, dict):
        request_id = payload.get("request_id")
        if isinstance(request_id, str) and request_id.strip():
            return request_id.strip()
    aggregate_id = data.get("aggregate_id")
    if isinstance(aggregate_id, str) and aggregate_id.strip():
        return aggregate_id.strip()
    return None


def _payload_text(data: Any, key: str) -> str | None:
    payload = data.get("payload") if isinstance(data, dict) else None
    value = payload.get(key) if isinstance(payload, dict) else None
    return value.strip() if isinstance(value, str) and value.strip() else None


async def _fail_unparsed_request(container, bull_job: Any, exc: Exception) -> None:
    """
    Gửi kết quả FAILED cho một job hỏng để dòng request không treo ở QUEUED.

    Lỗi gửi được ném ra cho chỗ gọi: job phải retry để còn gửi lại được.
    """
    data = getattr(bull_job, "data", None)
    request_id = _request_id_from_job_data(data)
    source = _result_source(bull_job, request_id) if request_id else None
    if request_id is None or source is None:
        logger.error(
            "content_generation_worker.job.unidentifiable",
            error=str(exc),
            reason="không lấy được request_id; dòng QUEUED phải dọn tay",
        )
        return
    await _publish(
        container,
        source,
        _failed_payload(
            request_id,
            _payload_text(data, "course_id"),
            _payload_text(data, "type"),
            f"invalid content-generation job: {exc}",
        ),
    )


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


def _target_from_scope(scope: dict[str, Any]) -> tuple[str, str]:
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
    return target_kind, target_code


def _quiz_request(job: ContentGenerationJob) -> GenerateCurriculumQuizRequest:
    if job.content_type != "quiz":
        raise ValueError(f"expected type='quiz', got {job.content_type!r}")

    scope = job.scope
    target_kind, target_code = _target_from_scope(scope)
    return GenerateCurriculumQuizRequest(
        course_id=job.course_id,
        target_kind=target_kind,
        target_code=target_code,
        style=str(scope.get("style") or "quiz"),
        bloom_level=scope.get("bloom_level") or scope.get("bloomLevel") or scope.get("bloom"),
        source_document_ids=scope.get("source_document_ids"),
        count=int(scope.get("count") or 5),
        question_type=str(scope.get("question_type") or "MCQ_SINGLE"),
    )


def _card_request(job: ContentGenerationJob) -> GenerateLessonCardsRequest:
    if job.content_type != "card":
        raise ValueError(f"expected type='card', got {job.content_type!r}")

    scope = job.scope
    target_kind, target_code = _target_from_scope(scope)
    count = int(scope.get("count") or DEFAULT_CARD_COUNT)
    if not 1 <= count <= MAX_CARD_COUNT:
        raise ValueError(f"card count must be between 1 and {MAX_CARD_COUNT}, got {count}")
    return GenerateLessonCardsRequest(
        course_id=job.course_id,
        target_kind=target_kind,
        target_code=target_code,
        count=count,
        source_document_ids=scope.get("source_document_ids"),
    )


def _relations_request(job: ContentGenerationJob) -> ExtractRelationsRequest:
    if job.content_type != "relations":
        raise ValueError(f"expected type='relations', got {job.content_type!r}")
    raw = job.scope.get("source_document_ids")
    if not isinstance(raw, list) or not raw or not all(isinstance(v, str) and v for v in raw):
        raise ValueError("relations scope must include source_document_ids")
    return ExtractRelationsRequest(course_id=job.course_id, document_ids=list(raw))


def _scope_text(scope: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = scope.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _scope_int(scope: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = scope.get(key)
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and value.strip():
            return int(value.strip())
    return None


def _micro_outline_request(job: ContentGenerationJob) -> GenerateMicroOutlineRequest:
    if job.content_type != "micro_outline":
        raise ValueError(f"expected type='micro_outline', got {job.content_type!r}")
    scope = job.scope
    document_id = _scope_text(scope, "document_id", "documentId", "source_document_id", "sourceDocumentId")
    if not document_id:
        raise ValueError("micro_outline scope must include document_id")
    pages = scope.get("pages") if isinstance(scope.get("pages"), list) else None
    raw_los = scope.get("candidate_los") or scope.get("candidateLos")
    candidate_los = raw_los if isinstance(raw_los, list) else None
    return GenerateMicroOutlineRequest(
        course_id=job.course_id,
        document_id=document_id,
        document_title=_scope_text(scope, "document_title", "documentTitle") or "",
        chapter_code=_scope_text(scope, "chapter_code", "chapterCode"),
        total_pages=_scope_int(scope, "total_pages", "totalPages"),
        pages=pages,
        candidate_los=candidate_los,
    )


def _video_script_request(job: ContentGenerationJob) -> GenerateVideoScriptRequest:
    if job.content_type != "video_script":
        raise ValueError(f"expected type='video_script', got {job.content_type!r}")
    scope = job.scope
    micro_content_id = _scope_text(scope, "micro_content_id", "microContentId")
    if not micro_content_id:
        raise ValueError("video_script scope must include micro_content_id")
    document_id = _scope_text(
        scope, "document_id", "documentId", "source_document_id", "sourceDocumentId"
    )
    page_from = _scope_int(scope, "page_from", "pageFrom", "source_page_from", "sourcePageFrom") or 1
    page_to = _scope_int(scope, "page_to", "pageTo", "source_page_to", "sourcePageTo") or page_from
    page_from = max(1, page_from)
    page_to = max(page_from, page_to)
    raw_pages = scope.get("pages")
    pages = tuple(raw_pages) if isinstance(raw_pages, list) else None
    if not document_id and not pages:
        raise ValueError("video_script scope must include document_id or pages")
    raw_neighbors = scope.get("neighbor_topics") or scope.get("neighborTopics")
    neighbor_topics = (
        tuple(str(item).strip() for item in raw_neighbors if isinstance(item, str) and item.strip())
        if isinstance(raw_neighbors, list)
        else ()
    )
    return GenerateVideoScriptRequest(
        course_id=job.course_id,
        micro_content_id=micro_content_id,
        document_id=document_id,
        page_from=page_from,
        page_to=page_to,
        topic_title=_scope_text(scope, "topic_title", "topicTitle", "title") or "",
        topic_summary=_scope_text(scope, "topic_summary", "topicSummary", "summary"),
        lo_id=_scope_text(scope, "lo_id", "loId"),
        lo_code=_scope_text(scope, "lo_code", "loCode"),
        lo_statement_vi=_scope_text(scope, "lo_statement_vi", "loStatementVi"),
        lo_statement_en=_scope_text(scope, "lo_statement_en", "loStatementEn"),
        voice=_scope_text(scope, "voice") or DEFAULT_VOICE,
        neighbor_topics=neighbor_topics,
        low_text=bool(scope.get("low_text") or scope.get("lowText")),
        instructor_prompt=_scope_text(scope, "instructor_prompt", "instructorPrompt", "prompt"),
        pages=pages,
    )


def _micro_quiz_request(job: ContentGenerationJob) -> GenerateMicroQuizRequest:
    if job.content_type != "micro_quiz":
        raise ValueError(f"expected type='micro_quiz', got {job.content_type!r}")
    scope = job.scope
    micro_content_id = _scope_text(scope, "micro_content_id", "microContentId")
    if not micro_content_id:
        raise ValueError("micro_quiz scope must include micro_content_id")
    document_id = _scope_text(
        scope, "document_id", "documentId", "source_document_id", "sourceDocumentId"
    )
    page_from = _scope_int(scope, "page_from", "pageFrom", "source_page_from", "sourcePageFrom") or 1
    page_to = _scope_int(scope, "page_to", "pageTo", "source_page_to", "sourcePageTo") or page_from
    page_from = max(1, page_from)
    page_to = max(page_from, page_to)
    raw_types = scope.get("question_types") or scope.get("questionTypes")
    question_types = (
        tuple(str(t).strip() for t in raw_types if isinstance(t, str) and t.strip())
        if isinstance(raw_types, list)
        else DEFAULT_MICRO_QUIZ_TYPES
    ) or DEFAULT_MICRO_QUIZ_TYPES
    raw_chunks = scope.get("source_chunks") or scope.get("sourceChunks")
    source_chunks = tuple(raw_chunks) if isinstance(raw_chunks, list) else None
    script = scope.get("script") if isinstance(scope.get("script"), dict) else None
    if not document_id and not source_chunks and not script:
        raise ValueError("micro_quiz scope must include document_id, source_chunks, or script")
    count = _scope_int(scope, "count") or DEFAULT_MICRO_QUIZ_COUNT
    count = max(1, min(10, count))
    return GenerateMicroQuizRequest(
        course_id=job.course_id,
        micro_content_id=micro_content_id,
        topic_title=_scope_text(scope, "topic_title", "topicTitle", "title") or "",
        lo_id=_scope_text(scope, "lo_id", "loId"),
        lo_code=_scope_text(scope, "lo_code", "loCode"),
        lo_statement_vi=_scope_text(scope, "lo_statement_vi", "loStatementVi"),
        lo_statement_en=_scope_text(scope, "lo_statement_en", "loStatementEn"),
        document_id=document_id,
        page_from=page_from,
        page_to=page_to,
        script=script,
        count=count,
        question_types=question_types,
        source_chunks=source_chunks,
    )


def _plan(container, job: ContentGenerationJob):
    """(use case, request) cho loại nội dung của job; ValueError nếu không hỗ trợ."""
    if job.content_type == "quiz":
        return container.generate_curriculum_quiz_use_case, _quiz_request(job)
    if job.content_type == "card":
        return container.generate_lesson_cards_use_case, _card_request(job)
    if job.content_type == "relations":
        return container.extract_relations_use_case, _relations_request(job)
    if job.content_type == "micro_outline":
        return container.generate_micro_outline_use_case, _micro_outline_request(job)
    if job.content_type == "video_script":
        return container.generate_video_script_use_case, _video_script_request(job)
    if job.content_type == "micro_quiz":
        return container.generate_micro_quiz_use_case, _micro_quiz_request(job)
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
        job = _parse_job(bull_job.data)
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

    source = _result_source(bull_job, job.request_id)
    if source is None:
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
        await _publish(
            container, source,
            _failed_payload(job.request_id, job.course_id, job.content_type, exc),
        )
        raise UnrecoverableError(f"invalid content-generation request: {exc}") from exc

    await _publish(container, source, _started_payload(job))

    try:
        # Use case gọi LLM, MCP và psycopg đồng bộ: chạy ngoài event loop để
        # không chặn heartbeat gia hạn lock BullMQ và HTTP healthcheck.
        response = await asyncio.to_thread(_run_job_sync, use_case, request)
    except Exception as exc:
        # Chỉ báo FAILED ở lần thử cuối: FAILED là trạng thái kết thúc, core-api
        # không cho lần retry sau kéo request về RUNNING/SUCCEEDED. Giữa các lần
        # thử, request vẫn RUNNING.
        if _is_final_attempt(bull_job) or isinstance(exc, UnrecoverableError):
            await _publish(
                container, source,
                _failed_payload(job.request_id, job.course_id, job.content_type, exc),
            )
        else:
            logger.warning(
                "content_generation_worker.job.will_retry",
                request_id=job.request_id,
                attempts_made=getattr(bull_job, "attemptsMade", None),
                error=str(exc),
            )
        raise

    await _publish(container, source, _succeeded_payload(job, response))
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
