"""
Đọc lệnh CONTENT_GENERATION_REQUESTED (contracts/integration-events/v1/
content-generation-requested.json) thành request của từng use case.

Lệnh sai dạng → ``ValueError``/``KeyError``/``TypeError``: runner báo FAILED cho
yêu cầu (nếu còn lấy lại được ``request_id``) rồi bỏ job, không thử lại.
"""
from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True)
class ContentGenerationJob:
    request_id: str
    course_id: str
    content_type: str
    scope: dict[str, Any]


def request_id_from_job_data(data: Any) -> str | None:
    """
    Lấy lại ``request_id`` từ một job mà ``parse_job`` không đọc được.

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


def payload_text(data: Any, key: str) -> str | None:
    payload = data.get("payload") if isinstance(data, dict) else None
    value = payload.get(key) if isinstance(payload, dict) else None
    return value.strip() if isinstance(value, str) and value.strip() else None


def parse_job(data: Any) -> ContentGenerationJob:
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


def quiz_request(job: ContentGenerationJob) -> GenerateCurriculumQuizRequest:
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


def card_request(job: ContentGenerationJob) -> GenerateLessonCardsRequest:
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


def relations_request(job: ContentGenerationJob) -> ExtractRelationsRequest:
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


def micro_outline_request(job: ContentGenerationJob) -> GenerateMicroOutlineRequest:
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


def video_script_request(job: ContentGenerationJob) -> GenerateVideoScriptRequest:
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


def micro_quiz_request(job: ContentGenerationJob) -> GenerateMicroQuizRequest:
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


__all__ = [
    "ContentGenerationJob",
    "card_request",
    "micro_outline_request",
    "micro_quiz_request",
    "parse_job",
    "payload_text",
    "quiz_request",
    "relations_request",
    "request_id_from_job_data",
    "video_script_request",
]
