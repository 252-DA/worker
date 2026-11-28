"""
CONTENT_GENERATION_RESULT: vòng đời một yêu cầu sinh nội dung (bắt đầu chạy, xong
kèm câu/card đã kiểm chứng, thất bại) → core-api
(contracts/integration-events/v1/content-generation-result.json). core-api là nơi
duy nhất ghi ``content_generation_requests``, ``quiz_items``, ``lesson_cards``
(src/modules/content-generation/handlers/content-generation-result.handler.ts).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from worker.adapters.result_publisher import CONTENT_GENERATION_RESULT, make_result_id

# Trạng thái đích của content_generation_requests mà mỗi kết quả mang theo; cũng
# là phần cuối của result_id, nên mỗi bước có đúng một id cho một lệnh: job gốc
# retry gửi lại cùng id và BullMQ bỏ bản trùng.
STATUS_RUNNING = "RUNNING"
STATUS_SUCCEEDED = "SUCCEEDED"
STATUS_FAILED = "FAILED"

CONTENT_TYPES = frozenset(
    {"quiz", "card", "relations", "micro_outline", "video_script", "micro_quiz"}
)
# last_error là TEXT; chặn để một thông báo lỗi dài không đi nguyên vào payload.
_MAX_ERROR_CHARS = 4000
# Khớp _ID_PART của result_publisher: một phần của result_id.
_RESULT_ID_PART = re.compile(r"[A-Za-z0-9_-]+")
_MAX_SOURCE_EVENT_ID = 200


@dataclass(frozen=True)
class ResultSource:
    """Lệnh đã sinh ra việc này: gốc của result_id và source_event_id."""

    source_id: str
    source_event_id: str | None


def result_source(event_id: Any, job_id: Any, request_id: str | None) -> ResultSource | None:
    """
    ``event_id`` outbox của lệnh là gốc tốt nhất (mỗi lần yêu cầu là một sự
    kiện mới); job id (outbox đặt = event_id) rồi request_id là dự phòng.
    """
    if not isinstance(event_id, str) or not event_id.strip():
        event_id = None
    candidates = (event_id, job_id, request_id)
    source_id = next(
        (c for c in candidates if isinstance(c, str) and _RESULT_ID_PART.fullmatch(c)),
        None,
    )
    if source_id is None:
        return None
    if event_id is not None and len(event_id) > _MAX_SOURCE_EVENT_ID:
        event_id = None
    return ResultSource(source_id=source_id, source_event_id=event_id)


@dataclass(frozen=True)
class ContentGenerationReporter:
    """Báo vòng đời của một yêu cầu; mỗi job một reporter."""

    publisher: Any
    source: ResultSource
    request_id: str
    course_id: str | None
    content_type: str | None

    async def started(self) -> str:
        return await self._publish(self._base(STATUS_RUNNING))

    async def failed(self, error: object) -> str:
        return await self._publish({**self._base(STATUS_FAILED), "error": _error_text(error)})

    async def succeeded(self, response: Any) -> str:
        return await self._publish(succeeded_payload(self._base(STATUS_SUCCEEDED), self.content_type, response))

    def _base(self, status: str) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "course_id": self.course_id,
            # Loại lạ (job hỏng) gửi null: core-api chỉ đối chiếu khi có giá trị.
            "content_type": self.content_type if self.content_type in CONTENT_TYPES else None,
            "status": status,
        }

    async def _publish(self, payload: dict[str, Any]) -> str:
        """Chỉ trả về khi Redis đã nhận; lỗi nổi lên để job gốc fail và BullMQ retry."""
        return await self.publisher.publish(
            CONTENT_GENERATION_RESULT,
            payload,
            result_id=make_result_id(CONTENT_GENERATION_RESULT, self.source.source_id, payload["status"]),
            source_event_id=self.source.source_event_id,
        )


def succeeded_payload(payload: dict[str, Any], content_type: str | None, response: Any) -> dict[str, Any]:
    if content_type == "relations":
        # knowledge_relations vẫn do worker ghi (chưa thuộc đợt này); core-api
        # chỉ cập nhật trạng thái request.
        return {**payload, "generated_count": response.relations_written}
    if content_type == "micro_outline":
        return {
            **payload,
            "document_id": response.document_id,
            "generated_count": response.section_count,
            "sections": [sec.to_payload() for sec in response.sections],
        }
    rejected = [r.to_payload() for r in response.rejected]
    if content_type == "video_script":
        return {
            **payload,
            "micro_content_id": response.micro_content_id,
            "generated_count": response.scene_count,
            "script": response.script,
            "rejected": rejected,
        }
    if content_type == "micro_quiz":
        return {
            **payload,
            "micro_content_id": response.micro_content_id,
            "lo_id": response.lo_id,
            "generated_count": response.quiz_count,
            "quiz_items": [item.to_payload() for item in response.items],
            "rejected": rejected,
        }
    if content_type == "card":
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


def _error_text(error: object) -> str:
    text = str(error).strip() or type(error).__name__
    return text if len(text) <= _MAX_ERROR_CHARS else text[: _MAX_ERROR_CHARS - 1] + "…"


__all__ = [
    "CONTENT_TYPES",
    "STATUS_FAILED",
    "STATUS_RUNNING",
    "STATUS_SUCCEEDED",
    "ContentGenerationReporter",
    "ResultSource",
    "result_source",
    "succeeded_payload",
]
