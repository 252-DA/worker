"""
ENRICHMENT_CONTENT_RESULT: học liệu nháp (lesson card + câu quiz) của một lượt
enrichment tài liệu → core-api
(contracts/integration-events/v1/enrichment-content-result.json).

core-api thay toàn bộ học liệu nháp của tài liệu bằng bộ này (danh sách rỗng =
xoá nháp). Một kết quả cho mỗi lượt xử lý tài liệu: ``result_id`` gắn với
``processing_generation`` nên job retry gửi lại cùng id và BullMQ bỏ bản trùng.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from document_chunk.domain.ports.metadata_store import StoredLessonCard, StoredQuizItem

from worker.adapters.result_publisher import ENRICHMENT_CONTENT_RESULT, make_result_id

# lesson_cards.title là VARCHAR(255); LLM thỉnh thoảng trả tiêu đề dài hơn.
MAX_CARD_TITLE = 255


class EnrichmentContentReporter:
    def __init__(self, publisher: Any) -> None:
        self._publisher = publisher

    def report(
        self,
        *,
        document_id: str,
        processing_generation: int,
        model_id: str,
        lesson_cards: Sequence[StoredLessonCard],
        quiz_items: Sequence[StoredQuizItem],
    ) -> str:
        """Gửi từ use case chạy trong thread; lỗi (Redis, sai hợp đồng) nổi lên cho người gọi."""
        return self._publisher.publish_sync(
            ENRICHMENT_CONTENT_RESULT,
            enrichment_content_payload(
                document_id=document_id,
                processing_generation=processing_generation,
                model_id=model_id,
                lesson_cards=lesson_cards,
                quiz_items=quiz_items,
            ),
            result_id=make_result_id(
                ENRICHMENT_CONTENT_RESULT,
                str(document_id),
                f"g{processing_generation}",
            ),
        )


def enrichment_content_payload(
    *,
    document_id: str,
    processing_generation: int,
    model_id: str,
    lesson_cards: Sequence[StoredLessonCard],
    quiz_items: Sequence[StoredQuizItem],
) -> dict[str, Any]:
    return {
        "document_id": str(document_id),
        "processing_generation": processing_generation,
        "model_id": model_id,
        "lesson_cards": [
            {
                "card_id": card.card_id,
                # list_chunks trả chunk_id kiểu uuid.UUID: JSON chỉ nhận str.
                "primary_chunk_id": str(card.primary_chunk_id),
                "source_chunk_ids": [
                    str(chunk_id)
                    for chunk_id in (card.source_chunk_ids or (card.primary_chunk_id,))
                ],
                "heading_path": list(card.heading_path),
                "title": card.title[:MAX_CARD_TITLE],
                "bullets": list(card.bullets),
                "key_insight": card.key_insight,
                "card_index": card.card_index,
            }
            for card in lesson_cards
        ],
        "quiz_items": [
            {
                "quiz_id": item.question_id,
                "primary_chunk_id": str(item.primary_chunk_id),
                "source_chunk_ids": [
                    str(chunk_id)
                    for chunk_id in (item.source_chunk_ids or (item.primary_chunk_id,))
                ],
                "question": item.question,
                "choices": list(item.choices),
                "correct_index": item.correct_index,
                "explanation": item.explanation,
            }
            for item in quiz_items
        ],
    }


__all__ = ["MAX_CARD_TITLE", "EnrichmentContentReporter", "enrichment_content_payload"]
