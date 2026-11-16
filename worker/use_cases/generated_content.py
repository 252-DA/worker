"""
Học liệu do worker sinh, ở dạng core-api sẽ ghi (payload CONTENT_GENERATION_RESULT).

Worker chỉ tính; core-api (src/content-generation) là nơi duy nhất ghi
``lessons``, ``quiz_items``, ``lesson_cards``. Dạng từng phần tử khớp
contracts/integration-events/v1/content-generation-result.json và dạng cột mà
``persist_curriculum_quiz_items`` / ``persist_curriculum_lesson_cards`` từng ghi.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from worker.use_cases.verification import Verification

# Lý do loại là văn bản của judge; chỉ cần đủ để đọc lướt ở kết quả.
_MAX_REJECT_REASON_CHARS = 300


def as_uuid(value: str) -> str:
    """
    Giữ nguyên UUID hợp lệ, còn lại băm uuid5 — đúng như ``stable_uuid`` mà
    câu SQL cũ áp lên id câu/card và chunk, để id gửi đi trùng id đã lưu trước đây.
    """
    try:
        return str(uuid.UUID(str(value)))
    except (TypeError, ValueError):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, str(value)))


@dataclass(frozen=True)
class GeneratedQuizItem:
    question_id: str
    question_type: str
    question: str
    choices: tuple[str, ...]
    correct_index: int
    # Đáp án đúng đã quy về dạng cột quiz_items.correct_answer theo kiểu câu
    # (core-api quiz/question-types.ts); MCQ_SINGLE là chính phương án đúng.
    correct_answer: Any
    explanation: str | None
    bloom_level: int | None
    source_chunk_ids: tuple[str, ...]
    model_id: str | None
    judge_model: str | None
    verification: Verification
    difficulty: str = "medium"

    @property
    def quiz_id(self) -> str:
        return as_uuid(self.question_id)

    def to_payload(self) -> dict[str, Any]:
        chunk_ids = [as_uuid(c) for c in self.source_chunk_ids]
        return {
            "quiz_id": self.quiz_id,
            "type": self.question_type,
            "question": self.question,
            # Điền khuyết và tự luận không có phương án: cột options là NULL.
            "options": list(self.choices) if self.choices else None,
            "correct_answer": self.correct_answer,
            "explanation": self.explanation,
            "bloom_level": self.bloom_level,
            "source_chunk_ids": chunk_ids,
            "verification_status": self.verification.status,
            "verification_evidence": self.verification.evidence(
                writer_model=self.model_id,
                judge_model=self.judge_model,
                source_chunk_ids=chunk_ids,
                # Mức Bloom được YÊU CẦU khi sinh; không có bước nào đánh giá
                # câu thực sự đạt mức này.
                requested_bloom_level=self.bloom_level,
            ),
        }


@dataclass(frozen=True)
class GeneratedLessonCard:
    card_id: str
    title: str
    bullets: tuple[str, ...]
    key_insight: str
    example: str | None
    misconception: str | None
    check_question: str
    check_answer: str
    card_index: int
    primary_chunk_id: str
    source_chunk_ids: tuple[str, ...]
    model_id: str | None
    judge_model: str | None
    verification: Verification

    def to_payload(self) -> dict[str, Any]:
        chunk_ids = [as_uuid(c) for c in (self.source_chunk_ids or (self.primary_chunk_id,))]
        return {
            "card_id": as_uuid(self.card_id),
            "title": self.title,
            # Đúng object mà lesson_cards.content từng nhận.
            "content": {
                "key_insight": self.key_insight,
                "bullets": list(self.bullets),
                "example": self.example,
                "misconception": self.misconception,
                "check_question": self.check_question,
                "check_answer": self.check_answer,
                "model_id": self.model_id,
            },
            "source_chunk_ids": chunk_ids,
            "verification_status": self.verification.status,
            "verification_evidence": self.verification.evidence(
                writer_model=self.model_id,
                judge_model=self.judge_model,
                source_chunk_ids=chunk_ids,
            ),
        }


@dataclass(frozen=True)
class RejectedDraft:
    """Phần tử không được gửi đi (judge loại sau lượt cuối, hoặc đáp án sai dạng)."""

    reason: str
    regeneration_attempts: int = 0

    def to_payload(self) -> dict[str, Any]:
        reason = (self.reason or "").strip() or "rejected"
        if len(reason) > _MAX_REJECT_REASON_CHARS:
            reason = reason[: _MAX_REJECT_REASON_CHARS - 1] + "…"
        return {"reason": reason, "regeneration_attempts": self.regeneration_attempts}


__all__ = [
    "GeneratedLessonCard",
    "GeneratedQuizItem",
    "RejectedDraft",
    "as_uuid",
]
