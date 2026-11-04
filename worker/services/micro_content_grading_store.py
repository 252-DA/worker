from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psycopg
from document_chunk.domain.ports.metadata_store import StoredAnswerForGrading
from document_chunk.infrastructure.config import SqlConfig
from document_chunk.shared.result import Err, Ok, Result


@dataclass(frozen=True)
class MicroContentAnswerJob:
    answer_id: str
    attempt_id: str
    question_type: str
    question: str
    correct_answer: Any
    chosen_answer: Any
    grading_status: str


class MicroContentGradingStore:
    def __init__(self, sql: SqlConfig) -> None:
        self._sql = sql

    def _connect(self):
        return psycopg.connect(self._sql.dsn, connect_timeout=self._sql.connect_timeout_seconds)

    def list_answers_for_grading(
        self, answer_ids: list[str]
    ) -> Result[list[StoredAnswerForGrading], Exception]:
        if not answer_ids:
            return Ok([])
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT a.answer_id::text, a.question_id::text, m.course_id::text,
                               a.grading_input->>'type', a.grading_input->>'question',
                               a.grading_input->'correct_answer', a.chosen_answer,
                               a.grading_status
                          FROM micro_content_answers a
                          JOIN micro_content_attempts att ON att.attempt_id = a.attempt_id
                          JOIN micro_contents m ON m.micro_content_id = att.micro_content_id
                         WHERE a.answer_id = ANY(%s::uuid[]);
                        """,
                        (answer_ids,),
                    )
                    rows = cur.fetchall()
            return Ok([
                StoredAnswerForGrading(
                    attempt_id=str(row[0]),  # answer_id dùng làm định danh để update
                    quiz_id=str(row[1]),
                    course_id=str(row[2]),
                    question_type=row[3],
                    question=row[4],
                    correct_answer=row[5],
                    chosen_answer=row[6],
                    grading_status=row[7],
                )
                for row in rows
            ])
        except Exception as exc:
            return Err(exc)

    def save_answer_grade(
        self,
        *,
        answer_id: str,
        score: int,
        is_correct: bool,
        feedback: str | None,
    ) -> Result[bool, Exception]:
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE micro_content_answers
                           SET score = %s, is_correct = %s, feedback = %s,
                               grading_status = 'GRADED'
                         WHERE answer_id = %s::uuid AND grading_status = 'PENDING';
                        """,
                        (score, is_correct, feedback, answer_id),
                    )
                    updated = cur.rowcount == 1
                conn.commit()
            return Ok(updated)
        except Exception as exc:
            return Err(exc)

    def mark_answer_needs_review(
        self,
        *,
        answer_id: str,
        feedback: str,
    ) -> Result[bool, Exception]:
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE micro_content_answers
                           SET feedback = %s, grading_status = 'NEEDS_REVIEW'
                         WHERE answer_id = %s::uuid;
                        """,
                        (feedback, answer_id),
                    )
                    updated = cur.rowcount == 1
                conn.commit()
            return Ok(updated)
        except Exception as exc:
            return Err(exc)

    def finalize_attempt(self, attempt_id: str) -> Result[bool, Exception]:
        """Tính lại tổng điểm và cập nhật trạng thái của attempt sau khi worker chấm xong."""
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    # Lấy tất cả câu trả lời của attempt này
                    cur.execute(
                        """
                        SELECT a.score, a.grading_status, m.passing_score
                          FROM micro_content_answers a
                          JOIN micro_content_attempts att ON att.attempt_id = a.attempt_id
                          JOIN micro_contents m ON m.micro_content_id = att.micro_content_id
                         WHERE a.attempt_id = %s::uuid;
                        """,
                        (attempt_id,),
                    )
                    rows = cur.fetchall()
                    if not rows:
                        return Ok(False)

                    scores = [int(r[0]) for r in rows]
                    statuses = [str(r[1]) for r in rows]
                    passing_score = int(rows[0][2])

                    total_score = round(sum(scores) / len(scores)) if scores else 0
                    passed = total_score >= passing_score

                    if "NEEDS_REVIEW" in statuses:
                        overall_status = "NEEDS_REVIEW"
                    elif any(s == "PENDING" for s in statuses):
                        overall_status = "PENDING"
                    else:
                        overall_status = "GRADED"

                    cur.execute(
                        """
                        UPDATE micro_content_attempts
                           SET score = %s, passed = %s, grading_status = %s
                         WHERE attempt_id = %s::uuid;
                        """,
                        (total_score, passed, overall_status, attempt_id),
                    )
                conn.commit()
            return Ok(True)
        except Exception as exc:
            return Err(exc)

