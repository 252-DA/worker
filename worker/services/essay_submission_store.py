"""Đọc/ghi `essay_submissions` — bài luận nộp qua LTI Asset Processor.

core-api tạo dòng, tải file về MinIO (status QUEUED) rồi phát
ESSAY_GRADING_REQUESTED. Worker chỉ chuyển QUEUED → GRADED hoặc FAILED; gửi
báo cáo về Canvas lại là việc của core-api (essay-report.publisher.ts).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from document_chunk.infrastructure.config import SqlConfig

QUEUED = "QUEUED"

_GET_SQL = """
    SELECT s.essay_submission_id::text, s.status, s.assets, s.user_id::text,
           p.course_id::text, p.title, p.instructions, p.rubric, p.max_score
      FROM essay_submissions s
      JOIN essay_processors p ON p.processor_id = s.processor_id
     WHERE s.essay_submission_id = %s::uuid;
"""

_GRADED_SQL = """
    UPDATE essay_submissions
       SET status = 'GRADED', result = %s, error = NULL, error_code = NULL,
           attempts = 0, next_attempt_at = NULL, updated_at = now()
     WHERE essay_submission_id = %s::uuid AND status = 'QUEUED';
"""

_FAILED_SQL = """
    UPDATE essay_submissions
       SET status = 'FAILED', error_code = %s, error = %s, attempts = 0,
           next_attempt_at = NULL, reported_at = NULL, updated_at = now()
     WHERE essay_submission_id = %s::uuid AND status = 'QUEUED';
"""


@dataclass(frozen=True)
class EssayJob:
    essay_submission_id: str
    status: str
    assets: list[dict[str, Any]]
    user_id: str | None
    course_id: str
    title: str
    instructions: str
    rubric: list[dict[str, Any]]
    max_score: float


class EssaySubmissionStore:
    def __init__(self, sql: SqlConfig) -> None:
        self._sql = sql

    def _connect(self):
        return psycopg.connect(self._sql.dsn, connect_timeout=self._sql.connect_timeout_seconds)

    def get(self, essay_submission_id: str) -> EssayJob | None:
        with self._connect() as conn:
            row = conn.execute(_GET_SQL, (essay_submission_id,)).fetchone()
        if row is None:
            return None
        sid, status, assets, user_id, course_id, title, instructions, rubric, max_score = row
        return EssayJob(
            essay_submission_id=sid,
            status=status,
            assets=list(assets or []),
            user_id=user_id,
            course_id=course_id,
            title=title,
            instructions=instructions or "",
            rubric=list(rubric or []),
            max_score=float(max_score),
        )

    def save_result(self, essay_submission_id: str, result: dict[str, Any]) -> bool:
        """False khi bài không còn QUEUED (đã chấm ở lượt trước)."""
        with self._connect() as conn:
            cur = conn.execute(_GRADED_SQL, (Jsonb(result), essay_submission_id))
            return cur.rowcount == 1

    def mark_failed(self, essay_submission_id: str, error_code: str, error: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(_FAILED_SQL, (error_code, error[:2000], essay_submission_id))
            return cur.rowcount == 1


__all__ = ["EssayJob", "EssaySubmissionStore", "QUEUED"]
