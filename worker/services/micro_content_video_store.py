from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import psycopg
from document_chunk.infrastructure.config import SqlConfig
from document_chunk.shared.result import Err, Ok, Result


@dataclass(frozen=True)
class MicroContentRecord:
    micro_content_id: str
    course_id: str
    title: str
    ai_script: Any
    video_status: str


class MicroContentVideoStore:
    def __init__(self, sql: SqlConfig) -> None:
        self._sql = sql

    def _connect(self):
        return psycopg.connect(self._sql.dsn, connect_timeout=self._sql.connect_timeout_seconds)

    def get_micro_content(self, micro_content_id: str) -> Result[MicroContentRecord, Exception]:
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT micro_content_id::text, course_id::text, title, ai_script, video_status
                          FROM micro_contents
                         WHERE micro_content_id = %s::uuid
                           AND deleted_at IS NULL;
                        """,
                        (micro_content_id,),
                    )
                    row = cur.fetchone()
            if not row:
                return Err(ValueError(f"Micro content '{micro_content_id}' not found"))
            return Ok(
                MicroContentRecord(
                    micro_content_id=str(row[0]),
                    course_id=str(row[1]),
                    title=str(row[2]),
                    ai_script=row[3],
                    video_status=str(row[4]),
                )
            )
        except Exception as exc:
            return Err(exc)

    def update_script(self, micro_content_id: str, script: dict) -> Result[bool, Exception]:
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE micro_contents
                           SET ai_script = %s::jsonb,
                               updated_at = NOW()
                         WHERE micro_content_id = %s::uuid;
                        """,
                        (json.dumps(script), micro_content_id),
                    )
                conn.commit()
            return Ok(True)
        except Exception as exc:
            return Err(exc)

    def update_video_ready(
        self,
        micro_content_id: str,
        object_key: str,
        duration_seconds: float,
        size_bytes: int,
    ) -> Result[bool, Exception]:
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE micro_contents
                           SET video_status = 'READY',
                               video_source = 'AI',
                               video_object_key = %s,
                               video_duration_seconds = %s,
                               video_size_bytes = %s,
                               video_mime_type = 'video/mp4',
                               video_error = NULL,
                               updated_at = NOW()
                         WHERE micro_content_id = %s::uuid;
                        """,
                        (object_key, int(duration_seconds), size_bytes, micro_content_id),
                    )
                conn.commit()
            return Ok(True)
        except Exception as exc:
            return Err(exc)

    def update_video_failed(self, micro_content_id: str, error: str) -> Result[bool, Exception]:
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE micro_contents
                           SET video_status = 'FAILED',
                               video_error = %s,
                               updated_at = NOW()
                         WHERE micro_content_id = %s::uuid;
                        """,
                        (error[:1000], micro_content_id),
                    )
                conn.commit()
            return Ok(True)
        except Exception as exc:
            return Err(exc)
