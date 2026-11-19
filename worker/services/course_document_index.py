"""
Đọc (chỉ SELECT) các tài liệu đã index của một học phần, để map lại chunk → LO
sau khi core-api ghi đề cương. Bảng đọc: ``documents``, ``chunks``.

Thay ``CurriculumImportStore``: worker không còn đọc/ghi ``curriculum_imports``
— core-api đưa đủ dữ liệu trong job và nhận kết quả qua ``worker_results``.
"""
from __future__ import annotations

import psycopg

from document_chunk.infrastructure.config import SqlConfig

# Tài liệu có chunk mới map được sang LO.
_INDEXED_DOCUMENTS_SQL = """
    SELECT d.document_id::text
      FROM documents d
     WHERE d.course_id = %s::uuid
       AND d.deleted_at IS NULL
       AND EXISTS (SELECT 1 FROM chunks ch
                    WHERE ch.document_id = d.document_id AND ch.deleted_at IS NULL)
     ORDER BY d.created_at;
"""


class CourseDocumentIndex:
    def __init__(self, sql: SqlConfig) -> None:
        self._sql = sql

    def indexed_document_ids(self, course_id: str) -> list[str]:
        with psycopg.connect(
            self._sql.dsn, connect_timeout=self._sql.connect_timeout_seconds
        ) as conn, conn.cursor() as cur:
            cur.execute(_INDEXED_DOCUMENTS_SQL, (course_id,))
            return [row[0] for row in cur.fetchall()]
