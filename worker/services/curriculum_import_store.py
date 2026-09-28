"""Đọc/ghi bảng `curriculum_imports` — hàng core-api tạo khi nạp đề cương."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

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


@dataclass(frozen=True)
class CurriculumImport:
    import_id: str
    course_id: str
    status: str
    file_name: str
    storage_key: str


class CurriculumImportStore:
    def __init__(self, sql: SqlConfig) -> None:
        self._sql = sql

    def _connect(self):
        return psycopg.connect(self._sql.dsn, connect_timeout=self._sql.connect_timeout_seconds)

    def get(self, import_id: str) -> CurriculumImport | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT import_id::text, course_id::text, status, file_name, storage_key "
                "FROM curriculum_imports WHERE import_id = %s::uuid;",
                (import_id,),
            )
            row = cur.fetchone()
        return CurriculumImport(*row) if row else None

    def transition(
        self,
        import_id: str,
        *,
        to: str,
        expect: tuple[str, ...] | None = None,
        preview: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> bool:
        """Đổi trạng thái; với ``expect`` chỉ đổi khi đang ở một trong các trạng thái đó.

        Trả về False khi không có hàng nào khớp (ví dụ job chạy lại cho một lần
        nạp đã xong), để worker bỏ qua thay vì ghi đè.
        """
        sets = ["status = %s", "error = %s", "updated_at = NOW()"]
        params: list[Any] = [to, error]
        if preview is not None:
            sets.append("preview = %s")
            params.append(Jsonb(preview))
        if to == "APPLIED":
            sets.append("applied_at = NOW()")

        sql = f"UPDATE curriculum_imports SET {', '.join(sets)} WHERE import_id = %s::uuid"
        params.append(import_id)
        if expect:
            sql += " AND status = ANY(%s)"
            params.append(list(expect))

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.rowcount > 0

    def indexed_document_ids(self, course_id: str) -> list[str]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(_INDEXED_DOCUMENTS_SQL, (course_id,))
            return [row[0] for row in cur.fetchall()]
