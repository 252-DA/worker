"""Đọc bảng `llm_model_bindings` — admin chọn model cho từng task trên web.

Chỉ đọc: core-api là bên duy nhất ghi bảng này (kiểm tra tên profile, ghi
audit). Lỗi nào cũng ném ra để ``OverrideConfigSource`` giữ lựa chọn cũ và
rơi về env, kể cả khi bảng chưa có vì migration chưa chạy.
"""
from __future__ import annotations

import psycopg

from document_chunk.infrastructure.config import SqlConfig

_BINDINGS_SQL = "SELECT task, profile FROM llm_model_bindings;"

# Đọc cấu hình không được giữ job lâu: DB chậm thì thà dùng lựa chọn cũ.
_CONNECT_TIMEOUT_SECONDS = 5


class ModelBindingStore:
    def __init__(self, sql: SqlConfig) -> None:
        self._sql = sql

    def fetch(self) -> dict[str, str]:
        with psycopg.connect(
            self._sql.dsn,
            connect_timeout=min(self._sql.connect_timeout_seconds, _CONNECT_TIMEOUT_SECONDS),
        ) as conn:
            rows = conn.execute(_BINDINGS_SQL).fetchall()
        return {str(task): str(profile) for task, profile in rows}


__all__ = ["ModelBindingStore"]
