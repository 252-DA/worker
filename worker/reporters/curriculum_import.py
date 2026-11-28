"""
CURRICULUM_IMPORT_RESULT: bản xem trước + bản đề cương chuẩn hoá đầy đủ của một
lần nạp đề cương, hoặc FAILED → core-api
(contracts/integration-events/v1/curriculum-import-result.json).

``result_id`` theo ``import_id``: parse lại cho cùng kết quả và cùng id nên job
retry không sinh bản trùng. ``attempt`` chỉ dùng khi cùng một lần nạp có thể
nhận hai kết quả khác nhau (từ chối lệnh apply dạng cũ).
"""
from __future__ import annotations

from typing import Any

from worker.adapters.result_publisher import CURRICULUM_IMPORT_RESULT, make_result_id

MAX_ERROR_CHARS = 2000


class CurriculumImportReporter:
    def __init__(self, publisher: Any) -> None:
        self._publisher = publisher

    def preview(
        self, import_id: str, preview: dict[str, Any], curriculum: dict[str, Any]
    ) -> str:
        """
        Gửi bản xem trước; trả trạng thái đã báo (BLOCKED nếu có lỗi chặn, không
        thì READY). Lỗi gửi nổi lên, kể cả ``InvalidWorkerResult`` khi đề cương
        không đóng gói được thành JSON.
        """
        status = "BLOCKED" if preview["blocking"] else "READY"
        self._publisher.publish_sync(
            CURRICULUM_IMPORT_RESULT,
            {
                "import_id": import_id,
                "status": status,
                "preview": preview,
                "curriculum": curriculum,
                "error": None,
            },
            result_id=make_result_id(CURRICULUM_IMPORT_RESULT, import_id),
        )
        return status

    def failure(self, import_id: str, error: str, *, attempt: str | None = None) -> None:
        self._publisher.publish_sync(
            CURRICULUM_IMPORT_RESULT,
            {
                "import_id": import_id,
                "status": "FAILED",
                "preview": None,
                "curriculum": None,
                "error": error[:MAX_ERROR_CHARS],
            },
            result_id=make_result_id(CURRICULUM_IMPORT_RESULT, import_id, attempt),
        )


__all__ = ["MAX_ERROR_CHARS", "CurriculumImportReporter"]
