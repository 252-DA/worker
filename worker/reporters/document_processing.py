"""
Trạng thái và chương của tài liệu → core-api qua ``worker_results``.

`documents` là bảng của core-api: worker không UPDATE nữa mà gửi kết quả
DOCUMENT_PROCESSING_RESULT (contracts/integration-events/v1/
document-processing-result.json); core-api
(src/modules/document/handlers/document-processing-result.handler.ts) ghi.

Kết quả đi qua hàng đợi nên có thể đến trễ hoặc lệch thứ tự (handler bên
core-api retry). Mỗi báo trạng thái mang đủ dữ kiện để core-api xếp thứ tự:

- ``processing_generation``: lượt xử lý core-api cấp khi đưa tài liệu vào hàng
  đợi (job ``process_document`` mang theo). Kết quả của lượt cũ bị bỏ.
- ``stage``: ``processing`` (job tài liệu) rồi ``enrichment`` (job enrichment và
  bước chiếu concept graph sau đó).
- ``attempt``: lần chạy thứ mấy của job BullMQ, để lần chạy lại (PARSING sau
  ERROR) thắng kết quả của lần trước nhưng PARSING đến trễ không đè ERROR.

Thứ tự do core-api tính từ ba dữ kiện trên và trạng thái; worker chỉ báo sự thật.
"""
from __future__ import annotations

import uuid
from typing import Any

from document_chunk.domain.exceptions import DocumentStaleError
from document_chunk.domain.ports.document_reporter import (
    IDocumentChapterReporter,
    IDocumentStatusReporter,
)
from document_chunk.domain.ports.metadata_store import IMetadataStore, IngestionStatus
from document_chunk.shared.logger import get_logger
from document_chunk.shared.result import Err, Ok, Result

from worker.adapters.result_publisher import (
    DOCUMENT_PROCESSING_RESULT,
    InvalidWorkerResult,
    make_result_id,
)

logger = get_logger(__name__)

STAGE_PROCESSING = "processing"
STAGE_ENRICHMENT = "enrichment"

# Trạng thái mỗi giai đoạn được báo; khớp schema và handler bên core-api.
# Enrichment lỗi thì lùi về INDEXED để tài liệu vẫn tìm kiếm được, không ERROR.
STAGE_STATUSES: dict[str, frozenset[IngestionStatus]] = {
    STAGE_PROCESSING: frozenset(
        {
            IngestionStatus.PARSING,
            IngestionStatus.CHUNKING,
            IngestionStatus.EMBEDDING,
            IngestionStatus.INDEXED,
            IngestionStatus.ERROR,
        }
    ),
    STAGE_ENRICHMENT: frozenset(
        {
            IngestionStatus.ENRICHING,
            IngestionStatus.INDEXED,
            IngestionStatus.GENERATED_DRAFT,
        }
    ),
}

MAX_ERROR_CHARS = 1000
MAX_REASON_CHARS = 255
MAX_CHAPTER_CODE_CHARS = 20


class PublishingDocumentStatusReporter(IDocumentStatusReporter):
    """Báo trạng thái của một lượt (generation, stage, attempt) cố định."""

    def __init__(
        self,
        publisher: Any,
        metadata_store: IMetadataStore,
        *,
        stage: str,
        processing_generation: int,
        attempt: int,
    ) -> None:
        if stage not in STAGE_STATUSES:
            raise ValueError(f"unknown document processing stage: {stage!r}")
        if isinstance(processing_generation, bool) or not isinstance(processing_generation, int):
            raise ValueError("processing_generation must be an int")
        if processing_generation < 0:
            raise ValueError("processing_generation must be >= 0")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise ValueError("attempt must be an int >= 1")
        self._publisher = publisher
        self._metadata_store = metadata_store
        self._stage = stage
        self._generation = processing_generation
        self._attempt = attempt

    def report_status(
        self,
        document_id: str,
        status: IngestionStatus,
        error_msg: str | None = None,
    ) -> Result[None, Exception]:
        status = IngestionStatus(status)
        if status not in STAGE_STATUSES[self._stage]:
            return Err(
                InvalidWorkerResult(
                    f"status {status.value} is not reported in stage {self._stage}"
                )
            )

        # Delete-race guard (trước đây nằm trong UPDATE ... AND deleted_at IS
        # NULL): tài liệu đã xoá thì dừng lượt xử lý, không ghi dữ liệu dẫn xuất.
        context = self._metadata_store.get_document_context(document_id)
        if context.is_err():
            return Err(context.error)
        if context.unwrap() is None:
            return Err(
                DocumentStaleError(
                    f"document {document_id} not found or deleted",
                    document_id=document_id,
                )
            )

        report: dict[str, Any] = {
            "value": status.value,
            "stage": self._stage,
            "processing_generation": self._generation,
            "attempt": self._attempt,
        }
        if error_msg:
            report["error"] = error_msg[:MAX_ERROR_CHARS]
            # documents chưa có cột lỗi: log ở đây để lỗi không mất.
            logger.error(
                "document.status.error",
                document_id=document_id,
                status=status.value,
                stage=self._stage,
                error=error_msg,
            )

        try:
            # Mỗi (lượt, giai đoạn, lần chạy, trạng thái) xảy ra tối đa một lần:
            # job gốc chạy lại sau khi đã gửi sinh lại đúng id, BullMQ bỏ bản trùng.
            result_id = make_result_id(
                DOCUMENT_PROCESSING_RESULT,
                str(document_id),
                f"g{self._generation}-{self._stage}{self._attempt}-{status.value}",
            )
            self._publisher.publish_sync(
                DOCUMENT_PROCESSING_RESULT,
                {"document_id": str(document_id), "status": report},
                result_id=result_id,
            )
        except Exception as exc:  # noqa: BLE001 -- Redis/hợp đồng: lượt xử lý phải lỗi
            return Err(exc)
        return Ok(None)


class PublishingDocumentChapterReporter(IDocumentChapterReporter):
    """
    Chương suy ra từ nội dung. Core-api chỉ ghi khi tài liệu chưa có chương
    hoặc chương hiện tại cũng do suy luận nội dung: không đè chương giảng viên
    xác nhận ('confirmed') hay chương lấy từ module/tên file.
    """

    def __init__(self, publisher: Any) -> None:
        self._publisher = publisher

    def report_chapter(
        self,
        document_id: str,
        chapter_code: str,
        provenance: str,
        confidence: float,
        reason: str,
    ) -> Result[None, Exception]:
        payload = {
            "document_id": str(document_id),
            "chapter": {
                "chapter_code": chapter_code[:MAX_CHAPTER_CODE_CHARS],
                "provenance": provenance,
                # documents.chapter_confidence là DECIMAL(3,2).
                "confidence": round(min(max(float(confidence), 0.0), 1.0), 2),
                "reason": reason[:MAX_REASON_CHARS],
            },
        }
        try:
            # Id riêng mỗi lần báo: cùng tài liệu có thể được map lại nhiều lần
            # (C1 → C2 → C1), id theo nội dung sẽ bị BullMQ chặn nhầm lần thứ ba.
            # Gửi lại khi job retry chỉ ghi lại đúng giá trị đó — vô hại.
            result_id = make_result_id(
                DOCUMENT_PROCESSING_RESULT,
                str(document_id),
                f"chapter-{uuid.uuid4().hex}",
            )
            self._publisher.publish_sync(
                DOCUMENT_PROCESSING_RESULT, payload, result_id=result_id
            )
        except Exception as exc:  # noqa: BLE001
            return Err(exc)
        return Ok(None)
