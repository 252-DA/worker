"""
Bỏ job của lượt xử lý cũ.

core-api tăng ``documents.processing_generation`` mỗi lần đưa tài liệu vào hàng
đợi (xử lý lại tài liệu ERROR, đồng bộ lại từ Canvas) và gửi lượt kèm job. Job của
lượt cũ vẫn nằm trong hàng đợi hoặc đang chạy; nếu để nó chạy hết pipeline thì
trạng thái nó báo bị core-api bỏ (sai lượt) nhưng chunk, vector và graph vẫn bị ghi
chồng lên kết quả của lượt mới.
"""
from __future__ import annotations

from document_chunk.domain.ports.metadata_store import IMetadataStore


def newer_generation(
    metadata_store: IMetadataStore, document_id: str, job_generation: int
) -> int | None:
    """
    Lượt hiện tại của tài liệu nếu job (``job_generation``) cũ hơn nó, ngược lại None.

    - Job cũ chưa mang lượt (0): giữ hành vi cũ, không kiểm.
    - Tài liệu không còn (đã xoá): None, để pipeline xử lý như trước (DocumentStaleError).
    - Không đọc được: ném lỗi cho BullMQ thử lại — không đoán rồi chạy hay bỏ.
    """
    if job_generation <= 0:
        return None
    result = metadata_store.get_document_processing_generation(document_id)
    if result.is_err():
        raise result.error
    current = result.unwrap()
    if current is not None and job_generation < current:
        return current
    return None
