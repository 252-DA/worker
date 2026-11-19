"""Test double dùng chung."""
from document_chunk.domain.ports.llm_client import LLMUsage
from document_chunk.shared.result import Ok


def as_llm_client(mock):
    """
    Cho một MagicMock hành xử đúng như ``ILLMClient``.

    Chỗ gọi nào cần token sẽ dùng ``generate_with_usage``; test chỉ stub
    ``generate`` nên cần nối hai đường lại, nếu không mock trả về MagicMock và
    lỗi chỉ hiện ra ở tận chỗ giải nén tuple. Usage để rỗng: test double không
    có số liệu thật, và 0 token nghĩa là "không đo được", không phải "miễn phí".
    """
    def _with_usage(prompt, system=None):
        result = mock.generate(prompt, system=system)
        if result.is_err():
            return result
        return Ok((result.unwrap(), LLMUsage()))

    mock.generate_with_usage.side_effect = _with_usage
    return mock


class RecordingResultPublisher:
    """
    Thay ``WorkerResultPublisher`` trong test use case/runner: kiểm envelope
    đúng như bản thật (kind, result_id, payload JSON) rồi ghi lại thay vì gửi
    Redis. ``fail_with`` giả Redis không nhận để thử đường job gốc phải retry.
    """

    def __init__(self, fail_with: Exception | None = None) -> None:
        self.envelopes: list[dict] = []
        self.fail_with = fail_with

    def publish_sync(self, kind, payload, *, result_id, source_event_id=None, occurred_at=None) -> str:
        from worker.services.result_publisher import build_envelope

        envelope = build_envelope(
            kind, payload, result_id=result_id,
            source_event_id=source_event_id, occurred_at=occurred_at,
        )
        if self.fail_with is not None:
            raise self.fail_with
        self.envelopes.append(envelope)
        return result_id

    async def publish(self, kind, payload, *, result_id, source_event_id=None, occurred_at=None) -> str:
        return self.publish_sync(
            kind, payload, result_id=result_id,
            source_event_id=source_event_id, occurred_at=occurred_at,
        )

    def close(self) -> None:
        pass


# Tên hàm ghi của IMetadataStore. ``record_llm_usage`` cố ý không nằm trong đây:
# llm_usage_logs là bảng AI của worker.
_WRITER_PREFIXES = ("update_", "persist_", "upsert_", "save_", "mark_", "release_", "delete", "set_", "soft_delete")


def assert_store_made_no_writes(store) -> None:
    """
    Store (MagicMock) không nhận lời gọi ghi nào.

    Bền hơn ``assert_not_called`` trên từng tên hàm: hàm ghi bị xoá hay đổi tên
    thì kiểm tra theo tên cũ trở thành vô nghĩa, còn kiểm tra theo tiền tố thì
    vẫn bắt được một lời gọi ghi mới.
    """
    writes = [name for name, _args, _kwargs in store.method_calls if name.startswith(_WRITER_PREFIXES)]
    assert writes == [], f"use case gọi hàm ghi của store: {writes}"
