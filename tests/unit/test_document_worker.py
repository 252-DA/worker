from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bullmq.custom_errors import UnrecoverableError

from document_chunk.domain.ports.metadata_store import IngestionStatus
from document_chunk.shared.result import Err, Ok
from worker.runners.document_worker import _process_job


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [
        IngestionStatus.INDEXED,
        IngestionStatus.ENRICHING,
        IngestionStatus.GENERATED_DRAFT,
    ],
)
async def test_process_job_skips_documents_that_are_already_indexed(status):
    container = MagicMock()
    container.metadata_store.get_document_status.return_value = Ok(
        (status, None, "documents/test.pdf")
    )
    container.metadata_store.list_chunks.return_value = Ok([MagicMock(), MagicMock()])
    job = SimpleNamespace(
        id="5",
        data={
            "document_id": "019fdbaa-c096-78e5-a91a-982998d5b5b0",
            "storage_key": "documents/test.pdf",
            "file_name": "Architecture Design Specification.pdf",
        },
    )

    result = await _process_job(container, job, "token")

    assert result == {
        "document_id": "019fdbaa-c096-78e5-a91a-982998d5b5b0",
        "chunk_count": 2,
    }
    container.file_storage.download.assert_not_called()
    container.run_pipeline_use_case.execute.assert_not_called()


@pytest.mark.asyncio
async def test_process_job_offloads_pipeline_from_the_bullmq_event_loop():
    container = MagicMock()
    container.metadata_store.get_document_status.return_value = Ok(
        (IngestionStatus.QUEUED, None, "documents/test.pdf")
    )
    job = SimpleNamespace(
        id="6",
        data={
            "document_id": "019fdc42-080c-7248-afc8-1b55a7a885ec",
            "storage_key": "documents/test.pdf",
            "file_name": "Architecture Design Specification.pdf",
        },
    )
    expected = {"document_id": job.data["document_id"], "chunk_count": 135}

    with patch(
        "worker.runners.document_worker.asyncio.to_thread",
        new=AsyncMock(return_value=expected),
    ) as to_thread:
        result = await _process_job(container, job, "token")

    assert result == expected
    to_thread.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_job_passes_generation_and_attempt_to_the_pipeline():
    """Core-api xếp thứ tự kết quả trạng thái theo lượt + lần chạy của job."""
    container = MagicMock()
    container.metadata_store.get_document_status.return_value = Ok(
        (IngestionStatus.QUEUED, None, "documents/test.pdf")
    )
    container.metadata_store.get_document_processing_generation.return_value = Ok(4)
    container.file_storage.download.return_value = Ok(None)
    container.run_pipeline_use_case.execute.return_value = Ok(
        SimpleNamespace(chunk_count=3, processing_time_ms=1.0)
    )
    job = SimpleNamespace(
        id="7",
        attemptsStarted=2,
        attemptsMade=1,
        data={
            "document_id": "019fdc42-080c-7248-afc8-1b55a7a885ec",
            "storage_key": "documents/test.pdf",
            "file_name": "a.pdf",
            "processing_generation": 4,
        },
    )

    result = await _process_job(container, job, "token")

    assert result["chunk_count"] == 3
    request = container.run_pipeline_use_case.execute.call_args.args[0]
    assert (request.processing_generation, request.attempt) == (4, 2)


@pytest.mark.asyncio
async def test_pipeline_error_fails_the_job_for_retry():
    """Kể cả khi không gửi được trạng thái về core-api: job lỗi, BullMQ retry."""
    container = MagicMock()
    container.metadata_store.get_document_status.return_value = Ok(None)
    container.file_storage.download.return_value = Ok(None)
    container.run_pipeline_use_case.execute.return_value = Err(ConnectionError("redis down"))
    job = SimpleNamespace(
        id="8",
        data={"document_id": "d", "storage_key": "k/a.pdf", "file_name": "a.pdf"},
    )

    with pytest.raises(ConnectionError):
        await _process_job(container, job, "token")


def _generation_job(generation, **extra):
    data = {"document_id": "doc-1", "storage_key": "k/a.pdf", "file_name": "a.pdf", **extra}
    if generation is not None:
        data["processing_generation"] = generation
    return SimpleNamespace(id="12", attemptsStarted=1, attemptsMade=0, data=data)


def _generation_container(current):
    container = MagicMock()
    container.metadata_store.get_document_processing_generation.return_value = current
    container.metadata_store.get_document_status.return_value = Ok(
        (IngestionStatus.QUEUED, None, "k/a.pdf")
    )
    container.file_storage.download.return_value = Ok(None)
    container.run_pipeline_use_case.execute.return_value = Ok(
        SimpleNamespace(chunk_count=2, processing_time_ms=1.0)
    )
    return container


@pytest.mark.asyncio
async def test_job_of_an_older_generation_is_acked_without_running_or_publishing():
    """Tài liệu đã vào hàng đợi lại (lượt 5): job lượt 3 không được ghi chồng chunk/vector."""
    container = _generation_container(Ok(5))

    result = await _process_job(container, _generation_job(3), "token")

    assert result == {
        "document_id": "doc-1", "skipped": "stale_generation",
        "job_generation": 3, "current_generation": 5,
    }
    container.metadata_store.get_document_processing_generation.assert_called_once_with("doc-1")
    container.file_storage.download.assert_not_called()
    container.run_pipeline_use_case.execute.assert_not_called()
    container.result_publisher.publish_sync.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("current", [Ok(3), Ok(2), Ok(None)])
async def test_job_of_the_current_or_a_newer_generation_still_runs(current):
    # Bằng lượt hiện tại, mới hơn (đọc ở replica trễ), hoặc tài liệu đã xoá (None:
    # pipeline tự xử lý DocumentStaleError như trước).
    container = _generation_container(current)

    result = await _process_job(container, _generation_job(3), "token")

    assert result["chunk_count"] == 2
    container.run_pipeline_use_case.execute.assert_called_once()


@pytest.mark.asyncio
async def test_job_without_generation_keeps_the_old_behavior():
    """Job xếp trước khi có lượt (không có trường): không đọc lượt, chạy như cũ."""
    container = _generation_container(Ok(9))

    result = await _process_job(container, _generation_job(None), "token")

    assert result["chunk_count"] == 2
    container.metadata_store.get_document_processing_generation.assert_not_called()
    container.run_pipeline_use_case.execute.assert_called_once()


@pytest.mark.asyncio
async def test_generation_read_failure_fails_the_job_for_retry():
    container = _generation_container(Err(ConnectionError("db down")))

    with pytest.raises(ConnectionError):
        await _process_job(container, _generation_job(3), "token")
    container.run_pipeline_use_case.execute.assert_not_called()


@pytest.mark.asyncio
async def test_stale_job_is_skipped_before_the_already_completed_shortcut():
    """Job cũ không được trả "đã xong" theo trạng thái của lượt mới."""
    container = _generation_container(Ok(5))
    container.metadata_store.get_document_status.return_value = Ok(
        (IngestionStatus.INDEXED, None, "k/a.pdf")
    )

    result = await _process_job(container, _generation_job(2), "token")

    assert result["skipped"] == "stale_generation"
    container.metadata_store.list_chunks.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("generation", [-1, "2", 1.5, True])
async def test_invalid_generation_is_unrecoverable(generation):
    container = MagicMock()
    job = SimpleNamespace(
        id="9",
        data={
            "document_id": "d",
            "storage_key": "k/a.pdf",
            "file_name": "a.pdf",
            "processing_generation": generation,
        },
    )

    with pytest.raises(UnrecoverableError):
        await _process_job(container, job, "token")
    container.run_pipeline_use_case.execute.assert_not_called()
