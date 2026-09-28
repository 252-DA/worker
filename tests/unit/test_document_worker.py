from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from document_chunk.domain.ports.metadata_store import IngestionStatus
from document_chunk.shared.result import Ok
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
