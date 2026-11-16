from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from bullmq.custom_errors import UnrecoverableError

from document_chunk.shared.result import Err, Ok

from worker.runners.enrichment_worker import _process_job


def _job(**data):
    return SimpleNamespace(id="11", attemptsStarted=3, attemptsMade=2, data=data)


@pytest.mark.asyncio
async def test_generation_and_attempt_reach_the_use_case():
    container = MagicMock()
    container.metadata_store.get_document_processing_generation.return_value = Ok(6)
    container.run_enrichment_use_case.execute.return_value = Ok(
        SimpleNamespace(concept_count=1, mention_count=2, card_count=3, quiz_count=4)
    )

    result = await _process_job(
        container, _job(document_id="doc-1", processing_generation=6), "token"
    )

    assert result["card_count"] == 3
    request = container.run_enrichment_use_case.execute.call_args.args[0]
    assert (request.document_id, request.processing_generation, request.attempt) == (
        "doc-1", 6, 3,
    )


@pytest.mark.asyncio
async def test_enrichment_of_an_older_generation_is_skipped_without_running_the_llm():
    container = MagicMock()
    container.metadata_store.get_document_processing_generation.return_value = Ok(7)

    result = await _process_job(
        container, _job(document_id="doc-1", processing_generation=6), "token"
    )

    assert result == {
        "document_id": "doc-1", "skipped": "stale_generation",
        "job_generation": 6, "current_generation": 7,
    }
    container.run_enrichment_use_case.execute.assert_not_called()


@pytest.mark.asyncio
async def test_enrichment_of_a_deleted_document_runs_as_before():
    container = MagicMock()
    container.metadata_store.get_document_processing_generation.return_value = Ok(None)
    container.run_enrichment_use_case.execute.return_value = Err(RuntimeError("stale"))

    with pytest.raises(RuntimeError):
        await _process_job(container, _job(document_id="doc-1", processing_generation=6), "token")
    container.run_enrichment_use_case.execute.assert_called_once()


@pytest.mark.asyncio
async def test_enrichment_generation_read_failure_fails_the_job_for_retry():
    container = MagicMock()
    container.metadata_store.get_document_processing_generation.return_value = Err(
        ConnectionError("db down")
    )

    with pytest.raises(ConnectionError):
        await _process_job(container, _job(document_id="doc-1", processing_generation=6), "token")
    container.run_enrichment_use_case.execute.assert_not_called()


@pytest.mark.asyncio
async def test_old_jobs_without_generation_still_run():
    container = MagicMock()
    container.run_enrichment_use_case.execute.return_value = Ok(
        SimpleNamespace(concept_count=0, mention_count=0, card_count=0, quiz_count=0)
    )

    await _process_job(container, _job(document_id="doc-1"), "token")

    request = container.run_enrichment_use_case.execute.call_args.args[0]
    assert request.processing_generation == 0
    container.metadata_store.get_document_processing_generation.assert_not_called()


@pytest.mark.asyncio
async def test_use_case_error_fails_the_job_for_retry():
    """Gồm cả không gửi được kết quả về core-api: job không được hoàn tất."""
    container = MagicMock()
    container.run_enrichment_use_case.execute.return_value = Err(TimeoutError("no ack"))

    with pytest.raises(TimeoutError):
        await _process_job(container, _job(document_id="doc-1"), "token")


@pytest.mark.asyncio
async def test_invalid_payload_is_unrecoverable():
    container = MagicMock()

    with pytest.raises(UnrecoverableError):
        await _process_job(
            container, _job(document_id="doc-1", processing_generation="x"), "token"
        )
    container.run_enrichment_use_case.execute.assert_not_called()
