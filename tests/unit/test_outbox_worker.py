from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from bullmq.custom_errors import UnrecoverableError

from document_chunk.domain.outbox_events import OutboxEventType
from document_chunk.shared.result import Err, Ok
from worker.runners.outbox_worker import _process_job


@pytest.fixture(autouse=True)
def _run_to_thread_inline(monkeypatch):
    async def inline(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("worker.runners.outbox_worker.asyncio.to_thread", inline)


def _job_data(event_type: str = "DOCUMENT_DELETED") -> dict:
    return {
        "event_id": "event-001",
        "event_type": event_type,
        "aggregate_type": "document",
        "aggregate_id": "doc-001",
        "payload": {"document_id": "doc-001"},
    }


@pytest.mark.asyncio
async def test_process_job_parses_and_projects_relay_event():
    container = MagicMock()
    container.outbox_projector.process.return_value = Ok(None)

    result = await _process_job(
        container,
        SimpleNamespace(id="job-001", data=_job_data()),
        None,
    )

    event = container.outbox_projector.process.call_args.args[0]
    assert event.event_id == "event-001"
    assert event.event_type == OutboxEventType.DOCUMENT_DELETED
    assert result == {
        "event_id": "event-001",
        "event_type": "DOCUMENT_DELETED",
        "aggregate_id": "doc-001",
    }


@pytest.mark.asyncio
async def test_malformed_job_is_unrecoverable():
    container = MagicMock()

    with pytest.raises(UnrecoverableError, match="invalid outbox relay event"):
        await _process_job(
            container,
            SimpleNamespace(id="job-001", data={"event_type": "DOCUMENT_DELETED"}),
            None,
        )

    container.outbox_projector.process.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_event_type_is_unrecoverable():
    container = MagicMock()

    with pytest.raises(UnrecoverableError, match="invalid outbox relay event"):
        await _process_job(
            container,
            SimpleNamespace(id="job-001", data=_job_data("UNKNOWN_EVENT")),
            None,
        )

    container.outbox_projector.process.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_projection_event_is_unrecoverable():
    container = MagicMock()
    container.outbox_projector.process.return_value = Err(
        ValueError("payload.document_id must be a non-empty string")
    )

    with pytest.raises(UnrecoverableError, match="invalid outbox projection event"):
        await _process_job(
            container,
            SimpleNamespace(id="job-001", data=_job_data()),
            None,
        )


@pytest.mark.asyncio
async def test_transient_projection_failure_is_retryable():
    container = MagicMock()
    failure = RuntimeError("neo4j unavailable")
    container.outbox_projector.process.return_value = Err(failure)

    with pytest.raises(RuntimeError, match="neo4j unavailable"):
        await _process_job(
            container,
            SimpleNamespace(id="job-001", data=_job_data()),
            None,
        )
