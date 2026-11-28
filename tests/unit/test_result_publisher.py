"""
Tests for worker/adapters/result_publisher.py — producer của hàng đợi worker_results.

Queue của bullmq được thay bằng FakeQueue; phần được thử thật là thread + event
loop riêng của publisher, vì đó là chỗ dễ sai (deadlock, nuốt lỗi, loop sai).
"""
import asyncio
import concurrent.futures
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from worker.adapters.result_publisher import (
    DEFAULT_JOB_OPTIONS,
    ESSAY_GRADING_RESULT,
    QUIZ_GRADING_RESULT,
    RESULT_ID_PATTERN,
    WORKER_RESULT_KINDS,
    WORKER_RESULTS_QUEUE_NAME,
    InvalidWorkerResult,
    WorkerResultPublisher,
    build_envelope,
    make_result_id,
)
from tests.support import RecordingResultPublisher

ATTEMPT_ID = "6f1c2d3e-4b5a-4c6d-8e7f-9a0b1c2d3e4f"
EVENT_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
REDIS = {"host": "redis.test", "port": 6380, "db": 2}


class FakeQueue:
    def __init__(self, name, opts, *, add_delay=0.0, add_error=None, block=False):
        self.name = name
        self.opts = opts
        self.calls = []
        self.closed = False
        self.cancelled = 0
        self.created_on = threading.current_thread().name
        self._add_delay = add_delay
        self._add_error = add_error
        self._block = block
        self.events = []

    async def add(self, name, data, opts):
        self.calls.append((name, data, opts, threading.current_thread().name))
        try:
            if self._block:
                await asyncio.Event().wait()
            if self._add_delay:
                await asyncio.sleep(self._add_delay)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        if self._add_error is not None:
            raise self._add_error
        self.events.append("added")
        return SimpleNamespace(id=opts["jobId"])

    async def close(self):
        self.events.append("closed")
        self.closed = True


class QueueFactory:
    def __init__(self, **queue_kwargs):
        self.queues: list[FakeQueue] = []
        self._kwargs = queue_kwargs

    def __call__(self, name, opts):
        queue = FakeQueue(name, opts, **self._kwargs)
        self.queues.append(queue)
        return queue


@pytest.fixture
def factory():
    return QueueFactory()


@pytest.fixture
def publisher(factory):
    pub = WorkerResultPublisher(REDIS, queue_factory=factory, timeout_seconds=5)
    yield pub
    pub.close()


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.01)


def _publish_sync(pub, **overrides):
    kwargs = {
        "result_id": make_result_id(QUIZ_GRADING_RESULT, EVENT_ID),
        "source_event_id": EVENT_ID,
    }
    kwargs.update(overrides)
    return pub.publish_sync(QUIZ_GRADING_RESULT, {"attempt_id": ATTEMPT_ID, "score": 7.5}, **kwargs)


# ---------------------------------------------------------------------------
# result_id
# ---------------------------------------------------------------------------

class TestMakeResultId:
    def test_is_deterministic(self):
        first = make_result_id(QUIZ_GRADING_RESULT, ATTEMPT_ID, 2)
        assert first == make_result_id(QUIZ_GRADING_RESULT, ATTEMPT_ID, 2)
        assert first == f"QUIZ_GRADING_RESULT.{ATTEMPT_ID}.2"
        assert make_result_id(QUIZ_GRADING_RESULT, ATTEMPT_ID) == f"QUIZ_GRADING_RESULT.{ATTEMPT_ID}"

    def test_distinguishes_kind_source_and_attempt(self):
        ids = {
            make_result_id(QUIZ_GRADING_RESULT, ATTEMPT_ID),
            make_result_id(ESSAY_GRADING_RESULT, ATTEMPT_ID),
            make_result_id(QUIZ_GRADING_RESULT, EVENT_ID),
            make_result_id(QUIZ_GRADING_RESULT, ATTEMPT_ID, 0),
            make_result_id(QUIZ_GRADING_RESULT, ATTEMPT_ID, 1),
        }
        assert len(ids) == 5

    def test_is_a_valid_bullmq_custom_job_id_on_both_sides(self):
        result_id = make_result_id(QUIZ_GRADING_RESULT, ATTEMPT_ID, 3)
        # BullMQ Node 5.x: custom id không chứa ':' và không phải số nguyên.
        assert ":" not in result_id
        assert not result_id.isdigit()
        assert RESULT_ID_PATTERN.fullmatch(result_id)

    @pytest.mark.parametrize(
        ("kind", "source_id", "attempt"),
        [
            ("NOT_A_KIND", ATTEMPT_ID, None),
            (QUIZ_GRADING_RESULT, f"grading:{ATTEMPT_ID}", None),
            (QUIZ_GRADING_RESULT, "a.b", None),  # '.' là dấu nối
            (QUIZ_GRADING_RESULT, "", None),
            (QUIZ_GRADING_RESULT, None, None),
            (QUIZ_GRADING_RESULT, "has space", None),
            (QUIZ_GRADING_RESULT, ATTEMPT_ID, -1),
            (QUIZ_GRADING_RESULT, ATTEMPT_ID, True),
            (QUIZ_GRADING_RESULT, ATTEMPT_ID, "1:2"),
            (QUIZ_GRADING_RESULT, ATTEMPT_ID, 1.5),
            (QUIZ_GRADING_RESULT, "x" * 200, None),
        ],
    )
    def test_rejects_unsafe_parts(self, kind, source_id, attempt):
        with pytest.raises(InvalidWorkerResult):
            make_result_id(kind, source_id, attempt)


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------

class TestBuildEnvelope:
    def test_shape_matches_contract(self):
        when = datetime(2026, 11, 7, 17, 30, 15, 123456, tzinfo=timezone(timedelta(hours=7)))
        envelope = build_envelope(
            QUIZ_GRADING_RESULT,
            {"attempt_id": ATTEMPT_ID, "criteria": ("a", "b")},
            result_id="QUIZ_GRADING_RESULT.x",
            source_event_id=EVENT_ID,
            occurred_at=when,
        )
        assert envelope == {
            "result_id": "QUIZ_GRADING_RESULT.x",
            "kind": QUIZ_GRADING_RESULT,
            "source_event_id": EVENT_ID,
            "occurred_at": "2026-11-07T10:30:15.123Z",
            "payload": {"attempt_id": ATTEMPT_ID, "criteria": ["a", "b"]},
        }

    def test_defaults_to_now_in_utc_and_null_source(self):
        before = datetime.now(timezone.utc).replace(microsecond=0)
        envelope = build_envelope(QUIZ_GRADING_RESULT, {}, result_id="QUIZ_GRADING_RESULT.x")
        assert envelope["source_event_id"] is None
        assert envelope["occurred_at"].endswith("Z")
        occurred = datetime.fromisoformat(envelope["occurred_at"].replace("Z", "+00:00"))
        assert before <= occurred <= datetime.now(timezone.utc)

    def test_payload_is_detached_from_caller(self):
        payload = {"items": [1]}
        envelope = build_envelope(QUIZ_GRADING_RESULT, payload, result_id="QUIZ_GRADING_RESULT.x")
        payload["items"].append(2)
        assert envelope["payload"] == {"items": [1]}

    @pytest.mark.parametrize(
        "overrides",
        [
            {"kind": "UNKNOWN_RESULT"},
            {"result_id": "QUIZ_GRADING_RESULT:x"},
            {"result_id": "12345"},
            {"result_id": ""},
            {"source_event_id": ""},
            {"source_event_id": 42},
            {"payload": ["not", "an", "object"]},
            {"payload": {"score": float("nan")}},
            {"payload": {"when": datetime.now(timezone.utc)}},
            {"occurred_at": datetime(2026, 11, 7, 10, 0, 0)},  # thiếu múi giờ
        ],
    )
    def test_rejects_contract_violations(self, overrides):
        args = {
            "kind": QUIZ_GRADING_RESULT,
            "payload": {"ok": True},
            "result_id": "QUIZ_GRADING_RESULT.x",
            "source_event_id": None,
            "occurred_at": None,
        }
        args.update(overrides)
        kind, payload = args.pop("kind"), args.pop("payload")
        with pytest.raises(InvalidWorkerResult):
            build_envelope(kind, payload, **args)


# ---------------------------------------------------------------------------
# Publisher
# ---------------------------------------------------------------------------

class TestPublishSync:
    def test_adds_job_named_by_kind_with_result_id_as_job_id(self, publisher, factory):
        result_id = make_result_id(QUIZ_GRADING_RESULT, EVENT_ID)

        job_id = _publish_sync(publisher)

        assert job_id == result_id
        [queue] = factory.queues
        assert queue.name == WORKER_RESULTS_QUEUE_NAME
        assert queue.opts == {"connection": REDIS}
        [(name, data, opts, thread_name)] = queue.calls
        assert name == QUIZ_GRADING_RESULT
        assert data["result_id"] == result_id
        assert data["kind"] == QUIZ_GRADING_RESULT
        assert data["source_event_id"] == EVENT_ID
        assert data["payload"] == {"attempt_id": ATTEMPT_ID, "score": 7.5}
        assert opts == {**DEFAULT_JOB_OPTIONS, "jobId": result_id}
        # Queue sống trên loop riêng của publisher, không trên thread người gọi.
        assert thread_name == "worker-results-publisher"
        assert queue.created_on == "worker-results-publisher"

    def test_default_job_options_retry_and_keep_failures(self):
        assert DEFAULT_JOB_OPTIONS["attempts"] >= 5
        assert DEFAULT_JOB_OPTIONS["backoff"]["type"] == "exponential"
        # Job hỏng là kết quả chưa ghi: không được xoá theo số lượng.
        assert "count" not in DEFAULT_JOB_OPTIONS["removeOnFail"]
        assert DEFAULT_JOB_OPTIONS["removeOnFail"]["age"] >= 7 * 24 * 3600

    def test_reuses_one_queue_across_calls(self, publisher, factory):
        _publish_sync(publisher)
        _publish_sync(publisher, result_id=make_result_id(QUIZ_GRADING_RESULT, ATTEMPT_ID))
        assert len(factory.queues) == 1
        assert len(factory.queues[0].calls) == 2

    def test_redis_error_propagates(self):
        factory = QueueFactory(add_error=ConnectionError("redis down"))
        pub = WorkerResultPublisher(REDIS, queue_factory=factory, timeout_seconds=5)
        try:
            with pytest.raises(ConnectionError, match="redis down"):
                _publish_sync(pub)
            # Lỗi một lần không làm hỏng publisher.
            factory.queues[0]._add_error = None
            assert _publish_sync(pub)
        finally:
            pub.close()

    def test_times_out_and_cancels_the_pending_add(self):
        factory = QueueFactory(block=True)
        pub = WorkerResultPublisher(REDIS, queue_factory=factory, timeout_seconds=0.2)
        try:
            with pytest.raises(TimeoutError, match="did not acknowledge"):
                _publish_sync(pub)
            _wait_until(lambda: factory.queues[0].cancelled == 1)
        finally:
            pub.close()

    def test_invalid_result_fails_before_touching_redis(self, publisher, factory):
        with pytest.raises(InvalidWorkerResult):
            publisher.publish_sync("NOT_A_KIND", {}, result_id="NOT_A_KIND.x")
        assert factory.queues == []
        assert publisher._thread is None

    def test_many_threads_publish_concurrently(self, publisher, factory):
        def one(i):
            return _publish_sync(publisher, result_id=make_result_id(QUIZ_GRADING_RESULT, f"src-{i}"))

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            ids = list(pool.map(one, range(40)))

        assert ids == [f"QUIZ_GRADING_RESULT.src-{i}" for i in range(40)]
        assert len(factory.queues) == 1
        assert len(factory.queues[0].calls) == 40

    async def test_refuses_to_block_a_running_event_loop(self, publisher, factory):
        with pytest.raises(RuntimeError, match="await publish"):
            _publish_sync(publisher)
        assert factory.queues == []


class TestPublishAsync:
    async def test_runner_coroutine_publishes(self, publisher, factory):
        result_id = make_result_id(ESSAY_GRADING_RESULT, ATTEMPT_ID, 1)

        job_id = await publisher.publish(ESSAY_GRADING_RESULT, {"score": 8}, result_id=result_id)

        assert job_id == result_id
        [(name, data, opts, thread_name)] = factory.queues[0].calls
        assert (name, data["payload"], opts["jobId"]) == (ESSAY_GRADING_RESULT, {"score": 8}, result_id)
        assert thread_name == "worker-results-publisher"

    async def test_redis_error_propagates(self):
        pub = WorkerResultPublisher(
            REDIS, queue_factory=QueueFactory(add_error=ConnectionError("redis down")), timeout_seconds=5
        )
        try:
            with pytest.raises(ConnectionError, match="redis down"):
                await pub.publish(QUIZ_GRADING_RESULT, {}, result_id="QUIZ_GRADING_RESULT.x")
        finally:
            pub.close()

    async def test_times_out(self):
        factory = QueueFactory(block=True)
        pub = WorkerResultPublisher(REDIS, queue_factory=factory, timeout_seconds=0.2)
        try:
            with pytest.raises(TimeoutError):
                await pub.publish(QUIZ_GRADING_RESULT, {}, result_id="QUIZ_GRADING_RESULT.x")
            _wait_until(lambda: factory.queues[0].cancelled == 1)
        finally:
            pub.close()

    async def test_use_case_in_worker_thread_uses_publish_sync(self, publisher, factory):
        """Đúng hình runner gọi use case: ``asyncio.to_thread`` trong khi loop của runner vẫn chạy."""
        heartbeat = asyncio.create_task(asyncio.sleep(0.05))

        job_id = await asyncio.to_thread(_publish_sync, publisher)

        assert job_id == make_result_id(QUIZ_GRADING_RESULT, EVENT_ID)
        await heartbeat


class TestClose:
    def test_close_without_publishing_starts_nothing(self, factory):
        pub = WorkerResultPublisher(REDIS, queue_factory=factory)
        pub.close()
        pub.close()
        assert pub._thread is None
        assert factory.queues == []

    def test_close_closes_queue_stops_thread_and_rejects_new_results(self, factory):
        pub = WorkerResultPublisher(REDIS, queue_factory=factory, timeout_seconds=5)
        _publish_sync(pub)
        thread = pub._thread

        pub.close()
        pub.close()

        assert factory.queues[0].closed
        assert not thread.is_alive()
        with pytest.raises(RuntimeError, match="closed"):
            _publish_sync(pub)

    def test_close_lets_an_inflight_publish_finish(self):
        factory = QueueFactory(add_delay=0.2)
        pub = WorkerResultPublisher(REDIS, queue_factory=factory, timeout_seconds=5)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(_publish_sync, pub)
            while not factory.queues or not factory.queues[0].calls:
                time.sleep(0.01)
            pub.close()
            assert pending.result(timeout=5) == make_result_id(QUIZ_GRADING_RESULT, EVENT_ID)
        assert factory.queues[0].events == ["added", "closed"]


# ---------------------------------------------------------------------------
# Test double, DI và hợp đồng
# ---------------------------------------------------------------------------

def test_recording_publisher_validates_like_the_real_one():
    fake = RecordingResultPublisher()
    _publish_sync(fake)
    assert fake.envelopes[0]["kind"] == QUIZ_GRADING_RESULT
    with pytest.raises(InvalidWorkerResult):
        fake.publish_sync(QUIZ_GRADING_RESULT, {}, result_id="bad:id")

    failing = RecordingResultPublisher(fail_with=ConnectionError("redis down"))
    with pytest.raises(ConnectionError):
        _publish_sync(failing)
    assert failing.envelopes == []


def test_container_registers_publisher_with_worker_redis_settings(monkeypatch):
    from worker.config import WorkerSettings
    from worker.container.base import build_result_publisher
    from worker.container.enrichment import EnrichmentWorkerContainer

    monkeypatch.setenv("REDIS__HOST", "redis.internal")
    monkeypatch.setenv("REDIS__PORT", "6390")
    monkeypatch.setenv("REDIS__PASSWORD", "s3cret")
    settings = WorkerSettings()

    built = build_result_publisher(settings)
    assert isinstance(built, WorkerResultPublisher)
    assert built._connection == {"host": "redis.internal", "port": 6390, "db": 0, "password": "s3cret"}

    container = EnrichmentWorkerContainer(settings)
    assert isinstance(container.result_publisher, WorkerResultPublisher)
    container.close()
    assert container.result_publisher._closed


def test_kinds_match_contract_schema():
    schema_path = (
        Path(__file__).resolve().parents[3]
        / "contracts" / "integration-events" / "v1" / "worker-result-envelope.json"
    )
    if not schema_path.exists():
        pytest.skip("contracts/ không có trong image này")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert sorted(schema["properties"]["kind"]["enum"]) == sorted(WORKER_RESULT_KINDS)
    assert schema["properties"]["result_id"]["pattern"] == f"^{RESULT_ID_PATTERN.pattern}$"
