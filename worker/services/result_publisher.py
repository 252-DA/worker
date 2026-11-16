"""
Đẩy kết quả của worker về core-api qua hàng đợi BullMQ ``worker_results``.

Worker tính xong thì gửi kết quả; core-api là consumer duy nhất
(core-api/src/worker-results) và là nơi duy nhất ghi bảng nghiệp vụ. Hợp đồng
envelope: contracts/integration-events/v1/worker-result-envelope.json; kế hoạch:
docs/ke-hoach-ranh-gioi-du-lieu.md mục 4.

Hai cách gọi, dùng chung một Queue:

- runner (coroutine trên event loop của BullMQ Worker)::

      await container.result_publisher.publish(kind, payload, result_id=...)

- use case đồng bộ chạy trong ``asyncio.to_thread``::

      self._publisher.publish_sync(kind, payload, result_id=...)

Cả hai chỉ trả về khi Redis đã nhận job và ném lỗi nếu không: job gốc phải
retry chứ không được hoàn tất khi kết quả chưa đi.

Queue sống trên một event loop riêng trong thread nền, không trên loop của
runner: client ``redis.asyncio`` gắn với loop tạo ra nó, use case trong thread
không có loop, còn ``container.close()`` là hàm đồng bộ — loop riêng phục vụ
được cả ba mà không phụ thuộc container được dựng ở đâu.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import re
import threading
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any

from document_chunk.shared.logger import get_logger

logger = get_logger(__name__)

WORKER_RESULTS_QUEUE_NAME = "worker_results"

QUIZ_GRADING_RESULT = "QUIZ_GRADING_RESULT"
MICRO_CONTENT_GRADING_RESULT = "MICRO_CONTENT_GRADING_RESULT"
ESSAY_GRADING_RESULT = "ESSAY_GRADING_RESULT"
CONTENT_GENERATION_RESULT = "CONTENT_GENERATION_RESULT"
ENRICHMENT_CONTENT_RESULT = "ENRICHMENT_CONTENT_RESULT"
CURRICULUM_IMPORT_RESULT = "CURRICULUM_IMPORT_RESULT"
MICRO_CONTENT_VIDEO_RESULT = "MICRO_CONTENT_VIDEO_RESULT"
DOCUMENT_PROCESSING_RESULT = "DOCUMENT_PROCESSING_RESULT"

# Khớp WORKER_RESULT_KINDS ở core-api/src/worker-results/worker-results.constants.ts
# và enum `kind` của schema envelope; test hai phía đối chiếu với schema.
WORKER_RESULT_KINDS: tuple[str, ...] = (
    QUIZ_GRADING_RESULT,
    MICRO_CONTENT_GRADING_RESULT,
    ESSAY_GRADING_RESULT,
    CONTENT_GENERATION_RESULT,
    ENRICHMENT_CONTENT_RESULT,
    CURRICULUM_IMPORT_RESULT,
    MICRO_CONTENT_VIDEO_RESULT,
    DOCUMENT_PROCESSING_RESULT,
)

# Tuỳ chọn job do producer đặt; worker Node của core-api làm theo.
DEFAULT_JOB_OPTIONS: dict[str, Any] = {
    # Lùi 5s, 10s, … ~21 phút: tổng ~43 phút, đủ qua một lần core-api
    # restart/deploy hay Postgres chập chờn mà kết quả không rơi vào failed.
    "attempts": 10,
    "backoff": {"type": "exponential", "delay": 5_000},
    # Giữ job xong 1 ngày để BullMQ còn chặn trùng khi worker gửi lại và để xem
    # lại đã gửi gì; payload có thể lớn (đề cương, học liệu) nên có trần số lượng.
    "removeOnComplete": {"age": 24 * 3600, "count": 1000},
    # Job hỏng là kết quả chưa được ghi: giữ 14 ngày, không trần số lượng, để
    # điều tra và cho chạy lại.
    "removeOnFail": {"age": 14 * 24 * 3600},
}
DEFAULT_TIMEOUT_SECONDS = 30.0

# result_id = jobId. BullMQ Node 5.x từ chối custom jobId là số nguyên hoặc chứa
# ':' (trừ dạng 3 phần của repeat job cũ); bullmq Python 3.0.4 không kiểm gì,
# nhưng id phải hợp lệ ở cả hai phía. Bắt đầu bằng chữ cái nên không bao giờ là
# số. Khớp pattern trong schema envelope và RESULT_ID_PATTERN bên core-api.
RESULT_ID_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,199}")
# Mỗi phần không có '.': '.' là dấu nối, nên hai bộ phần khác nhau không thể
# cho cùng một id. UUID lọt.
_ID_PART = re.compile(r"[A-Za-z0-9_-]+")
_MAX_SOURCE_EVENT_ID = 200


class InvalidWorkerResult(ValueError):
    """Kết quả sai hợp đồng: lỗi lập trình, gửi lại cũng không khá hơn."""


def make_result_id(kind: str, source_id: str, attempt: int | str | None = None) -> str:
    """
    result_id tất định: ``<KIND>.<source_id>`` hoặc ``<KIND>.<source_id>.<attempt>``.

    ``source_id`` là định danh của việc đã làm: tốt nhất là ``event_id`` outbox
    của lệnh (mỗi lần yêu cầu lại là một sự kiện mới), không có thì id dòng
    nguồn (attempt_id, import_id, document_id…).

    ``attempt`` là lần làm lại *về nghiệp vụ* của cùng nguồn (lần chấm lại thứ
    n, revision tài liệu) khi ``source_id`` chưa tự phân biệt. KHÔNG truyền
    ``attemptsMade`` của job BullMQ gốc: job gốc retry sau khi đã gửi phải sinh
    lại đúng id cũ thì BullMQ mới chặn trùng.
    """
    if kind not in WORKER_RESULT_KINDS:
        raise InvalidWorkerResult(f"unknown worker result kind: {kind!r}")
    parts = [kind, _id_part("source_id", source_id)]
    if attempt is not None:
        if isinstance(attempt, bool) or not isinstance(attempt, (int, str)):
            raise InvalidWorkerResult(f"attempt must be int or str, got {type(attempt).__name__}")
        if isinstance(attempt, int) and attempt < 0:
            raise InvalidWorkerResult(f"attempt must be >= 0, got {attempt}")
        parts.append(_id_part("attempt", str(attempt)))
    result_id = ".".join(parts)
    if not RESULT_ID_PATTERN.fullmatch(result_id):
        raise InvalidWorkerResult(f"result_id too long ({len(result_id)} > 200)")
    return result_id


def _id_part(name: str, value: object) -> str:
    if not isinstance(value, str) or not _ID_PART.fullmatch(value):
        raise InvalidWorkerResult(
            f"{name} must be non-empty and contain only letters, digits, '_' or '-': {value!r}"
        )
    return value


def build_envelope(
    kind: str,
    payload: Mapping[str, Any],
    *,
    result_id: str,
    source_event_id: str | None = None,
    occurred_at: datetime | None = None,
) -> dict[str, Any]:
    """Envelope đúng schema ``worker-result-envelope.json``; sai thì ném ``InvalidWorkerResult``."""
    if kind not in WORKER_RESULT_KINDS:
        raise InvalidWorkerResult(f"unknown worker result kind: {kind!r}")
    if not isinstance(result_id, str) or not RESULT_ID_PATTERN.fullmatch(result_id):
        raise InvalidWorkerResult(f"invalid result_id {result_id!r}; build it with make_result_id()")
    if source_event_id is not None and (
        not isinstance(source_event_id, str)
        or not source_event_id
        or len(source_event_id) > _MAX_SOURCE_EVENT_ID
    ):
        raise InvalidWorkerResult(f"source_event_id must be a non-empty string or None: {source_event_id!r}")
    if not isinstance(payload, Mapping):
        raise InvalidWorkerResult(f"payload must be a JSON object, got {type(payload).__name__}")
    try:
        # Khứ hồi qua JSON: cái gửi đi đúng là cái đã kiểm, và tách khỏi object
        # của người gọi (việc gửi chạy ở thread khác).
        payload_json = json.loads(json.dumps(payload, allow_nan=False, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise InvalidWorkerResult(f"payload is not JSON-serializable: {exc}") from exc

    when = occurred_at or datetime.now(timezone.utc)
    if when.tzinfo is None or when.utcoffset() is None:
        raise InvalidWorkerResult("occurred_at must be timezone-aware")
    occurred = when.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    return {
        "result_id": result_id,
        "kind": kind,
        "source_event_id": source_event_id,
        "occurred_at": occurred,
        "payload": payload_json,
    }


def _default_queue_factory(name: str, opts: dict[str, Any]):
    from bullmq import Queue

    return Queue(name, opts)


class WorkerResultPublisher:
    """
    Producer của ``worker_results``. An toàn khi gọi đồng thời từ nhiều thread
    và từ runner async; chỉ tạo thread và kết nối Redis ở lần gửi đầu tiên.
    """

    def __init__(
        self,
        connection: dict[str, Any] | str,
        *,
        queue_name: str = WORKER_RESULTS_QUEUE_NAME,
        job_options: Mapping[str, Any] | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        queue_factory: Callable[[str, dict[str, Any]], Any] = _default_queue_factory,
    ) -> None:
        self._connection = connection
        self._queue_name = queue_name
        self._job_options = dict(DEFAULT_JOB_OPTIONS if job_options is None else job_options)
        self._timeout = timeout_seconds
        self._queue_factory = queue_factory
        self._queue = None  # chỉ chạm tới trên loop của publisher
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._inflight: set[concurrent.futures.Future] = set()
        self._closed = False
        self._lock = threading.Lock()

    async def publish(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        result_id: str,
        source_event_id: str | None = None,
        occurred_at: datetime | None = None,
    ) -> str:
        """Gửi từ coroutine; trả jobId sau khi Redis đã nhận."""
        envelope = build_envelope(
            kind, payload, result_id=result_id,
            source_event_id=source_event_id, occurred_at=occurred_at,
        )
        future = self._submit(envelope)
        try:
            job_id = await asyncio.wait_for(asyncio.wrap_future(future), timeout=self._timeout)
        except asyncio.TimeoutError as exc:
            raise self._timeout_error(result_id) from exc
        self._log_published(envelope, job_id)
        return job_id

    def publish_sync(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        result_id: str,
        source_event_id: str | None = None,
        occurred_at: datetime | None = None,
    ) -> str:
        """Gửi từ code đồng bộ chạy trong thread (use case); chặn tới khi Redis đã nhận."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            # Chặn loop của runner thì BullMQ không gia hạn được lock của job gốc.
            raise RuntimeError(
                "publish_sync() would block the running event loop; use await publish()"
            )
        envelope = build_envelope(
            kind, payload, result_id=result_id,
            source_event_id=source_event_id, occurred_at=occurred_at,
        )
        future = self._submit(envelope)
        try:
            job_id = future.result(timeout=self._timeout)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise self._timeout_error(result_id) from exc
        except concurrent.futures.CancelledError as exc:
            raise RuntimeError(f"worker_results: publishing {result_id} was cancelled") from exc
        self._log_published(envelope, job_id)
        return job_id

    def close(self) -> None:
        """Đóng Queue và dừng thread nền; gọi được nhiều lần (container.close())."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            loop, thread, inflight = self._loop, self._thread, set(self._inflight)
        if loop is None or thread is None:
            return
        # Lần gửi đang dở là kết quả đã tính xong: cho nó chạy hết rồi mới đóng.
        concurrent.futures.wait(inflight, timeout=self._timeout)
        try:
            asyncio.run_coroutine_threadsafe(self._close_queue(), loop).result(timeout=self._timeout)
        except Exception as exc:  # noqa: BLE001 -- đang tắt, chỉ ghi lại
            logger.warning("worker_results.close_failed", error=str(exc))
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=self._timeout)

    def _submit(self, envelope: dict[str, Any]) -> concurrent.futures.Future:
        with self._lock:
            if self._closed:
                raise RuntimeError("WorkerResultPublisher is closed")
            if self._loop is None:
                loop = asyncio.new_event_loop()
                thread = threading.Thread(
                    target=_run_loop, args=(loop,), name="worker-results-publisher", daemon=True
                )
                thread.start()
                self._loop, self._thread = loop, thread
            # Nộp trong lock: close() đã đặt _closed thì không còn lần gửi mới lọt vào.
            future = asyncio.run_coroutine_threadsafe(self._add(envelope), self._loop)
            self._inflight.add(future)
        future.add_done_callback(self._forget)
        return future

    def _forget(self, future: concurrent.futures.Future) -> None:
        with self._lock:
            self._inflight.discard(future)

    async def _add(self, envelope: dict[str, Any]) -> str:
        if self._queue is None:
            self._queue = self._queue_factory(self._queue_name, {"connection": self._connection})
        job = await self._queue.add(
            envelope["kind"],
            envelope,
            {**self._job_options, "jobId": envelope["result_id"]},
        )
        return str(job.id)

    async def _close_queue(self) -> None:
        queue, self._queue = self._queue, None
        if queue is not None:
            await queue.close()

    def _timeout_error(self, result_id: str) -> TimeoutError:
        # Có thể Redis đã nhận mà trả lời chậm: job gốc retry sẽ gửi lại cùng
        # jobId và BullMQ bỏ bản trùng.
        return TimeoutError(
            f"worker_results: Redis did not acknowledge {result_id} within {self._timeout}s"
        )

    @staticmethod
    def _log_published(envelope: dict[str, Any], job_id: str) -> None:
        logger.info(
            "worker_results.published",
            kind=envelope["kind"],
            result_id=envelope["result_id"],
            job_id=job_id,
            source_event_id=envelope["source_event_id"],
        )


def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    try:
        loop.run_forever()
    finally:
        # Còn task treo (quá timeout lúc close) thì huỷ cho gọn rồi đóng loop.
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
