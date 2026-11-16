"""
DOCUMENT_DELETED qua OutboxProjector + PostgresMetadataStore thật: chunks của tài
liệu mà core-api đã soft-delete bị ẩn (``chunks.deleted_at``), Qdrant/Neo4j được dọn.

Chạy khi có ``TEST_SQL_DSN`` trỏ tới database đã áp dụng migration; bỏ qua nếu không.
``TEST_SQL_ADMIN_DSN`` (mặc định trùng ``TEST_SQL_DSN``) dựng dữ liệu của core-api và
đọc kết quả, để chạy được với role worker chỉ ghi bảng AI.
"""
import os
from contextlib import contextmanager
from unittest.mock import MagicMock

import psycopg
import pytest
from document_chunk.adapters.metadata.postgres_metadata_store import PostgresMetadataStore
from document_chunk.domain.outbox_events import OutboxEventType, OutboxRelayEvent
from document_chunk.infrastructure.config import SqlConfig
from document_chunk.shared.result import Ok

from tests.support import RecordingResultPublisher
from worker.services.outbox_projector import OutboxProjector

_DSN = os.getenv("TEST_SQL_DSN")
_ADMIN_DSN = os.getenv("TEST_SQL_ADMIN_DSN") or _DSN
pytestmark = pytest.mark.skipif(not _DSN, reason="cần TEST_SQL_DSN")

COURSE = "01900000-0000-7000-8000-00000000d001"
DOC = "01900000-0000-7000-8000-00000000d010"
CHUNKS = ("01900000-0000-7000-8000-00000000d011", "01900000-0000-7000-8000-00000000d012")


@contextmanager
def _admin():
    with psycopg.connect(_ADMIN_DSN) as conn, conn.cursor() as cur:
        yield cur  # commit khi thoát khối


@pytest.fixture
def store():
    s = PostgresMetadataStore(SqlConfig(dsn=_DSN))
    with _admin() as cur:
        cur.execute("DELETE FROM chunks WHERE document_id = %s;", (DOC,))
        cur.execute("DELETE FROM documents WHERE document_id = %s;", (DOC,))
        cur.execute(
            "INSERT INTO courses (course_id, code, name) VALUES (%s, 'PRJ1', 'Projector IT') "
            "ON CONFLICT DO NOTHING;", (COURSE,))
        cur.execute(
            "INSERT INTO documents (document_id, course_id, title, file_path, status) "
            "VALUES (%s, %s, 't', 'documents/x.pdf', 'INDEXED');", (DOC, COURSE))
        for index, chunk in enumerate(CHUNKS):
            cur.execute(
                "INSERT INTO chunks (chunk_id, document_id, course_id, content, sort_order) "
                "VALUES (%s, %s, %s, 'nội dung', %s);", (chunk, DOC, COURSE, index))
    yield s
    s.close()


def _live_chunks(store):
    with _admin() as cur:
        cur.execute(
            "SELECT count(*) FROM chunks WHERE document_id = %s AND deleted_at IS NULL;", (DOC,))
        return cur.fetchone()[0]


def _event():
    return OutboxRelayEvent.from_mapping({
        "event_id": "event-del", "event_type": OutboxEventType.DOCUMENT_DELETED.value,
        "aggregate_type": "document", "aggregate_id": DOC, "payload": {"document_id": DOC},
    })


def test_document_deleted_event_hides_chunks_and_cleans_derived_stores(store):
    vector, graph = MagicMock(), MagicMock()
    vector.delete_by_document.return_value = Ok(None)
    graph.delete_document.return_value = Ok(None)
    projector = OutboxProjector(store, graph, vector, RecordingResultPublisher())
    with _admin() as cur:
        # core-api soft-delete tài liệu cùng transaction với việc phát sự kiện.
        cur.execute("UPDATE documents SET deleted_at = now() WHERE document_id = %s;", (DOC,))
    assert _live_chunks(store) == 2

    assert projector.process(_event()).is_ok()
    assert _live_chunks(store) == 0
    vector.delete_by_document.assert_called_once_with(DOC)
    graph.delete_document.assert_called_once_with(DOC)

    # BullMQ chạy lại job: vẫn thành công, không đổi gì.
    assert projector.process(_event()).is_ok()
    assert _live_chunks(store) == 0


def test_event_for_a_live_document_does_not_hide_its_chunks(store):
    vector, graph = MagicMock(), MagicMock()
    vector.delete_by_document.return_value = Ok(None)
    graph.delete_document.return_value = Ok(None)

    assert OutboxProjector(store, graph, vector, RecordingResultPublisher()).process(_event()).is_ok()

    assert _live_chunks(store) == 2
