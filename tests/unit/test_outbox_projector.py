from unittest.mock import call

from document_chunk.domain.entities.document import DocumentType
from document_chunk.domain.outbox_events import OutboxEventType, OutboxRelayEvent
from document_chunk.shared.result import Err, Ok

from tests.support import RecordingResultPublisher
from worker.services.outbox_projector import OutboxProjector


def _event(
    event_type: str,
    payload: dict,
    *,
    event_id: str = "event-001",
    aggregate_type: str = "document",
    aggregate_id: str = "doc-001",
) -> OutboxRelayEvent:
    return OutboxRelayEvent.from_mapping(
        {
            "event_id": event_id,
            "event_type": event_type,
            "aggregate_type": aggregate_type,
            "aggregate_id": aggregate_id,
            "payload": payload,
        }
    )


def _projector(mock_metadata_store, mock_graph_store, mock_vector_store, publisher=None):
    return OutboxProjector(
        metadata_store=mock_metadata_store,
        graph_store=mock_graph_store,
        vector_store=mock_vector_store,
        result_publisher=publisher if publisher is not None else RecordingResultPublisher(),
    )


class TestOutboxProjector:
    def test_projects_heading_event_with_graph_document(
        self,
        mock_metadata_store,
        mock_graph_store,
        mock_vector_store,
    ):
        event = _event(
            OutboxEventType.HEADING_GRAPH_PROJECT.value,
            {
                "document_id": "doc-001",
                "document_name": "lecture.pdf",
                "doc_type": "pdf",
                "course_id": "course-001",
                "owner_id": "owner-001",
                "chunks": [
                    {
                        "chunk_id": "chunk-001",
                        "chunk_index": 0,
                        "heading_path": ["Calculus", "Derivatives"],
                        "page_number": 2,
                        "language": "en",
                    }
                ],
            },
        )

        result = _projector(
            mock_metadata_store,
            mock_graph_store,
            mock_vector_store,
        ).process(event)

        assert result.is_ok()
        call = mock_graph_store.upsert_heading_graph.call_args
        document = call.kwargs["document"]
        assert document.document_id == "doc-001"
        assert document.document_name == "lecture.pdf"
        assert document.doc_type == DocumentType.PDF
        assert call.kwargs["chunks"][0].page_number == 2
        assert mock_metadata_store.method_calls == [
            call.get_document_context("doc-001"),
        ]

    def test_projects_legacy_lowercase_concept_event_and_marks_document_enriched(
        self,
        mock_metadata_store,
        mock_graph_store,
        mock_vector_store,
    ):
        event = _event(
            "concept_graph_project",
            {
                "document_id": "doc-001",
                "concepts": [
                    {
                        "concept_id": "concept-001",
                        "name": "Đạo hàm",
                        "canonical_name": "đạo hàm",
                        "slug": "dao-ham",
                        "category": "other",
                        "language": "vi",
                    }
                ],
                "mentions": [
                    {
                        "chunk_id": "chunk-001",
                        "concept_id": "concept-001",
                        "confidence": 1.0,
                        "source": "heading",
                    }
                ],
            },
        )

        publisher = RecordingResultPublisher()
        result = _projector(
            mock_metadata_store,
            mock_graph_store,
            mock_vector_store,
            publisher,
        ).process(event)

        assert result.is_ok()
        mock_graph_store.upsert_concept_graph.assert_called_once()
        # Chỉ đọc documents (kiểm còn tồn tại); trạng thái đi về core-api.
        assert mock_metadata_store.method_calls == [
            call.get_document_context("doc-001"),
            call.get_document_context("doc-001"),
        ]
        [envelope] = publisher.envelopes
        assert envelope["kind"] == "DOCUMENT_PROCESSING_RESULT"
        # Sự kiện cũ chưa có lượt/lần chạy: lượt 0, lần 1.
        assert envelope["payload"] == {
            "document_id": "doc-001",
            "status": {
                "value": "GENERATED_DRAFT",
                "stage": "enrichment",
                "processing_generation": 0,
                "attempt": 1,
            },
        }

    def test_concept_event_reports_generated_draft_for_its_own_run(
        self,
        mock_metadata_store,
        mock_graph_store,
        mock_vector_store,
    ):
        event = _event(
            OutboxEventType.CONCEPT_GRAPH_PROJECT.value,
            {
                "document_id": "doc-001",
                "processing_generation": 5,
                "enrichment_attempt": 3,
                "concepts": [],
                "mentions": [],
            },
        )
        publisher = RecordingResultPublisher()

        result = _projector(
            mock_metadata_store, mock_graph_store, mock_vector_store, publisher
        ).process(event)

        assert result.is_ok()
        [envelope] = publisher.envelopes
        assert envelope["payload"]["status"]["processing_generation"] == 5
        assert envelope["payload"]["status"]["attempt"] == 3
        # Sự kiện outbox relay lại (at-least-once) sinh cùng id: BullMQ bỏ bản trùng.
        assert envelope["result_id"] == (
            "DOCUMENT_PROCESSING_RESULT.doc-001.g5-enrichment3-GENERATED_DRAFT"
        )
        mock_metadata_store.update_document_status.assert_not_called()

    def test_status_publish_failure_fails_the_projection_for_retry(
        self,
        mock_metadata_store,
        mock_graph_store,
        mock_vector_store,
    ):
        event = _event(
            OutboxEventType.CONCEPT_GRAPH_PROJECT.value,
            {"document_id": "doc-001", "concepts": [], "mentions": []},
        )
        failure = ConnectionError("redis down")

        result = _projector(
            mock_metadata_store,
            mock_graph_store,
            mock_vector_store,
            RecordingResultPublisher(fail_with=failure),
        ).process(event)

        assert result.is_err()
        assert result.error is failure

    def test_document_deleted_after_projection_is_not_an_error(
        self,
        mock_metadata_store,
        mock_graph_store,
        mock_vector_store,
    ):
        mock_metadata_store.get_document_context.side_effect = [
            Ok(object()),
            Ok(None),
        ]
        event = _event(
            OutboxEventType.CONCEPT_GRAPH_PROJECT.value,
            {"document_id": "doc-001", "concepts": [], "mentions": []},
        )
        publisher = RecordingResultPublisher()

        result = _projector(
            mock_metadata_store, mock_graph_store, mock_vector_store, publisher
        ).process(event)

        assert result.is_ok()
        assert publisher.envelopes == []

    def test_malformed_status_fields_are_rejected(
        self,
        mock_metadata_store,
        mock_graph_store,
        mock_vector_store,
    ):
        event = _event(
            OutboxEventType.CONCEPT_GRAPH_PROJECT.value,
            {"document_id": "doc-001", "processing_generation": "5", "concepts": [], "mentions": []},
        )

        result = _projector(mock_metadata_store, mock_graph_store, mock_vector_store).process(event)

        assert result.is_err()
        assert isinstance(result.error, ValueError)
        mock_graph_store.upsert_concept_graph.assert_not_called()

    def test_processes_document_deleted_event(
        self,
        mock_metadata_store,
        mock_graph_store,
        mock_vector_store,
    ):
        event = _event(
            OutboxEventType.DOCUMENT_DELETED.value,
            {"document_id": "doc-001"},
            event_id="event-002",
        )

        result = _projector(
            mock_metadata_store,
            mock_graph_store,
            mock_vector_store,
        ).process(event)

        assert result.is_ok()
        mock_vector_store.delete_by_document.assert_called_once_with("doc-001")
        mock_graph_store.delete_document.assert_called_once_with("doc-001")
        # chunks là bảng AI của worker: soft-delete ở đây; documents do core-api
        # xoá nên không có lời gọi nào khác lên metadata store.
        assert mock_metadata_store.method_calls == [call.soft_delete_document_chunks("doc-001")]

    def test_document_deleted_soft_deletes_chunks_before_the_derived_stores(
        self,
        mock_metadata_store,
        mock_graph_store,
        mock_vector_store,
    ):
        order = []
        mock_metadata_store.soft_delete_document_chunks.side_effect = (
            lambda _id: order.append("chunks") or Ok(2)
        )
        mock_vector_store.delete_by_document.side_effect = lambda _id: order.append("qdrant") or Ok(None)
        mock_graph_store.delete_document.side_effect = lambda _id: order.append("neo4j") or Ok(None)

        result = _projector(mock_metadata_store, mock_graph_store, mock_vector_store).process(
            _event(OutboxEventType.DOCUMENT_DELETED.value, {"document_id": "doc-001"})
        )

        assert result.is_ok()
        assert order == ["chunks", "qdrant", "neo4j"]

    def test_document_deleted_is_idempotent_when_chunks_are_already_hidden(
        self,
        mock_metadata_store,
        mock_graph_store,
        mock_vector_store,
    ):
        """Lần chạy lại (BullMQ retry sau khi Qdrant/Neo4j lỗi) ẩn 0 chunk và vẫn thành công."""
        mock_metadata_store.soft_delete_document_chunks.return_value = Ok(0)
        projector = _projector(mock_metadata_store, mock_graph_store, mock_vector_store)
        event = _event(OutboxEventType.DOCUMENT_DELETED.value, {"document_id": "doc-001"})

        assert projector.process(event).is_ok()
        assert projector.process(event).is_ok()
        assert mock_metadata_store.soft_delete_document_chunks.call_count == 2

    def test_chunk_soft_delete_failure_is_returned_and_stops_the_projection(
        self,
        mock_metadata_store,
        mock_graph_store,
        mock_vector_store,
    ):
        failure = RuntimeError("postgres unavailable")
        mock_metadata_store.soft_delete_document_chunks.return_value = Err(failure)

        result = _projector(mock_metadata_store, mock_graph_store, mock_vector_store).process(
            _event(OutboxEventType.DOCUMENT_DELETED.value, {"document_id": "doc-001"})
        )

        # BullMQ chạy lại cả job: chưa dọn gì ở Qdrant/Neo4j nên không mất gì.
        assert result.is_err()
        assert result.error is failure
        mock_vector_store.delete_by_document.assert_not_called()
        mock_graph_store.delete_document.assert_not_called()

    def test_projection_failure_is_returned_without_mutating_outbox(
        self,
        mock_metadata_store,
        mock_graph_store,
        mock_vector_store,
    ):
        failure = RuntimeError("qdrant unavailable")
        mock_vector_store.delete_by_document.return_value = Err(failure)
        event = _event(
            OutboxEventType.DOCUMENT_DELETED.value,
            {"document_id": "doc-001"},
        )

        result = _projector(
            mock_metadata_store,
            mock_graph_store,
            mock_vector_store,
        ).process(event)

        assert result.is_err()
        assert result.error is failure
        mock_graph_store.delete_document.assert_not_called()
        assert mock_metadata_store.method_calls == [call.soft_delete_document_chunks("doc-001")]

    def test_missing_document_skips_stale_projection(
        self,
        mock_metadata_store,
        mock_graph_store,
        mock_vector_store,
    ):
        mock_metadata_store.get_document_context.return_value = Ok(None)
        event = _event(
            OutboxEventType.CONCEPT_GRAPH_PROJECT.value,
            {"document_id": "doc-001", "concepts": [], "mentions": []},
        )

        result = _projector(
            mock_metadata_store,
            mock_graph_store,
            mock_vector_store,
        ).process(event)

        assert result.is_ok()
        mock_graph_store.upsert_concept_graph.assert_not_called()
        mock_metadata_store.update_document_status.assert_not_called()

    def test_lesson_published_is_explicit_noop(
        self,
        mock_metadata_store,
        mock_graph_store,
        mock_vector_store,
    ):
        event = _event(
            OutboxEventType.LESSON_PUBLISHED.value,
            {"lesson_id": "lesson-001", "course_id": "course-001"},
            aggregate_type="lesson",
            aggregate_id="lesson-001",
        )

        result = _projector(
            mock_metadata_store,
            mock_graph_store,
            mock_vector_store,
        ).process(event)

        assert result.is_ok()
        mock_graph_store.upsert_heading_graph.assert_not_called()
        mock_graph_store.upsert_concept_graph.assert_not_called()
        mock_graph_store.delete_document.assert_not_called()
        mock_vector_store.delete_by_document.assert_not_called()

    def test_malformed_heading_payload_fails(
        self,
        mock_metadata_store,
        mock_graph_store,
        mock_vector_store,
    ):
        event = _event(
            OutboxEventType.HEADING_GRAPH_PROJECT.value,
            {
                "document_id": "doc-001",
                "doc_type": "pdf",
                "chunks": [],
            },
        )

        result = _projector(
            mock_metadata_store,
            mock_graph_store,
            mock_vector_store,
        ).process(event)

        assert result.is_err()
        assert "document_name" in str(result.error)
        mock_graph_store.upsert_heading_graph.assert_not_called()
