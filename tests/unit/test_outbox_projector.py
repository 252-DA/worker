from unittest.mock import call

from document_chunk.domain.entities.document import DocumentType
from document_chunk.domain.outbox_events import OutboxEventType, OutboxRelayEvent
from document_chunk.domain.ports.metadata_store import IngestionStatus
from document_chunk.shared.result import Err, Ok

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


def _projector(mock_metadata_store, mock_graph_store, mock_vector_store):
    return OutboxProjector(
        metadata_store=mock_metadata_store,
        graph_store=mock_graph_store,
        vector_store=mock_vector_store,
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

        result = _projector(
            mock_metadata_store,
            mock_graph_store,
            mock_vector_store,
        ).process(event)

        assert result.is_ok()
        mock_graph_store.upsert_concept_graph.assert_called_once()
        mock_metadata_store.update_document_status.assert_called_once_with(
            "doc-001",
            IngestionStatus.ENRICHED,
        )
        assert mock_metadata_store.method_calls == [
            call.get_document_context("doc-001"),
            call.update_document_status("doc-001", IngestionStatus.ENRICHED),
        ]

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
        assert mock_metadata_store.method_calls == []

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
        assert mock_metadata_store.method_calls == []

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
