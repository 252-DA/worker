from datetime import datetime, timezone

from document_chunk.domain.ports.metadata_store import IngestionStatus, OutboxEvent
from document_chunk.shared.result import Ok

from worker.services.outbox_projector import OutboxProjector


class TestOutboxProjector:
    def test_projects_concept_event_and_marks_document_enriched(
        self,
        mock_metadata_store,
        mock_graph_store,
        mock_vector_store,
    ):
        event = OutboxEvent(
            id="event-001",
            event_type="concept_graph_project",
            aggregate_id="doc-001",
            payload={
                "document_id": "doc-001",
                "concepts": [
                    {
                        "concept_id": "dao-ham",
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
                        "concept_id": "dao-ham",
                        "confidence": 1.0,
                        "source": "heading",
                    }
                ],
            },
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        mock_metadata_store.fetch_pending_outbox.return_value = Ok([event])

        projector = OutboxProjector(
            metadata_store=mock_metadata_store,
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
        )

        result = projector.run_once()

        assert result.is_ok()
        assert result.unwrap() == 1
        mock_graph_store.upsert_concept_graph.assert_called_once()
        mock_metadata_store.update_document_status.assert_called_with(
            "doc-001",
            IngestionStatus.ENRICHED,
        )
        mock_metadata_store.mark_outbox_done.assert_called_once_with("event-001")

    def test_processes_document_deleted_event(self, mock_metadata_store, mock_graph_store, mock_vector_store):
        event = OutboxEvent(
            id="event-002",
            event_type="document_deleted",
            aggregate_id="doc-001",
            payload={"document_id": "doc-001"},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        mock_metadata_store.fetch_pending_outbox.return_value = Ok([event])

        projector = OutboxProjector(
            metadata_store=mock_metadata_store,
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
        )

        result = projector.run_once()

        assert result.is_ok()
        assert result.unwrap() == 1
        mock_vector_store.delete_by_document.assert_called_once_with("doc-001")
        mock_graph_store.delete_document.assert_called_once_with("doc-001")
        mock_metadata_store.mark_outbox_done.assert_called_once_with("event-002")
