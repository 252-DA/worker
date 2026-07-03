from document_chunk.domain.ports.graph_store import (
    GraphChunk,
    GraphChunkConcept,
    GraphConcept,
    IGraphStore,
)
from document_chunk.domain.ports.metadata_store import IMetadataStore, IngestionStatus, OutboxEvent
from document_chunk.domain.ports.vector_store import IVectorStore
from document_chunk.shared.logger import get_logger
from document_chunk.shared.result import Err, Ok, Result

logger = get_logger(__name__)

_EVENT_HEADING_GRAPH_PROJECT = "heading_graph_project"
_EVENT_CONCEPT_GRAPH_PROJECT = "concept_graph_project"
_EVENT_DOCUMENT_DELETED = "document_deleted"


class OutboxProjector:
    def __init__(
        self,
        metadata_store: IMetadataStore,
        graph_store: IGraphStore,
        vector_store: IVectorStore,
    ) -> None:
        self._metadata_store = metadata_store
        self._graph_store = graph_store
        self._vector_store = vector_store

    def run_once(self, limit: int = 100) -> Result[int, Exception]:
        pending_result = self._metadata_store.fetch_pending_outbox(limit=limit)
        if pending_result.is_err():
            return pending_result

        processed = 0
        for event in pending_result.unwrap():
            result = self._process_event(event)
            if result.is_err():
                logger.error("outbox_projector.event_failed", event_id=event.id, error=str(result.error))
            else:
                processed += 1

        return Ok(processed)

    def _process_event(self, event: OutboxEvent) -> Result[None, Exception]:
        if event.event_type == _EVENT_HEADING_GRAPH_PROJECT:
            return self._project_headings(event)
        if event.event_type == _EVENT_CONCEPT_GRAPH_PROJECT:
            return self._project_concepts(event)
        if event.event_type == _EVENT_DOCUMENT_DELETED:
            return self._delete_document(event)

        done_result = self._metadata_store.mark_outbox_done(event.id)
        if done_result.is_err():
            return done_result
        return Ok(None)

    def _project_headings(self, event: OutboxEvent) -> Result[None, Exception]:
        payload = event.payload
        ensure_result = self._ensure_document_still_exists(event, payload["document_id"])
        if ensure_result.is_err():
            return ensure_result
        if ensure_result.unwrap() is False:
            return Ok(None)

        chunks_payload = payload.get("chunks", [])
        chunks = [
            GraphChunk(
                chunk_id=item["chunk_id"],
                chunk_index=item["chunk_index"],
                heading_path=tuple(item.get("heading_path", [])),
            )
            for item in chunks_payload
        ]

        upsert_result = self._graph_store.upsert_heading_graph(
            document_id=payload["document_id"],
            course_id=payload.get("course_id"),
            owner_id=payload.get("owner_id"),
            chunks=chunks,
        )
        if upsert_result.is_err():
            fail_result = self._metadata_store.mark_outbox_failed(
                event.id,
                str(upsert_result.error),
            )
            if fail_result.is_err():
                return fail_result
            return Err(upsert_result.error)

        done_result = self._metadata_store.mark_outbox_done(event.id)
        if done_result.is_err():
            return done_result
        return Ok(None)

    def _delete_document(self, event: OutboxEvent) -> Result[None, Exception]:
        document_id = event.payload["document_id"]

        vector_result = self._vector_store.delete_by_document(document_id)
        if vector_result.is_err():
            fail_result = self._metadata_store.mark_outbox_failed(event.id, str(vector_result.error))
            if fail_result.is_err():
                return fail_result
            return Err(vector_result.error)

        graph_result = self._graph_store.delete_document(document_id)
        if graph_result.is_err():
            fail_result = self._metadata_store.mark_outbox_failed(event.id, str(graph_result.error))
            if fail_result.is_err():
                return fail_result
            return Err(graph_result.error)

        done_result = self._metadata_store.mark_outbox_done(event.id)
        if done_result.is_err():
            return done_result
        return Ok(None)

    def _project_concepts(self, event: OutboxEvent) -> Result[None, Exception]:
        payload = event.payload
        document_id = payload["document_id"]
        ensure_result = self._ensure_document_still_exists(event, document_id)
        if ensure_result.is_err():
            return ensure_result
        if ensure_result.unwrap() is False:
            return Ok(None)

        concepts = [
            GraphConcept(
                concept_id=item["concept_id"],
                name=item["name"],
                canonical_name=item["canonical_name"],
                slug=item["slug"],
                category=item.get("category", "other"),
                language=item.get("language"),
                domain=item.get("domain"),
            )
            for item in payload.get("concepts", [])
        ]
        mentions = [
            GraphChunkConcept(
                chunk_id=item["chunk_id"],
                concept_id=item["concept_id"],
                confidence=float(item.get("confidence", 1.0)),
                source=item.get("source", "heading"),
            )
            for item in payload.get("mentions", [])
        ]

        upsert_result = self._graph_store.upsert_concept_graph(
            document_id=document_id,
            concepts=concepts,
            mentions=mentions,
        )
        if upsert_result.is_err():
            fail_result = self._metadata_store.mark_outbox_failed(event.id, str(upsert_result.error))
            if fail_result.is_err():
                return fail_result
            return Err(upsert_result.error)

        status_result = self._metadata_store.update_document_status(
            document_id,
            IngestionStatus.ENRICHED,
        )
        if status_result.is_err():
            return status_result

        done_result = self._metadata_store.mark_outbox_done(event.id)
        if done_result.is_err():
            return done_result
        return Ok(None)

    def _ensure_document_still_exists(
        self,
        event: OutboxEvent,
        document_id: str,
    ) -> Result[bool, Exception]:
        context_result = self._metadata_store.get_document_context(document_id)
        if context_result.is_err():
            return Err(context_result.error)

        if context_result.unwrap() is not None:
            return Ok(True)

        done_result = self._metadata_store.mark_outbox_done(event.id)
        if done_result.is_err():
            return done_result
        return Ok(False)
