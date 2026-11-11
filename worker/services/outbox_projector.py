from document_chunk.domain.entities.document import DocumentType
from document_chunk.domain.exceptions import DocumentStaleError
from document_chunk.domain.outbox_events import (
    OutboxEventType,
    OutboxRelayEvent,
    normalize_outbox_event_type,
)
from document_chunk.domain.ports.graph_store import (
    GraphChunk,
    GraphChunkConcept,
    GraphConcept,
    GraphDocument,
    IGraphStore,
)
from document_chunk.domain.ports.metadata_store import IMetadataStore, IngestionStatus
from document_chunk.domain.ports.vector_store import IVectorStore
from document_chunk.shared.logger import get_logger
from document_chunk.shared.result import Err, Ok, Result

from worker.services.document_reporter import (
    STAGE_ENRICHMENT,
    PublishingDocumentStatusReporter,
)

logger = get_logger(__name__)


class OutboxProjector:
    """Apply one outbox relay event to the appropriate derived store.

    PostgreSQL outbox ownership belongs to the Core API relay. This service only
    consumes durable BullMQ jobs, so it must never fetch or mutate outbox rows.
    BullMQ owns retries for projection failures.

    DOCUMENT_DELETED dọn cả ba kho dẫn xuất của tài liệu: ``chunks`` (soft-delete,
    bảng AI của worker), Qdrant và Neo4j. ``documents`` do core-api xoá.

    Document status (GENERATED_DRAFT after the concept graph) goes to core-api
    as DOCUMENT_PROCESSING_RESULT; this service never writes ``documents``.
    """

    def __init__(
        self,
        metadata_store: IMetadataStore,
        graph_store: IGraphStore,
        vector_store: IVectorStore,
        result_publisher,
    ) -> None:
        self._metadata_store = metadata_store
        self._graph_store = graph_store
        self._vector_store = vector_store
        self._result_publisher = result_publisher

    def process(self, event: OutboxRelayEvent) -> Result[None, Exception]:
        try:
            event_type = normalize_outbox_event_type(event.event_type)

            if event_type == OutboxEventType.HEADING_GRAPH_PROJECT:
                return self._project_headings(event)
            if event_type == OutboxEventType.CONCEPT_GRAPH_PROJECT:
                return self._project_concepts(event)
            if event_type == OutboxEventType.DOCUMENT_DELETED:
                return self._delete_document(event)
            if event_type == OutboxEventType.LESSON_PUBLISHED:
                logger.info(
                    "outbox_projector.lesson_published_noop",
                    event_id=event.event_id,
                    aggregate_id=event.aggregate_id,
                )
                return Ok(None)

            return Err(
                ValueError(
                    f"outbox event {event_type.value!r} is not supported by the projection worker"
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            return Err(ValueError(f"invalid outbox relay event {event.event_id}: {exc}"))

    def _project_headings(self, event: OutboxRelayEvent) -> Result[None, Exception]:
        payload = event.payload
        document_id = self._required_string(payload, "document_id")
        ensure_result = self._ensure_document_still_exists(document_id)
        if ensure_result.is_err():
            return ensure_result
        if ensure_result.unwrap() is False:
            return Ok(None)

        chunks_payload = payload.get("chunks", [])
        if not isinstance(chunks_payload, list):
            return Err(ValueError("heading projection payload.chunks must be a list"))

        try:
            document = GraphDocument(
                document_id=document_id,
                document_name=self._required_string(payload, "document_name"),
                doc_type=DocumentType(self._required_string(payload, "doc_type").lower()),
            )
            chunks = [
                GraphChunk(
                    chunk_id=self._required_string(item, "chunk_id"),
                    chunk_index=int(item["chunk_index"]),
                    heading_path=tuple(item.get("heading_path", [])),
                    page_number=item.get("page_number"),
                    language=item.get("language"),
                )
                for item in chunks_payload
            ]
        except (KeyError, TypeError, ValueError) as exc:
            return Err(ValueError(f"invalid heading projection payload: {exc}"))

        return self._graph_store.upsert_heading_graph(
            document=document,
            course_id=self._optional_string(payload, "course_id"),
            owner_id=self._optional_string(payload, "owner_id"),
            chunks=chunks,
        )

    def _delete_document(self, event: OutboxRelayEvent) -> Result[None, Exception]:
        try:
            document_id = self._required_string(event.payload, "document_id")
        except (TypeError, ValueError) as exc:
            return Err(exc)

        # Postgres trước: chunks là bảng AI do worker sở hữu, core-api chỉ
        # soft-delete documents. Làm trước để việc ẩn chunks không phụ thuộc
        # Qdrant/Neo4j đang sống; mọi bước idempotent nên BullMQ chạy lại cả job
        # khi một bước sau lỗi cũng không sao.
        chunks_result = self._metadata_store.soft_delete_document_chunks(document_id)
        if chunks_result.is_err():
            return chunks_result
        logger.info(
            "outbox_projector.document_chunks_soft_deleted",
            document_id=document_id,
            chunks=chunks_result.unwrap(),
        )

        vector_result = self._vector_store.delete_by_document(document_id)
        if vector_result.is_err():
            return vector_result

        return self._graph_store.delete_document(document_id)

    def _project_concepts(self, event: OutboxRelayEvent) -> Result[None, Exception]:
        payload = event.payload
        try:
            document_id = self._required_string(payload, "document_id")
        except (TypeError, ValueError) as exc:
            return Err(exc)

        ensure_result = self._ensure_document_still_exists(document_id)
        if ensure_result.is_err():
            return ensure_result
        if ensure_result.unwrap() is False:
            return Ok(None)

        concepts_payload = payload.get("concepts", [])
        mentions_payload = payload.get("mentions", [])
        # Sự kiện ghi trước khi có hai trường này: lượt 0, lần chạy 1.
        generation = payload.get("processing_generation", 0)
        attempt = payload.get("enrichment_attempt", 1)
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            return Err(ValueError("concept projection payload.processing_generation must be an integer >= 0"))
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            return Err(ValueError("concept projection payload.enrichment_attempt must be an integer >= 1"))
        if not isinstance(concepts_payload, list):
            return Err(ValueError("concept projection payload.concepts must be a list"))
        if not isinstance(mentions_payload, list):
            return Err(ValueError("concept projection payload.mentions must be a list"))

        try:
            concepts = [
                GraphConcept(
                    concept_id=self._required_string(item, "concept_id"),
                    name=self._required_string(item, "name"),
                    canonical_name=self._required_string(item, "canonical_name"),
                    slug=self._required_string(item, "slug"),
                    category=str(item.get("category") or "other"),
                    language=self._optional_string(item, "language"),
                    domain=self._optional_string(item, "domain"),
                )
                for item in concepts_payload
            ]
            mentions = [
                GraphChunkConcept(
                    chunk_id=self._required_string(item, "chunk_id"),
                    concept_id=self._required_string(item, "concept_id"),
                    confidence=float(item.get("confidence", 1.0)),
                    source=str(item.get("source") or "heading"),
                )
                for item in mentions_payload
            ]
        except (KeyError, TypeError, ValueError) as exc:
            return Err(ValueError(f"invalid concept projection payload: {exc}"))

        upsert_result = self._graph_store.upsert_concept_graph(
            document_id=document_id,
            concepts=concepts,
            mentions=mentions,
        )
        if upsert_result.is_err():
            return upsert_result

        reporter = PublishingDocumentStatusReporter(
            self._result_publisher,
            self._metadata_store,
            stage=STAGE_ENRICHMENT,
            processing_generation=generation,
            attempt=attempt,
        )
        status_result = reporter.report_status(document_id, IngestionStatus.ENRICHED)
        if status_result.is_err() and isinstance(status_result.error, DocumentStaleError):
            # Tài liệu bị xoá sau khi chiếu graph: DOCUMENT_DELETED sẽ dọn graph.
            logger.info("outbox_projector.stale_document_skipped", document_id=document_id)
            return Ok(None)
        return status_result

    def _ensure_document_still_exists(
        self,
        document_id: str,
    ) -> Result[bool, Exception]:
        context_result = self._metadata_store.get_document_context(document_id)
        if context_result.is_err():
            return Err(context_result.error)

        if context_result.unwrap() is not None:
            return Ok(True)

        logger.info(
            "outbox_projector.stale_document_skipped",
            document_id=document_id,
        )
        return Ok(False)

    @staticmethod
    def _required_string(payload: dict, key: str) -> str:
        if not isinstance(payload, dict):
            raise TypeError("payload must be an object")

        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"payload.{key} must be a non-empty string")
        return value

    @staticmethod
    def _optional_string(payload: dict, key: str) -> str | None:
        value = payload.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(f"payload.{key} must be a string or null")
        return value
