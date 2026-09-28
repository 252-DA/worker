"""
Shared fixtures cho worker test suite.
"""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.support import as_llm_client
from document_chunk.domain.entities.chunk import Chunk, ChunkMetadata
from document_chunk.domain.entities.document import Document, DocumentType, ElementType, ParsedDocument, Section
from document_chunk.domain.entities.embedding import Embedding
from document_chunk.domain.entities.search import SearchResult
from document_chunk.domain.ports.metadata_store import StoredChunkMetadata, StoredDocumentContext
from document_chunk.shared.result import Ok


# ---------------------------------------------------------------------------
# Domain entities
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_document(tmp_path: Path) -> Document:
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 sample")
    return Document(
        id="doc-001",
        name="test.pdf",
        path=pdf_file,
        doc_type=DocumentType.PDF,
        size_bytes=15,
        mime_type="application/pdf",
    )


@pytest.fixture
def sample_parsed_doc(sample_document: Document) -> ParsedDocument:
    return ParsedDocument(
        document=sample_document,
        sections=[
            Section(
                content="Introduction",
                element_type=ElementType.HEADING,
                heading="Introduction",
                heading_level=1,
                page_number=1,
            ),
            Section(
                content="This is the introduction paragraph with enough content.",
                element_type=ElementType.PARAGRAPH,
                page_number=1,
            ),
            Section(
                content="More content in the second paragraph here.",
                element_type=ElementType.PARAGRAPH,
                page_number=2,
            ),
        ],
        page_count=2,
        language="en",
    )


@pytest.fixture
def sample_chunk(sample_document: Document) -> Chunk:
    return Chunk(
        id="chunk-001",
        content="This is chunk content for testing purposes.",
        embedding_input="Introduction\n\nThis is chunk content for testing purposes.",
        content_hash="fake-hash-001",
        metadata=ChunkMetadata(
            document_id=sample_document.id,
            document_name=sample_document.name,
            document_type=DocumentType.PDF,
            chunk_index=0,
            heading_path=("Introduction",),
            heading_level=1,
            page_number=1,
            language="en",
        ),
    )


@pytest.fixture
def sample_embedding() -> Embedding:
    return Embedding(
        chunk_id="chunk-001",
        vector=tuple([0.1] * 1024),
        model="BAAI/bge-m3",
        dimension=1024,
    )


@pytest.fixture
def sample_search_result(sample_chunk: Chunk) -> SearchResult:
    return SearchResult(chunk=sample_chunk, score=0.95, rank=1)


@pytest.fixture
def sample_stored_chunk_metadata(sample_chunk: Chunk) -> StoredChunkMetadata:
    return StoredChunkMetadata(
        chunk_id=sample_chunk.id,
        document_id=sample_chunk.metadata.document_id,
        chunk_index=sample_chunk.metadata.chunk_index,
        heading_path=sample_chunk.metadata.heading_path,
        heading_level=sample_chunk.metadata.heading_level,
        page_number=sample_chunk.metadata.page_number,
        content_length=len(sample_chunk.content),
        language=sample_chunk.metadata.language,
        content_text=sample_chunk.content,
        embedding_input=sample_chunk.embedding_input,
    )


@pytest.fixture
def sample_document_context(sample_document: Document) -> StoredDocumentContext:
    return StoredDocumentContext(
        document_id=sample_document.id,
        course_id="course-001",
        owner_id="owner-001",
        language="en",
    )


# ---------------------------------------------------------------------------
# Mock ports
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_file_storage() -> MagicMock:
    storage = MagicMock()
    storage.upload.return_value = Ok("documents/pdf/doc-001/test.pdf")
    storage.download.return_value = Ok(None)
    storage.delete.return_value = Ok(None)
    storage.exists.return_value = True
    storage.get_url.return_value = Ok("http://minio/documents/pdf/doc-001/test.pdf")
    return storage


@pytest.fixture
def mock_vector_store(sample_search_result: SearchResult) -> MagicMock:
    store = MagicMock()
    store.upsert.return_value = Ok(None)
    store.search.return_value = Ok([sample_search_result])
    store.delete.return_value = Ok(None)
    store.delete_by_document.return_value = Ok(None)
    store.count.return_value = Ok(1)
    return store


@pytest.fixture
def mock_embedder(sample_embedding: Embedding) -> MagicMock:
    embedder = MagicMock()
    embedder.embed.return_value = Ok([[0.1] * 1024])
    embedder.embed_chunks.return_value = Ok([sample_embedding])
    return embedder


@pytest.fixture
def mock_parser(sample_parsed_doc: ParsedDocument) -> MagicMock:
    parser = MagicMock()
    parser.supports.return_value = True
    parser.parse.return_value = Ok(sample_parsed_doc)
    return parser


@pytest.fixture
def mock_chunker(sample_chunk: Chunk) -> MagicMock:
    chunker = MagicMock()
    chunker.chunk.return_value = Ok([sample_chunk])
    return chunker


@pytest.fixture
def mock_metadata_store(
    sample_document: Document,
    sample_stored_chunk_metadata: StoredChunkMetadata,
    sample_document_context: StoredDocumentContext,
) -> MagicMock:
    store = MagicMock()
    store.upsert_document.return_value = Ok(None)
    store.update_document_status.return_value = Ok(None)
    store.upsert_chunks.return_value = Ok(None)
    store.upsert_chunks_with_outbox.return_value = Ok(None)
    store.get_document_context.return_value = Ok(sample_document_context)
    store.list_chunks.return_value = Ok([sample_stored_chunk_metadata])
    store.upsert_concepts.return_value = Ok(None)
    store.upsert_chunk_concepts.return_value = Ok(None)
    store.persist_enrichment_batch.return_value = Ok("event-001")
    store.persist_curriculum_quiz_items.return_value = Ok(None)
    store.record_llm_usage.return_value = Ok(None)
    store.list_lesson_cards.return_value = Ok([])
    store.list_quiz_items.return_value = Ok([])
    store.get.return_value = Ok(sample_document)
    store.list.return_value = Ok([sample_document])
    store.delete.return_value = Ok(None)
    store.get_document_status.return_value = Ok(None)
    return store


@pytest.fixture
def mock_graph_store() -> MagicMock:
    store = MagicMock()
    store.upsert_heading_graph.return_value = Ok(None)
    store.upsert_concept_graph.return_value = Ok(None)
    store.delete_document.return_value = Ok(None)
    return store


@pytest.fixture
def mock_job_queue() -> MagicMock:
    queue = MagicMock()
    queue.enqueue.return_value = Ok("job-001")
    queue.enqueue_enrichment.return_value = Ok("enrichment-job-001")
    return queue


@pytest.fixture
def mock_llm_client() -> MagicMock:
    client = MagicMock()
    client.model_id = "gemini-test"
    return as_llm_client(client)
