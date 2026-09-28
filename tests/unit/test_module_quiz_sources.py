from unittest.mock import MagicMock
from uuid import UUID

from document_chunk.application.dto.generation_dto import GenerateCurriculumQuizRequest
from document_chunk.domain.ports.metadata_store import StoredChunkMetadata
from document_chunk.shared.result import Ok
from worker.use_cases.generate_curriculum_quiz import GenerateCurriculumQuizUseCase


def test_module_quiz_uses_only_chapter_documents_and_never_unfiltered_mcp():
    store = MagicMock()
    mcp = MagicMock()
    store.list_chunks_for_lo.return_value = Ok([
        StoredChunkMetadata(chunk_id="wrong", document_id="chapter2", chunk_index=0, heading_path=(), content_text="Wrong chapter"),
        StoredChunkMetadata(chunk_id="right", document_id="chapter1", chunk_index=0, heading_path=(), content_text="Right chapter"),
    ])
    use_case = GenerateCurriculumQuizUseCase(store, MagicMock(), context_client=mcp)
    request = GenerateCurriculumQuizRequest(course_id="course", target_kind="lo", target_code="LO1", source_document_ids=["chapter1"])
    parts, ids, document = use_case._load_context(request, "LO1", "LO1", "Statement", "understand").unwrap()
    assert ids == ["right"]
    assert document == "chapter1"
    assert all("Wrong chapter" not in p for p in parts)
    mcp.retrieve_quiz_context.assert_not_called()


def test_explicit_empty_source_scope_cannot_fall_back_to_another_chapter():
    store = MagicMock()
    store.list_chunks_for_lo.return_value = Ok([
        StoredChunkMetadata(chunk_id="wrong", document_id="other", chunk_index=0, heading_path=(), content_text="Other chapter"),
    ])
    use_case = GenerateCurriculumQuizUseCase(store, MagicMock(), context_client=MagicMock())
    request = GenerateCurriculumQuizRequest(course_id="course", target_kind="lo", target_code="LO1", source_document_ids=[])
    assert use_case._load_context(request, "LO1", "LO1", "Statement", "understand").unwrap() == ([], [], None)


def test_postgres_uuid_objects_match_document_ids_from_the_job():
    document_id = UUID("00000000-0000-0000-0000-000000000001")
    chunk_id = UUID("00000000-0000-0000-0000-000000000002")
    store = MagicMock()
    store.list_chunks_for_lo.return_value = Ok([
        StoredChunkMetadata(chunk_id=chunk_id, document_id=document_id, chunk_index=0, content_text="Chapter slide"),
    ])
    use_case = GenerateCurriculumQuizUseCase(store, MagicMock())
    request = GenerateCurriculumQuizRequest(course_id="course", target_kind="lo", target_code="LO1", source_document_ids=[str(document_id)])
    parts, ids, document = use_case._load_context(request, "LO1", "LO1", "Statement", "understand").unwrap()
    assert len(parts) == 1
    assert ids == [str(chunk_id)]
    assert document == str(document_id)
