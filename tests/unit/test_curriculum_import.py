from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from document_chunk.adapters.curriculum.dcmh_extractor import DcmhExtractor
from document_chunk.domain.entities.document import Document, DocumentType, ParsedDocument
from document_chunk.shared.result import Err, Ok
from worker.runners.document_worker import _process_job
from worker.services.curriculum_import_store import CurriculumImport
from worker.use_cases.curriculum_import import (
    CURRICULUM_APPLY_JOB,
    CURRICULUM_PREVIEW_JOB,
    CurriculumImportUseCase,
    curriculum_preview,
)

_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "packages-ai" / "tests" / "fixtures" / "syllabus" / "DCMH.CO3011.pdf"
)
IMPORT_ID = "01900000-0000-7000-8000-0000000000aa"
COURSE_ID = "01900000-0000-7000-8000-0000000000bb"


class FakeStore:
    def __init__(self, status: str) -> None:
        self.record = CurriculumImport(
            import_id=IMPORT_ID,
            course_id=COURSE_ID,
            status=status,
            file_name="DCMH.CO3011.pdf",
            storage_key=f"curricula/{COURSE_ID}/{IMPORT_ID}/DCMH.CO3011.pdf",
        )
        self.transitions: list[dict] = []
        self.document_ids = ["doc-1", "doc-2"]

    def get(self, import_id):
        return self.record if import_id == IMPORT_ID else None

    def transition(self, import_id, *, to, expect=None, preview=None, error=None):
        if expect and self.record.status not in expect:
            return False
        self.transitions.append({"to": to, "preview": preview, "error": error})
        self.record = CurriculumImport(**{**self.record.__dict__, "status": to})
        return True

    def indexed_document_ids(self, course_id):
        assert course_id == COURSE_ID
        return self.document_ids


@pytest.fixture(scope="module")
def co3011():
    if not _FIXTURE.exists():
        pytest.skip("cần fixture đề cương CO3011")
    parsed = ParsedDocument(
        document=Document(
            id="dcmh",
            name=_FIXTURE.name,
            path=_FIXTURE,
            doc_type=DocumentType.PDF,
            size_bytes=_FIXTURE.stat().st_size,
            mime_type="application/pdf",
        ),
        sections=[],
        page_count=14,
    )
    return DcmhExtractor().extract(parsed, course_id_hint=COURSE_ID).unwrap()


def build(status: str, *, ingest=None, mapper=None):
    store = FakeStore(status)
    file_storage = MagicMock()
    file_storage.download.return_value = Ok(None)
    ingest = ingest or MagicMock()
    mapper = mapper or MagicMock()
    use_case = CurriculumImportUseCase(
        store=store,
        file_storage=file_storage,
        ingest_curriculum_use_case=ingest,
        map_chunks_to_los_use_case=mapper,
    )
    return use_case, store, ingest, mapper, file_storage


def test_preview_lists_what_would_be_written(co3011):
    preview = curriculum_preview(co3011)

    assert preview["course"]["code"] == "CO3011"
    assert preview["counts"]["chapters"] == 12
    assert preview["counts"]["learning_outcomes"] == 9
    chapter_4 = next(ch for ch in preview["chapters"] if ch["code"] == "4")
    assert chapter_4["lo_codes"] == ["L.O.2.2"]
    assert preview["blocking"] is False
    assert any(issue["code"] == "lo.bloom_inferred" for issue in preview["issues"])


def test_preview_stores_result_without_writing_the_curriculum(co3011):
    use_case, store, ingest, _, file_storage = build("QUEUED")
    ingest.extract.return_value = Ok(co3011)

    result = use_case.run(CURRICULUM_PREVIEW_JOB, IMPORT_ID)

    assert result["status"] == "READY"
    assert [t["to"] for t in store.transitions] == ["EXTRACTING", "READY"]
    assert store.transitions[-1]["preview"]["counts"]["learning_outcomes"] == 9
    request = ingest.extract.call_args.args[0]
    assert request.course_id == COURSE_ID
    ingest.execute.assert_not_called()
    file_storage.download.assert_called_once()


def test_preview_with_blocking_issues_cannot_be_applied(co3011):
    blocking = SimpleNamespace(code="lo.none", severity="error", message="Không có LO", source=None)
    curriculum = SimpleNamespace(
        **{field: getattr(co3011, field) for field in co3011.__dataclass_fields__},
    )
    curriculum.issues = (*co3011.issues, blocking)
    curriculum.blocking_issues = (blocking,)
    use_case, store, ingest, _, _ = build("QUEUED")
    ingest.extract.return_value = Ok(curriculum)

    assert use_case.preview(IMPORT_ID)["status"] == "BLOCKED"
    assert store.record.status == "BLOCKED"


def test_preview_failure_is_recorded_on_the_import():
    use_case, store, ingest, _, _ = build("QUEUED")
    ingest.extract.return_value = Err(ValueError("PDF hỏng"))

    assert use_case.preview(IMPORT_ID)["status"] == "FAILED"
    assert store.record.status == "FAILED"
    assert "PDF hỏng" in store.transitions[-1]["error"]


def test_preview_job_replayed_after_completion_is_ignored():
    use_case, store, ingest, _, _ = build("READY")

    assert use_case.preview(IMPORT_ID) == {"import_id": IMPORT_ID, "skipped": "READY"}
    ingest.extract.assert_not_called()
    assert store.transitions == []


def test_apply_writes_curriculum_then_maps_existing_chunks():
    ingest = MagicMock()
    ingest.execute.return_value = Ok(SimpleNamespace(lo_count=9, chapter_count=12, warnings=[]))
    mapper = MagicMock()
    mapper.execute.side_effect = [
        Ok(SimpleNamespace(mapping_count=5)),
        Err(RuntimeError("qdrant down")),
    ]
    use_case, store, _, _, _ = build("APPLYING", ingest=ingest, mapper=mapper)

    result = use_case.run(CURRICULUM_APPLY_JOB, IMPORT_ID)

    assert result["status"] == "APPLIED"
    assert result["chunk_lo_mappings"] == 5
    assert store.record.status == "APPLIED"
    assert ingest.execute.call_args.args[0].course_id == COURSE_ID
    assert [call.args[0].document_id for call in mapper.execute.call_args_list] == ["doc-1", "doc-2"]


def test_apply_refused_by_the_quality_gate_marks_the_import_failed():
    ingest = MagicMock()
    ingest.execute.return_value = Err(ValueError("Đề cương chưa đạt kiểm tra cấu trúc"))
    use_case, store, _, mapper, _ = build("APPLYING", ingest=ingest)

    assert use_case.apply(IMPORT_ID)["status"] == "FAILED"
    assert store.record.status == "FAILED"
    mapper.execute.assert_not_called()


def test_apply_only_runs_for_an_import_core_api_marked_applying():
    use_case, store, ingest, _, _ = build("READY")

    assert use_case.apply(IMPORT_ID)["skipped"] == "READY"
    ingest.execute.assert_not_called()


@pytest.mark.asyncio
async def test_document_worker_routes_curriculum_jobs_by_name():
    container = MagicMock()
    container.curriculum_import_use_case.run.return_value = {"status": "READY"}
    job = SimpleNamespace(id="9", name=CURRICULUM_PREVIEW_JOB, data={"import_id": IMPORT_ID})

    result = await _process_job(container, job, "token")

    assert result == {"status": "READY"}
    container.curriculum_import_use_case.run.assert_called_once_with(CURRICULUM_PREVIEW_JOB, IMPORT_ID)
    container.metadata_store.get_document_status.assert_not_called()


@pytest.mark.asyncio
async def test_document_worker_remaps_a_document_on_request():
    container = MagicMock()
    container.map_chunks_to_los_use_case.execute.return_value = Ok(SimpleNamespace(mapping_count=4))
    job = SimpleNamespace(id="10", name="map_document_los",
                          data={"document_id": "doc-1", "course_id": COURSE_ID})

    result = await _process_job(container, job, "token")

    assert result == {"document_id": "doc-1", "mapping_count": 4}
    request = container.map_chunks_to_los_use_case.execute.call_args.args[0]
    assert (request.document_id, request.course_id) == ("doc-1", COURSE_ID)
    container.metadata_store.get_document_status.assert_not_called()
