import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from bullmq.custom_errors import UnrecoverableError

from document_chunk.adapters.curriculum.dcmh_extractor import DcmhExtractor
from document_chunk.domain.entities.document import Document, DocumentType, ParsedDocument
from document_chunk.domain.exceptions import DocumentStaleError
from document_chunk.shared.result import Err, Ok
from tests.support import RecordingResultPublisher
from worker.runners.document_worker import _process_job
import worker.use_cases.curriculum_import as curriculum_import_module
from worker.use_cases.curriculum_import import (
    COURSE_CHUNK_LO_REMAP_REQUESTED,
    CURRICULUM_PREVIEW_JOB,
    LEGACY_CURRICULUM_APPLY_JOB,
    CurriculumImportUseCase,
    InvalidCurriculumJob,
    curriculum_payload,
    curriculum_preview,
)

_ROOT = Path(__file__).resolve().parents[3]
_FIXTURE = _ROOT / "packages-ai" / "tests" / "fixtures" / "syllabus" / "DCMH.CO3011.pdf"
# Fixture hợp đồng sinh từ đúng đề cương này (cùng course_id/document_id bên dưới);
# core-api dùng nó làm đầu vào cho handler và writer.
_CONTRACT_FIXTURE = (
    _ROOT / "contracts" / "fixtures" / "v1" / "valid" / "curriculum-import-result.json"
)
IMPORT_ID = "01900000-0000-7000-8000-0000000000aa"
COURSE_ID = "01900000-0000-7000-8000-0000000000bb"
DOCUMENT_ID = "01900000-0000-7000-8000-0000000000dd"
STORAGE_KEY = f"curricula/{COURSE_ID}/{IMPORT_ID}/DCMH.CO3011.pdf"
PREVIEW_DATA = {
    "import_id": IMPORT_ID,
    "course_id": COURSE_ID,
    "file_name": "DCMH.CO3011.pdf",
    "storage_key": STORAGE_KEY,
}
RESULT_ID = f"CURRICULUM_IMPORT_RESULT.{IMPORT_ID}"


@pytest.fixture(scope="module")
def co3011():
    if not _FIXTURE.exists():
        pytest.skip("cần fixture đề cương CO3011")
    parsed = ParsedDocument(
        document=Document(
            id=DOCUMENT_ID,
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


class FakeCourseDocuments:
    def __init__(self, document_ids=("doc-1", "doc-2"), error: Exception | None = None):
        self.document_ids = list(document_ids)
        self.error = error
        self.calls: list[str] = []

    def indexed_document_ids(self, course_id):
        self.calls.append(course_id)
        if self.error:
            raise self.error
        return self.document_ids


def build(*, ingest=None, mapper=None, publisher=None, documents=None):
    file_storage = MagicMock()
    file_storage.download.return_value = Ok(None)
    ingest = ingest or MagicMock()
    mapper = mapper or MagicMock()
    publisher = publisher or RecordingResultPublisher()
    documents = documents or FakeCourseDocuments()
    use_case = CurriculumImportUseCase(
        publisher=publisher,
        file_storage=file_storage,
        ingest_curriculum_use_case=ingest,
        map_chunks_to_los_use_case=mapper,
        course_documents=documents,
    )
    return SimpleNamespace(
        use_case=use_case, ingest=ingest, mapper=mapper, publisher=publisher,
        documents=documents, file_storage=file_storage,
    )


# ---------------------------------------------------------------------------
# Bản xem trước và bản đề cương chuẩn hoá
# ---------------------------------------------------------------------------


def test_preview_lists_what_would_be_written(co3011):
    preview = curriculum_preview(co3011)

    assert preview["course"]["code"] == "CO3011"
    assert preview["counts"]["chapters"] == 12
    assert preview["counts"]["learning_outcomes"] == 9
    chapter_4 = next(ch for ch in preview["chapters"] if ch["code"] == "4")
    assert chapter_4["lo_codes"] == ["L.O.2.2"]
    assert preview["blocking"] is False
    assert any(issue["code"] == "lo.bloom_inferred" for issue in preview["issues"])


def test_curriculum_payload_carries_every_table_in_column_form(co3011):
    payload = curriculum_payload(co3011)

    assert {k: len(v) for k, v in payload.items() if isinstance(v, list)} == {
        "chapters": 12, "learning_outcomes": 9, "chapter_lo_links": 12, "goals": 2,
        "sessions": 15, "assessments": 3, "lo_assessment_links": 38, "issues": 7,
    }
    lo = next(lo for lo in payload["learning_outcomes"] if lo["code"] == "L.O.2.1")
    # Bloom đã quy về số như cột learning_outcomes.bloom_level; CDIO không có trong đề cương.
    assert lo["bloom_level"] == 3 and lo["cdio_level"] is None
    assert lo["bloom_provenance"] == "inferred"
    # Mã thay cho id: core-api tự sinh id tất định, worker không gửi id dòng.
    assert "lo_id" not in lo and "chapter_id" not in payload["chapters"][0]
    assert payload["source_document_id"] == DOCUMENT_ID
    json.dumps(payload, allow_nan=False)


def test_contract_fixture_matches_what_the_worker_builds(co3011):
    """Fixture hợp đồng là đầu vào test của core-api: phải đúng thứ worker gửi."""
    envelope = json.loads(_CONTRACT_FIXTURE.read_text(encoding="utf-8"))

    assert envelope["payload"]["curriculum"] == curriculum_payload(co3011)
    assert envelope["payload"]["preview"] == json.loads(json.dumps(curriculum_preview(co3011)))
    assert envelope["result_id"] == RESULT_ID


# ---------------------------------------------------------------------------
# Job xem trước
# ---------------------------------------------------------------------------


def test_preview_publishes_preview_and_full_curriculum_without_writing(co3011):
    t = build()
    t.ingest.extract.return_value = Ok(co3011)

    result = t.use_case.run(CURRICULUM_PREVIEW_JOB, PREVIEW_DATA)

    assert result["status"] == "READY"
    [envelope] = t.publisher.envelopes
    assert envelope["kind"] == "CURRICULUM_IMPORT_RESULT"
    assert envelope["result_id"] == RESULT_ID
    assert envelope["source_event_id"] is None
    payload = envelope["payload"]
    assert payload["status"] == "READY" and payload["error"] is None
    assert payload["preview"]["counts"]["learning_outcomes"] == 9
    assert payload["curriculum"] == json.loads(json.dumps(curriculum_payload(co3011)))
    request = t.ingest.extract.call_args.args[0]
    assert (request.course_id, request.file_name) == (COURSE_ID, "DCMH.CO3011.pdf")
    t.ingest.execute.assert_not_called()
    assert t.file_storage.download.call_args.args[0] == STORAGE_KEY
    t.mapper.execute.assert_not_called()


def test_preview_with_blocking_issues_is_reported_blocked(co3011):
    blocking = SimpleNamespace(code="lo.none", severity="error", message="Không có LO", source=None)
    curriculum = SimpleNamespace(
        **{field: getattr(co3011, field) for field in co3011.__dataclass_fields__},
    )
    curriculum.issues = (*co3011.issues, blocking)
    curriculum.blocking_issues = (blocking,)
    t = build()
    t.ingest.extract.return_value = Ok(curriculum)

    assert t.use_case.preview(PREVIEW_DATA)["status"] == "BLOCKED"
    payload = t.publisher.envelopes[0]["payload"]
    assert payload["status"] == "BLOCKED"
    assert payload["preview"]["blocking"] is True
    assert {"code": "lo.none", "severity": "error", "message": "Không có LO",
            "source_section": None, "source_page": None,
            "source_locator": None} in payload["curriculum"]["issues"]


def test_preview_failure_is_published_as_failed():
    t = build()
    t.ingest.extract.return_value = Err(ValueError("PDF hỏng"))

    assert t.use_case.preview(PREVIEW_DATA)["status"] == "FAILED"
    [envelope] = t.publisher.envelopes
    assert envelope["result_id"] == RESULT_ID
    assert envelope["payload"] == {
        "import_id": IMPORT_ID, "status": "FAILED", "preview": None,
        "curriculum": None, "error": "ValueError: PDF hỏng",
    }


def test_download_failure_is_published_as_failed():
    t = build()
    t.file_storage.download.return_value = Err(FileNotFoundError("không có object"))

    assert t.use_case.preview(PREVIEW_DATA)["status"] == "FAILED"
    assert "không có object" in t.publisher.envelopes[0]["payload"]["error"]
    t.ingest.extract.assert_not_called()


def test_publish_failure_fails_the_job_so_bullmq_retries(co3011):
    t = build(publisher=RecordingResultPublisher(fail_with=ConnectionError("redis down")))
    t.ingest.extract.return_value = Ok(co3011)

    with pytest.raises(ConnectionError):
        t.use_case.run(CURRICULUM_PREVIEW_JOB, PREVIEW_DATA)


def test_curriculum_that_cannot_be_serialised_is_reported_failed(co3011):
    """NaN không có trong JSON: chạy lại cũng hỏng, nên báo FAILED chứ không để job lỗi."""
    from dataclasses import replace

    broken = replace(co3011, assessments=(replace(co3011.assessments[0], weight=float("nan")),))
    t = build()
    t.ingest.extract.return_value = Ok(broken)

    assert t.use_case.preview(PREVIEW_DATA)["status"] == "FAILED"
    [envelope] = t.publisher.envelopes
    assert envelope["payload"]["status"] == "FAILED"
    assert "InvalidWorkerResult" in envelope["payload"]["error"]


def test_legacy_preview_job_with_only_import_id_reports_a_clear_failure():
    """Job xếp trước khi đổi luồng không có đủ dữ liệu; worker không đọc curriculum_imports nữa."""
    t = build()

    assert t.use_case.run(CURRICULUM_PREVIEW_JOB, {"import_id": IMPORT_ID})["status"] == "FAILED"
    payload = t.publisher.envelopes[0]["payload"]
    assert payload["status"] == "FAILED"
    assert "đồng bộ lại" in payload["error"]
    t.file_storage.download.assert_not_called()
    t.ingest.extract.assert_not_called()


@pytest.mark.parametrize("data", [{}, {"import_id": "not-a-uuid"}, {**PREVIEW_DATA, "course_id": "x"}])
def test_preview_job_with_invalid_ids_is_rejected(data):
    t = build()

    with pytest.raises(InvalidCurriculumJob):
        t.use_case.run(CURRICULUM_PREVIEW_JOB, data)
    assert t.publisher.envelopes == []


# ---------------------------------------------------------------------------
# Lệnh apply cũ và lệnh map lại chunk → LO
# ---------------------------------------------------------------------------


def test_legacy_apply_job_no_longer_writes_and_reports_failure():
    t = build()

    result = t.use_case.run(LEGACY_CURRICULUM_APPLY_JOB, {"import_id": IMPORT_ID})

    assert result == {"import_id": IMPORT_ID, "status": "FAILED"}
    [envelope] = t.publisher.envelopes
    # Khác result_id của bước xem trước: BullMQ không gộp nhầm với kết quả đã gửi.
    assert envelope["result_id"] == f"{RESULT_ID}.legacy-apply"
    assert envelope["payload"]["status"] == "FAILED"
    t.ingest.extract.assert_not_called()
    t.ingest.execute.assert_not_called()
    t.mapper.execute.assert_not_called()


def _remap_job_data(course_id=COURSE_ID):
    # Đúng dữ liệu job OutboxService.relay() đẩy vào document_processing.
    return {
        "event_id": "01a1f000-0000-7000-8000-0000000000e1",
        "event_type": COURSE_CHUNK_LO_REMAP_REQUESTED,
        "aggregate_type": "course",
        "aggregate_id": course_id,
        "payload": {"course_id": course_id, "import_id": IMPORT_ID},
    }


def test_remap_runs_map_chunks_to_los_for_every_indexed_document():
    mapper = MagicMock()
    mapper.execute.side_effect = [
        Ok(SimpleNamespace(mapping_count=5)),
        Ok(SimpleNamespace(mapping_count=3)),
    ]
    t = build(mapper=mapper)

    result = t.use_case.run(COURSE_CHUNK_LO_REMAP_REQUESTED, _remap_job_data())

    assert result == {"course_id": COURSE_ID, "documents": 2, "chunk_lo_mappings": 8}
    assert t.documents.calls == [COURSE_ID]
    assert [(c.args[0].document_id, c.args[0].course_id) for c in mapper.execute.call_args_list] == [
        ("doc-1", COURSE_ID), ("doc-2", COURSE_ID),
    ]
    # chunk_lo_mappings là dữ liệu của worker: không có kết quả nào gửi về core-api.
    assert t.publisher.envelopes == []
    t.ingest.extract.assert_not_called()


def test_remap_failure_of_a_live_document_surfaces_so_bullmq_reruns_the_job():
    """Log rồi bỏ qua thì tài liệu mãi không có cạnh chunk → LO mà job vẫn "xong"."""
    failure = RuntimeError("qdrant down")
    mapper = MagicMock()
    mapper.execute.side_effect = [
        Err(failure),
        Ok(SimpleNamespace(mapping_count=4)),
    ]
    t = build(mapper=mapper)

    with pytest.raises(RuntimeError) as raised:
        t.use_case.run(COURSE_CHUNK_LO_REMAP_REQUESTED, _remap_job_data())

    assert raised.value is failure
    # Tài liệu còn lại vẫn được map trước khi lỗi nổi lên.
    assert [c.args[0].document_id for c in mapper.execute.call_args_list] == ["doc-1", "doc-2"]


def test_remap_skips_a_document_deleted_while_mapping():
    class DeletedDuringRemap(FakeCourseDocuments):
        def indexed_document_ids(self, course_id):
            ids = super().indexed_document_ids(course_id)
            # Lần đọc lại (sau khi map lỗi) doc-1 đã bị xoá.
            return ids if len(self.calls) == 1 else [i for i in ids if i != "doc-1"]

    mapper = MagicMock()
    mapper.execute.side_effect = [
        Err(RuntimeError("FK violation: chunk vừa bị xoá")),
        Ok(SimpleNamespace(mapping_count=4)),
    ]
    t = build(mapper=mapper, documents=DeletedDuringRemap())

    result = t.use_case.run(COURSE_CHUNK_LO_REMAP_REQUESTED, _remap_job_data())

    assert result == {"course_id": COURSE_ID, "documents": 2, "chunk_lo_mappings": 4}


def test_remap_skips_a_stale_document_error_without_rechecking_the_list():
    mapper = MagicMock()
    mapper.execute.side_effect = [
        Err(DocumentStaleError("deleted", document_id="doc-1")),
        Ok(SimpleNamespace(mapping_count=4)),
    ]
    t = build(mapper=mapper)

    result = t.use_case.run(COURSE_CHUNK_LO_REMAP_REQUESTED, _remap_job_data())

    assert result["chunk_lo_mappings"] == 4
    assert t.documents.calls == [COURSE_ID]


def test_remap_retries_when_documents_cannot_be_listed():
    t = build(documents=FakeCourseDocuments(error=OSError("db down")))

    with pytest.raises(OSError):
        t.use_case.run(COURSE_CHUNK_LO_REMAP_REQUESTED, _remap_job_data())


@pytest.mark.parametrize("data", [{"payload": None}, {"payload": {"import_id": IMPORT_ID}}])
def test_remap_without_course_id_is_rejected(data):
    t = build()

    with pytest.raises(InvalidCurriculumJob):
        t.use_case.run(COURSE_CHUNK_LO_REMAP_REQUESTED, data)


def test_worker_no_longer_writes_the_curriculum():
    source = inspect.getsource(curriculum_import_module)
    assert "upsert_curriculum" not in source
    assert "CurriculumImportStore" not in source
    assert "psycopg" not in source
    assert importlib.util.find_spec("worker.services.curriculum_import_store") is None


# ---------------------------------------------------------------------------
# Định tuyến trong document worker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_document_worker_routes_curriculum_jobs_by_name():
    container = MagicMock()
    container.curriculum_import_use_case.run.return_value = {"status": "READY"}
    job = SimpleNamespace(id="9", name=CURRICULUM_PREVIEW_JOB, data=PREVIEW_DATA)

    result = await _process_job(container, job, "token")

    assert result == {"status": "READY"}
    container.curriculum_import_use_case.run.assert_called_once_with(CURRICULUM_PREVIEW_JOB, PREVIEW_DATA)
    container.metadata_store.get_document_status.assert_not_called()


@pytest.mark.asyncio
async def test_document_worker_runs_the_course_remap_command():
    t = build(mapper=MagicMock(**{"execute.return_value": Ok(SimpleNamespace(mapping_count=3))}))
    container = MagicMock(curriculum_import_use_case=t.use_case)
    job = SimpleNamespace(id="ev-1", name=COURSE_CHUNK_LO_REMAP_REQUESTED, data=_remap_job_data())

    result = await _process_job(container, job, "token")

    assert result == {"course_id": COURSE_ID, "documents": 2, "chunk_lo_mappings": 6}
    container.metadata_store.get_document_status.assert_not_called()


@pytest.mark.asyncio
async def test_document_worker_does_not_retry_an_invalid_curriculum_job():
    container = MagicMock(curriculum_import_use_case=build().use_case)
    job = SimpleNamespace(id="11", name=CURRICULUM_PREVIEW_JOB, data={"import_id": "bad"})

    with pytest.raises(UnrecoverableError):
        await _process_job(container, job, "token")


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
