"""
Trạng thái/chương tài liệu đi về core-api qua worker_results; worker không ghi
bảng `documents`, `lesson_cards`, `quiz_items` nữa.
"""
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from document_chunk.domain.exceptions import DocumentStaleError
from document_chunk.domain.ports.metadata_store import IngestionStatus
from document_chunk.shared.result import Err, Ok

from tests.support import RecordingResultPublisher
from worker.services.document_reporter import (
    PublishingDocumentChapterReporter,
    PublishingDocumentStatusReporter,
    job_attempt,
)
from worker.services.result_publisher import InvalidWorkerResult

DOC = "019fdbaa-c096-78e5-a91a-982998d5b5b0"


def _store(exists=True):
    store = MagicMock()
    store.get_document_context.return_value = Ok(object() if exists else None)
    return store


def _reporter(publisher, store=None, *, stage="processing", generation=3, attempt=1):
    return PublishingDocumentStatusReporter(
        publisher,
        store or _store(),
        stage=stage,
        processing_generation=generation,
        attempt=attempt,
    )


class TestJobAttempt:
    @pytest.mark.parametrize(
        ("started", "made", "expected"),
        [
            (1, 0, 1),  # lần đầu
            (2, 1, 2),  # chạy lại sau lỗi
            (3, 1, 3),  # chạy lại sau stall: attemptsMade không tăng
            (0, 0, 1),  # job cũ không có attemptsStarted
            (None, None, 1),
        ],
    )
    def test_later_runs_always_get_a_larger_number(self, started, made, expected):
        job = SimpleNamespace(attemptsStarted=started, attemptsMade=made)
        assert job_attempt(job) == expected

    def test_missing_attributes(self):
        assert job_attempt(object()) == 1


class TestStatusReporter:
    def test_payload_and_deterministic_result_id(self):
        publisher = RecordingResultPublisher()

        result = _reporter(publisher, attempt=2).report_status(DOC, IngestionStatus.CHUNKING)

        assert result.is_ok()
        [envelope] = publisher.envelopes
        assert envelope["kind"] == "DOCUMENT_PROCESSING_RESULT"
        assert envelope["result_id"] == f"DOCUMENT_PROCESSING_RESULT.{DOC}.g3-processing2-CHUNKING"
        assert envelope["payload"] == {
            "document_id": DOC,
            "status": {
                "value": "CHUNKING",
                "stage": "processing",
                "processing_generation": 3,
                "attempt": 2,
            },
        }

    def test_aliases_are_sent_as_real_values(self):
        publisher = RecordingResultPublisher()
        _reporter(publisher, stage="enrichment").report_status(DOC, IngestionStatus.ENRICHED)
        _reporter(publisher, stage="enrichment").report_status(DOC, IngestionStatus.DONE, "boom")

        values = [env["payload"]["status"]["value"] for env in publisher.envelopes]
        assert values == ["GENERATED_DRAFT", "INDEXED"]
        assert publisher.envelopes[1]["payload"]["status"]["error"] == "boom"

    def test_error_message_is_capped(self):
        publisher = RecordingResultPublisher()
        _reporter(publisher).report_status(DOC, IngestionStatus.ERROR, "x" * 5000)
        assert len(publisher.envelopes[0]["payload"]["status"]["error"]) == 1000

    @pytest.mark.parametrize(
        ("stage", "status"),
        [
            ("processing", IngestionStatus.ENRICHING),
            ("processing", IngestionStatus.GENERATED_DRAFT),
            ("processing", IngestionStatus.QUEUED),
            ("enrichment", IngestionStatus.ERROR),
            ("enrichment", IngestionStatus.PARSING),
        ],
    )
    def test_status_outside_its_stage_is_a_contract_error(self, stage, status):
        publisher = RecordingResultPublisher()
        result = _reporter(publisher, stage=stage).report_status(DOC, status)
        assert result.is_err()
        assert isinstance(result.error, InvalidWorkerResult)
        assert publisher.envelopes == []

    def test_deleted_document_is_stale_and_nothing_is_sent(self):
        publisher = RecordingResultPublisher()
        result = _reporter(publisher, _store(exists=False)).report_status(
            DOC, IngestionStatus.PARSING
        )
        assert isinstance(result.error, DocumentStaleError)
        assert publisher.envelopes == []

    def test_read_failure_is_returned(self):
        store = MagicMock()
        store.get_document_context.return_value = Err(RuntimeError("db down"))
        result = _reporter(RecordingResultPublisher(), store).report_status(
            DOC, IngestionStatus.PARSING
        )
        assert str(result.error) == "db down"

    def test_publish_failure_is_returned_not_raised(self):
        failure = TimeoutError("no ack")
        result = _reporter(RecordingResultPublisher(fail_with=failure)).report_status(
            DOC, IngestionStatus.PARSING
        )
        assert result.error is failure

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"stage": "upload"},
            {"generation": -1},
            {"generation": "1"},
            {"attempt": 0},
            {"attempt": True},
        ],
    )
    def test_invalid_run_identity_is_rejected_up_front(self, kwargs):
        with pytest.raises(ValueError):
            _reporter(RecordingResultPublisher(), **kwargs)


class TestChapterReporter:
    def test_payload_is_clamped_to_the_documents_columns(self):
        publisher = RecordingResultPublisher()

        result = PublishingDocumentChapterReporter(publisher).report_chapter(
            DOC, "C2", "content", 0.8349, "r" * 400
        )

        assert result.is_ok()
        [envelope] = publisher.envelopes
        assert envelope["payload"] == {
            "document_id": DOC,
            "chapter": {
                "chapter_code": "C2",
                "provenance": "content",
                "confidence": 0.83,
                "reason": "r" * 255,
            },
        }
        assert re.fullmatch(
            rf"DOCUMENT_PROCESSING_RESULT\.{DOC}\.chapter-[0-9a-f]{{32}}",
            envelope["result_id"],
        )

    def test_each_report_has_its_own_id(self):
        """C1 → C2 → C1: id theo nội dung sẽ bị BullMQ chặn nhầm lần thứ ba."""
        publisher = RecordingResultPublisher()
        reporter = PublishingDocumentChapterReporter(publisher)
        reporter.report_chapter(DOC, "C1", "content", 0.9, "r")
        reporter.report_chapter(DOC, "C1", "content", 0.9, "r")
        ids = {env["result_id"] for env in publisher.envelopes}
        assert len(ids) == 2

    def test_publish_failure_is_returned(self):
        failure = ConnectionError("redis down")
        result = PublishingDocumentChapterReporter(
            RecordingResultPublisher(fail_with=failure)
        ).report_chapter(DOC, "C1", "content", 0.9, "r")
        assert result.error is failure


class TestWorkerNoLongerWritesCoreTables:
    FORBIDDEN = (
        "update_document_status",
        "set_document_chapter",
        "upsert_document",
        "persist_enrichment_batch",
    )

    def test_no_worker_module_calls_the_old_writers(self):
        root = Path(__file__).resolve().parents[2] / "worker"
        offenders = [
            f"{path.relative_to(root)}: {name}"
            for path in root.rglob("*.py")
            for name in self.FORBIDDEN
            if re.search(rf"\.{name}\(", path.read_text(encoding="utf-8"))
        ]
        assert offenders == []

    def test_document_container_wires_publishing_reporters(self):
        from worker.config import WorkerSettings
        from worker.container import build_document_container

        container = build_document_container(WorkerSettings())
        try:
            assert container.run_pipeline_use_case._result_publisher is container.result_publisher
            reporter = container.map_chunks_to_los_use_case._chapter_reporter
            assert isinstance(reporter, PublishingDocumentChapterReporter)
            assert reporter._publisher is container.result_publisher
        finally:
            container.close()

    def test_enrichment_container_wires_the_publisher(self):
        from worker.config import WorkerSettings
        from worker.container import build_enrichment_container

        container = build_enrichment_container(WorkerSettings())
        try:
            assert container.run_enrichment_use_case._publisher is container.result_publisher
        finally:
            container.close()

    def test_outbox_container_wires_the_publisher(self):
        from worker.config import WorkerSettings
        from worker.container import build_outbox_container

        container = build_outbox_container(WorkerSettings())
        try:
            assert container.outbox_projector._result_publisher is container.result_publisher
        finally:
            container.close()


class TestEnvelopesMatchTheContract:
    """Payload worker gửi phải đúng schema mà handler core-api kiểm."""

    CONTRACTS = Path(__file__).resolve().parents[3] / "contracts" / "integration-events" / "v1"

    def _validate(self, envelope):
        jsonschema = pytest.importorskip("jsonschema")
        referencing = pytest.importorskip("referencing")
        if not self.CONTRACTS.exists():
            pytest.skip("contracts/ không có trong image này")
        import json
        import uuid as uuid_mod

        resources = []
        for path in self.CONTRACTS.glob("*.json"):
            schema = json.loads(path.read_text(encoding="utf-8"))
            resources.append(
                (schema["$id"], referencing.Resource.from_contents(
                    schema, default_specification=referencing.jsonschema.DRAFT7
                ))
            )
        registry = referencing.Registry().with_resources(resources)
        kind_file = envelope["kind"].lower().replace("_", "-") + ".json"
        schema = json.loads((self.CONTRACTS / kind_file).read_text(encoding="utf-8"))
        checker = jsonschema.FormatChecker()

        @checker.checks("uuid")
        def _uuid(value):
            uuid_mod.UUID(str(value))
            return True

        errors = list(
            jsonschema.Draft7Validator(schema, registry=registry, format_checker=checker)
            .iter_errors(envelope)
        )
        assert not errors, [e.message for e in errors]

    def test_status_and_chapter_envelopes(self):
        publisher = RecordingResultPublisher()
        _reporter(publisher, attempt=2).report_status(DOC, IngestionStatus.ERROR, "boom")
        _reporter(publisher, stage="enrichment").report_status(DOC, IngestionStatus.ENRICHED)
        PublishingDocumentChapterReporter(publisher).report_chapter(DOC, "C2", "content", 0.83, "r")
        assert len(publisher.envelopes) == 3
        for envelope in publisher.envelopes:
            self._validate(envelope)

    def test_enrichment_content_envelope(self, sample_document_context, mock_metadata_store, mock_llm_client):
        import uuid as uuid_mod

        from document_chunk.domain.ports.metadata_store import StoredChunkMetadata
        from worker.use_cases.run_enrichment import RunEnrichmentRequest, RunEnrichmentUseCase

        chunk_id = uuid_mod.uuid4()
        mock_metadata_store.get_document_context.return_value = Ok(sample_document_context)
        mock_metadata_store.list_chunks.return_value = Ok([
            StoredChunkMetadata(
                chunk_id=chunk_id, document_id=DOC, chunk_index=0,
                heading_path=("Giải tích",), heading_level=1, language="vi",
                content_text="Đạo hàm mô tả tốc độ thay đổi tức thời.",
            )
        ])
        mock_llm_client.generate.side_effect = [
            Ok('{"cards":[{"title":"' + "T" * 300 + '","bullets":["a","b"],"key_insight":"c"}]}'),
            Ok('{"questions":[{"question":"Để làm gì?","choices":["a","b","c","d"],'
               '"correct_index":1,"explanation":"vì vậy","difficulty":"medium"}]}'),
        ]
        publisher = RecordingResultPublisher()

        result = RunEnrichmentUseCase(
            metadata_store=mock_metadata_store,
            llm_client=mock_llm_client,
            result_publisher=publisher,
        ).execute(RunEnrichmentRequest(document_id=DOC, processing_generation=2))

        assert result.is_ok(), str(result.error)
        for envelope in publisher.envelopes:
            self._validate(envelope)
        content = next(e for e in publisher.envelopes if e["kind"] == "ENRICHMENT_CONTENT_RESULT")
        assert len(content["payload"]["lesson_cards"][0]["title"]) == 255
