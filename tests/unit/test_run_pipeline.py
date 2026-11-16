from types import SimpleNamespace
from dataclasses import replace
from unittest.mock import MagicMock

from document_chunk.domain.exceptions import DocumentStaleError
from document_chunk.shared.result import Err, Ok

from tests.support import RecordingResultPublisher
from worker.use_cases.run_pipeline import RunPipelineRequest, RunPipelineUseCase


class TestRunPipelineUseCase:
    def test_enqueues_enrichment_after_worker_pipeline_success(
        self,
        sample_document,
        mock_parser,
        mock_chunker,
        mock_embedder,
        mock_vector_store,
        mock_metadata_store,
        mock_job_queue,
    ):
        use_case = RunPipelineUseCase(
            parsers=[mock_parser],
            chunker=mock_chunker,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            metadata_store=mock_metadata_store,
            job_queue=mock_job_queue,
            result_publisher=RecordingResultPublisher(),
        )

        result = use_case.execute(
            RunPipelineRequest(
                document_id=sample_document.id,
                file_path=sample_document.path,
                storage_key="pdf/doc-001/test.pdf",
                original_file_name=sample_document.name,
                language="en",
                metadata={"course_id": "course-001"},
            )
        )

        assert result.is_ok()
        mock_job_queue.enqueue_enrichment.assert_called_once()
        payload = mock_job_queue.enqueue_enrichment.call_args.args[0]
        assert payload.document_id == sample_document.id

    def test_enrichment_enqueue_failure_does_not_fail_worker_pipeline(
        self,
        sample_document,
        mock_parser,
        mock_chunker,
        mock_embedder,
        mock_vector_store,
        mock_metadata_store,
        mock_job_queue,
    ):
        mock_job_queue.enqueue_enrichment.return_value = Err(RuntimeError("redis down"))
        use_case = RunPipelineUseCase(
            parsers=[mock_parser],
            chunker=mock_chunker,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            metadata_store=mock_metadata_store,
            job_queue=mock_job_queue,
            result_publisher=RecordingResultPublisher(),
        )

        result = use_case.execute(
            RunPipelineRequest(
                document_id=sample_document.id,
                file_path=sample_document.path,
                storage_key="pdf/doc-001/test.pdf",
                original_file_name=sample_document.name,
            )
        )

        assert result.is_ok()

    def test_preprocessor_failure_falls_back_to_original_file_path(
        self,
        sample_document,
        mock_parser,
        mock_chunker,
        mock_embedder,
        mock_vector_store,
        mock_metadata_store,
        mock_job_queue,
    ):
        failing_preprocessor = mock_parser.__class__()
        failing_preprocessor.should_apply.return_value = True
        failing_preprocessor.process.side_effect = RuntimeError("ocr failed")

        use_case = RunPipelineUseCase(
            parsers=[mock_parser],
            chunker=mock_chunker,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            metadata_store=mock_metadata_store,
            job_queue=mock_job_queue,
            result_publisher=RecordingResultPublisher(),
            preprocessors=[failing_preprocessor],
        )

        result = use_case.execute(
            RunPipelineRequest(
                document_id=sample_document.id,
                file_path=sample_document.path,
                storage_key="pdf/doc-001/test.pdf",
                original_file_name=sample_document.name,
            )
        )

        assert result.is_ok()
        mock_parser.parse.assert_called_once_with(sample_document.path)


class TestLoMappingWiring:
    """
    D1 — `chunk_lo_mappings` chưa bao giờ được ghi.

    `MapChunksToLosUseCase` trước đây chỉ được khai báo trong container mà không
    nơi nào gọi, nên bảng luôn rỗng: sinh quiz theo LO báo "No grounded source
    chunks", còn enrichment sinh xong rồi INSERT ra 0 dòng và mất im lặng.
    Tài liệu gọi đây là điều kiện để mọi thứ khác có ý nghĩa.
    """

    def _use_case(self, mapper, **mocks):
        return RunPipelineUseCase(
            parsers=[mocks["mock_parser"]],
            chunker=mocks["mock_chunker"],
            embedder=mocks["mock_embedder"],
            vector_store=mocks["mock_vector_store"],
            metadata_store=mocks["mock_metadata_store"],
            job_queue=mocks["mock_job_queue"],
            result_publisher=RecordingResultPublisher(),
            map_chunks_to_los_use_case=mapper,
        )

    def _request(self, sample_document):
        return RunPipelineRequest(
            document_id=sample_document.id,
            file_path=sample_document.path,
            storage_key="docs/test.pdf",
            original_file_name=sample_document.name,
        )

    def test_mapper_runs_after_a_successful_pipeline(
        self, sample_document, mock_parser, mock_chunker, mock_embedder,
        mock_vector_store, mock_metadata_store, mock_job_queue,
    ):
        mapper = MagicMock()
        mapper.execute.return_value = Ok(SimpleNamespace(mapping_count=7))

        result = self._use_case(
            mapper, mock_parser=mock_parser, mock_chunker=mock_chunker,
            mock_embedder=mock_embedder, mock_vector_store=mock_vector_store,
            mock_metadata_store=mock_metadata_store, mock_job_queue=mock_job_queue,
        ).execute(self._request(sample_document))

        assert result.is_ok()
        mapper.execute.assert_called_once()
        sent = mapper.execute.call_args.args[0]
        assert sent.document_id == sample_document.id
        assert sent.course_id == "course-001"

    def test_document_without_a_course_is_skipped(
        self, sample_document, sample_document_context, mock_parser, mock_chunker,
        mock_embedder, mock_vector_store, mock_metadata_store, mock_job_queue,
    ):
        """Không gắn học phần thì không có đề cương để đối chiếu."""
        mock_metadata_store.get_document_context.return_value = Ok(
            replace(sample_document_context, course_id=None)
        )
        mapper = MagicMock()

        result = self._use_case(
            mapper, mock_parser=mock_parser, mock_chunker=mock_chunker,
            mock_embedder=mock_embedder, mock_vector_store=mock_vector_store,
            mock_metadata_store=mock_metadata_store, mock_job_queue=mock_job_queue,
        ).execute(self._request(sample_document))

        assert result.is_ok()
        mapper.execute.assert_not_called()

    def test_mapping_failure_does_not_fail_the_pipeline(
        self, sample_document, mock_parser, mock_chunker, mock_embedder,
        mock_vector_store, mock_metadata_store, mock_job_queue,
    ):
        """Tài liệu đã index xong rồi; map hỏng thì để backfill, không huỷ cả lượt."""
        mapper = MagicMock()
        mapper.execute.return_value = Err(RuntimeError("no curriculum"))

        result = self._use_case(
            mapper, mock_parser=mock_parser, mock_chunker=mock_chunker,
            mock_embedder=mock_embedder, mock_vector_store=mock_vector_store,
            mock_metadata_store=mock_metadata_store, mock_job_queue=mock_job_queue,
        ).execute(self._request(sample_document))

        assert result.is_ok()
        mock_job_queue.enqueue_enrichment.assert_called_once()

    def test_pipeline_still_works_without_a_mapper(
        self, sample_document, mock_parser, mock_chunker, mock_embedder,
        mock_vector_store, mock_metadata_store, mock_job_queue,
    ):
        result = self._use_case(
            None, mock_parser=mock_parser, mock_chunker=mock_chunker,
            mock_embedder=mock_embedder, mock_vector_store=mock_vector_store,
            mock_metadata_store=mock_metadata_store, mock_job_queue=mock_job_queue,
        ).execute(self._request(sample_document))

        assert result.is_ok()


class TestStatusGoesToCoreApi:
    """
    documents là bảng của core-api: mỗi chuyển trạng thái là một
    DOCUMENT_PROCESSING_RESULT, worker không UPDATE documents nữa.
    """

    def _use_case(self, publisher, mocks):
        return RunPipelineUseCase(
            parsers=[mocks["mock_parser"]],
            chunker=mocks["mock_chunker"],
            embedder=mocks["mock_embedder"],
            vector_store=mocks["mock_vector_store"],
            metadata_store=mocks["mock_metadata_store"],
            job_queue=mocks["mock_job_queue"],
            result_publisher=publisher,
        )

    def _request(self, sample_document, **extra):
        return RunPipelineRequest(
            document_id=sample_document.id,
            file_path=sample_document.path,
            storage_key="docs/test.pdf",
            original_file_name=sample_document.name,
            **extra,
        )

    def _mocks(self, **kwargs):
        return {k: v for k, v in kwargs.items() if k.startswith("mock_")}

    def test_each_stage_is_published_with_generation_and_attempt(
        self, sample_document, mock_parser, mock_chunker, mock_embedder,
        mock_vector_store, mock_metadata_store, mock_job_queue,
    ):
        publisher = RecordingResultPublisher()
        mocks = self._mocks(
            mock_parser=mock_parser, mock_chunker=mock_chunker, mock_embedder=mock_embedder,
            mock_vector_store=mock_vector_store, mock_metadata_store=mock_metadata_store,
            mock_job_queue=mock_job_queue,
        )

        result = self._use_case(publisher, mocks).execute(
            self._request(sample_document, processing_generation=7, attempt=2)
        )

        assert result.is_ok(), str(result.error)
        assert [env["kind"] for env in publisher.envelopes] == ["DOCUMENT_PROCESSING_RESULT"] * 4
        assert [env["payload"]["status"]["value"] for env in publisher.envelopes] == [
            "PARSING", "CHUNKING", "EMBEDDING", "INDEXED",
        ]
        for env in publisher.envelopes:
            status = env["payload"]["status"]
            assert env["payload"]["document_id"] == sample_document.id
            assert (status["stage"], status["processing_generation"], status["attempt"]) == (
                "processing", 7, 2,
            )
        assert publisher.envelopes[0]["result_id"] == (
            "DOCUMENT_PROCESSING_RESULT.doc-001.g7-processing2-PARSING"
        )
        mock_metadata_store.update_document_status.assert_not_called()
        mock_metadata_store.upsert_document.assert_not_called()
        # Chunks + outbox chiếu heading graph vẫn do worker ghi (dữ liệu AI).
        mock_metadata_store.upsert_chunks_with_outbox.assert_called_once()
        # Lượt xử lý đi tiếp sang job enrichment.
        enrichment = mock_job_queue.enqueue_enrichment.call_args.args[0]
        assert enrichment.processing_generation == 7

    def test_failure_is_published_as_error(
        self, sample_document, mock_parser, mock_chunker, mock_embedder,
        mock_vector_store, mock_metadata_store, mock_job_queue,
    ):
        mock_parser.parse.return_value = Err(RuntimeError("bad pdf"))
        publisher = RecordingResultPublisher()
        mocks = self._mocks(
            mock_parser=mock_parser, mock_chunker=mock_chunker, mock_embedder=mock_embedder,
            mock_vector_store=mock_vector_store, mock_metadata_store=mock_metadata_store,
            mock_job_queue=mock_job_queue,
        )

        result = self._use_case(publisher, mocks).execute(self._request(sample_document))

        assert result.is_err()
        statuses = [env["payload"]["status"] for env in publisher.envelopes]
        assert [s["value"] for s in statuses] == ["PARSING", "ERROR"]
        assert statuses[-1]["error"] == "bad pdf"
        mock_metadata_store.update_document_status.assert_not_called()
        mock_job_queue.enqueue_enrichment.assert_not_called()

    def test_publish_failure_fails_the_job_before_any_work(
        self, sample_document, mock_parser, mock_chunker, mock_embedder,
        mock_vector_store, mock_metadata_store, mock_job_queue,
    ):
        publisher = RecordingResultPublisher(fail_with=ConnectionError("redis down"))
        mocks = self._mocks(
            mock_parser=mock_parser, mock_chunker=mock_chunker, mock_embedder=mock_embedder,
            mock_vector_store=mock_vector_store, mock_metadata_store=mock_metadata_store,
            mock_job_queue=mock_job_queue,
        )

        result = self._use_case(publisher, mocks).execute(self._request(sample_document))

        assert result.is_err()
        assert isinstance(result.error, ConnectionError)
        mock_parser.parse.assert_not_called()
        mock_metadata_store.upsert_chunks_with_outbox.assert_not_called()
        mock_metadata_store.update_document_status.assert_not_called()

    def test_deleted_document_stops_the_pipeline_without_writing(
        self, sample_document, mock_parser, mock_chunker, mock_embedder,
        mock_vector_store, mock_metadata_store, mock_job_queue,
    ):
        """Delete-race guard: tài liệu đã xoá thì không parse, không ghi gì."""
        mock_metadata_store.get_document_context.return_value = Ok(None)
        publisher = RecordingResultPublisher()
        mocks = self._mocks(
            mock_parser=mock_parser, mock_chunker=mock_chunker, mock_embedder=mock_embedder,
            mock_vector_store=mock_vector_store, mock_metadata_store=mock_metadata_store,
            mock_job_queue=mock_job_queue,
        )

        result = self._use_case(publisher, mocks).execute(self._request(sample_document))

        assert result.is_err()
        assert isinstance(result.error, DocumentStaleError)
        assert publisher.envelopes == []
        mock_parser.parse.assert_not_called()
        mock_vector_store.upsert.assert_not_called()
