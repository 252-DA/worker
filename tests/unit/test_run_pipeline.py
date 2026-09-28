from types import SimpleNamespace
from dataclasses import replace
from unittest.mock import MagicMock
from document_chunk.shared.result import Err, Ok
from document_chunk.shared.result import Err

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
