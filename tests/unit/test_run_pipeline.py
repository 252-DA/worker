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
