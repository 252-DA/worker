from document_chunk.domain.ports.metadata_store import IngestionStatus, StoredChunkMetadata
from document_chunk.shared.result import Err, Ok

from worker.use_cases.run_enrichment import RunEnrichmentRequest, RunEnrichmentUseCase


class TestRunEnrichmentUseCase:
    def test_generates_cards_and_quiz_then_persists_batch(
        self,
        sample_document,
        sample_document_context,
        mock_metadata_store,
        mock_llm_client,
    ):
        mock_metadata_store.get_document_context.return_value = Ok(sample_document_context)
        mock_metadata_store.list_chunks.return_value = Ok(
            [
                StoredChunkMetadata(
                    chunk_id="chunk-001",
                    document_id=sample_document.id,
                    chunk_index=0,
                    heading_path=("Giải tích", "Đạo hàm"),
                    heading_level=2,
                    language="vi",
                    content_text="Đạo hàm mô tả tốc độ thay đổi tức thời.",
                ),
                StoredChunkMetadata(
                    chunk_id="chunk-002",
                    document_id=sample_document.id,
                    chunk_index=1,
                    heading_path=("Giải tích", "Đạo hàm"),
                    heading_level=2,
                    language="vi",
                    content_text="Đạo hàm giúp phân tích xu hướng và tối ưu.",
                ),
            ]
        )
        mock_llm_client.generate.side_effect = [
            Ok(
                '{"cards":[{"title":"Khái niệm đạo hàm","bullets":["Mô tả tốc độ thay đổi","Áp dụng trong tối ưu"],'
                '"key_insight":"Đạo hàm biến sự thay đổi thành đại lượng đo được."}]}'
            ),
            Ok(
                '{"questions":[{"question":"Đạo hàm dùng để làm gì?","choices":["Trang trí biểu đồ","Đo tốc độ thay đổi",'
                '"Đổi đơn vị","Đếm số trang"],"correct_index":1,"explanation":"Đạo hàm đo tốc độ thay đổi tức thời.",'
                '"difficulty":"medium"}]}'
            ),
        ]

        use_case = RunEnrichmentUseCase(
            metadata_store=mock_metadata_store,
            llm_client=mock_llm_client,
            max_section_chars=500,
        )

        result = use_case.execute(RunEnrichmentRequest(document_id=sample_document.id))

        assert result.is_ok()
        response = result.unwrap()
        assert response.concept_count == 2
        assert response.mention_count == 4
        assert response.card_count == 1
        assert response.quiz_count == 1
        assert mock_metadata_store.update_document_status.call_args_list[0].args == (
            sample_document.id,
            IngestionStatus.ENRICHING,
        )

        persist_kwargs = mock_metadata_store.persist_enrichment_batch.call_args.kwargs
        lesson_cards = persist_kwargs["lesson_cards"]
        quiz_items = persist_kwargs["quiz_items"]
        assert persist_kwargs["document_id"] == sample_document.id
        assert persist_kwargs["outbox_event_type"] == "concept_graph_project"
        assert lesson_cards[0].document_id == sample_document.id
        assert lesson_cards[0].source_chunk_ids == ("chunk-001", "chunk-002")
        assert lesson_cards[0].heading_path == ("Giải tích", "Đạo hàm")
        assert quiz_items[0].document_id == sample_document.id
        assert quiz_items[0].primary_chunk_id == "chunk-001"
        assert mock_llm_client.generate.call_count == 2

    def test_retries_once_when_cards_json_is_invalid(
        self,
        sample_document,
        sample_document_context,
        mock_metadata_store,
        mock_llm_client,
    ):
        mock_metadata_store.get_document_context.return_value = Ok(sample_document_context)
        mock_metadata_store.list_chunks.return_value = Ok(
            [
                StoredChunkMetadata(
                    chunk_id="chunk-001",
                    document_id=sample_document.id,
                    chunk_index=0,
                    heading_path=("Linear Algebra", "Matrices"),
                    heading_level=2,
                    language="en",
                    content_text="A matrix is a rectangular array of numbers.",
                )
            ]
        )
        mock_llm_client.generate.side_effect = [
            Ok("not valid json"),
            Ok(
                '{"cards":[{"title":"Matrices","bullets":["Rectangular arrays","Useful for transformations"],'
                '"key_insight":"Matrices compress structured numeric relationships."}]}'
            ),
            Ok(
                '{"questions":[{"question":"What is a matrix?","choices":["A number line","A rectangular array",'
                '"A triangle","A function"],"correct_index":1,"explanation":"That is the standard definition.",'
                '"difficulty":"easy"}]}'
            ),
        ]

        use_case = RunEnrichmentUseCase(
            metadata_store=mock_metadata_store,
            llm_client=mock_llm_client,
        )

        result = use_case.execute(RunEnrichmentRequest(document_id=sample_document.id))

        assert result.is_ok()
        assert mock_llm_client.generate.call_count == 3

    def test_document_without_headings_uses_synthetic_section_and_marks_enriched(
        self,
        sample_document,
        sample_document_context,
        mock_metadata_store,
        mock_llm_client,
    ):
        mock_metadata_store.get_document_context.return_value = Ok(sample_document_context)
        mock_metadata_store.list_chunks.return_value = Ok(
            [
                StoredChunkMetadata(
                    chunk_id="chunk-001",
                    document_id=sample_document.id,
                    chunk_index=0,
                    heading_path=(),
                    heading_level=0,
                    language="en",
                    content_text="Plain section content without a heading.",
                )
            ]
        )
        mock_metadata_store.persist_enrichment_batch.return_value = Ok(None)
        mock_llm_client.generate.side_effect = [
            Ok(
                '{"cards":[{"title":"Section summary","bullets":["Plain section content"],'
                '"key_insight":"Even headingless content can become a card."}]}'
            ),
            Ok(
                '{"questions":[{"question":"What is missing from this section?","choices":["Text","A heading",'
                '"A verb","An object"],"correct_index":1,"explanation":"The content exists but no heading was provided.",'
                '"difficulty":"easy"}]}'
            ),
        ]

        use_case = RunEnrichmentUseCase(
            metadata_store=mock_metadata_store,
            llm_client=mock_llm_client,
        )

        result = use_case.execute(RunEnrichmentRequest(document_id=sample_document.id))

        assert result.is_ok()
        persist_kwargs = mock_metadata_store.persist_enrichment_batch.call_args.kwargs
        assert persist_kwargs["outbox_event_type"] is None
        lesson_cards = persist_kwargs["lesson_cards"]
        quiz_items = persist_kwargs["quiz_items"]
        assert lesson_cards[0].heading_path == ("Section 1",)
        assert quiz_items[0].heading_path == ("Section 1",)
        assert mock_metadata_store.update_document_status.call_args_list[-1].args == (
            sample_document.id,
            IngestionStatus.ENRICHED,
        )

    def test_persist_failure_returns_done_for_retry(
        self,
        sample_document,
        sample_document_context,
        mock_metadata_store,
        mock_llm_client,
    ):
        mock_metadata_store.get_document_context.return_value = Ok(sample_document_context)
        mock_metadata_store.list_chunks.return_value = Ok(
            [
                StoredChunkMetadata(
                    chunk_id="chunk-001",
                    document_id=sample_document.id,
                    chunk_index=0,
                    heading_path=("Linear Algebra",),
                    heading_level=1,
                    language="en",
                    content_text="Vector spaces are closed under addition and scalar multiplication.",
                )
            ]
        )
        mock_metadata_store.persist_enrichment_batch.return_value = Err(RuntimeError("sql down"))
        mock_llm_client.generate.side_effect = [
            Ok(
                '{"cards":[{"title":"Vector spaces","bullets":["Closed under addition"],'
                '"key_insight":"Closure properties define the space."}]}'
            ),
            Ok(
                '{"questions":[{"question":"Which operation preserves a vector space?","choices":["Coloring","Addition",'
                '"Sorting","Printing"],"correct_index":1,"explanation":"Closure under addition is required.",'
                '"difficulty":"medium"}]}'
            ),
        ]

        use_case = RunEnrichmentUseCase(
            metadata_store=mock_metadata_store,
            llm_client=mock_llm_client,
        )

        result = use_case.execute(RunEnrichmentRequest(document_id=sample_document.id))

        assert result.is_err()
        assert mock_metadata_store.update_document_status.call_args_list[0].args == (
            sample_document.id,
            IngestionStatus.ENRICHING,
        )
        assert mock_metadata_store.update_document_status.call_args_list[-1].kwargs == {
            "document_id": sample_document.id,
            "status": IngestionStatus.DONE,
            "error_msg": "sql down",
        }
