import uuid

from document_chunk.domain.exceptions import DocumentStaleError
from document_chunk.domain.outbox_events import OutboxEventType
from document_chunk.domain.ports.metadata_store import StoredChunkMetadata
from document_chunk.shared.result import Err, Ok

from tests.support import RecordingResultPublisher
from worker.use_cases.run_enrichment import RunEnrichmentRequest, RunEnrichmentUseCase

_CARDS = (
    '{"cards":[{"title":"Khái niệm đạo hàm","bullets":["Mô tả tốc độ thay đổi","Áp dụng trong tối ưu"],'
    '"key_insight":"Đạo hàm biến sự thay đổi thành đại lượng đo được."}]}'
)
_QUIZ = (
    '{"questions":[{"question":"Đạo hàm dùng để làm gì?","choices":["Trang trí biểu đồ","Đo tốc độ thay đổi",'
    '"Đổi đơn vị","Đếm số trang"],"correct_index":1,"explanation":"Đạo hàm đo tốc độ thay đổi tức thời.",'
    '"difficulty":"medium"}]}'
)


class _FailOnKind(RecordingResultPublisher):
    """Redis nhận mọi kết quả trừ một kind: thử đúng một bước publish hỏng."""

    def __init__(self, kind: str, error: Exception) -> None:
        super().__init__()
        self._kind, self._error = kind, error

    def publish_sync(self, kind, payload, **kwargs):
        if kind == self._kind:
            raise self._error
        return super().publish_sync(kind, payload, **kwargs)


def _statuses(publisher):
    return [
        env["payload"]["status"]
        for env in publisher.envelopes
        if env["kind"] == "DOCUMENT_PROCESSING_RESULT"
    ]


def _content(publisher):
    found = [env for env in publisher.envelopes if env["kind"] == "ENRICHMENT_CONTENT_RESULT"]
    assert len(found) <= 1
    return found[0] if found else None


def _assert_no_direct_document_writes(store):
    # documents, lesson_cards, quiz_items là bảng của core-api.
    store.update_document_status.assert_not_called()
    store.set_document_chapter.assert_not_called()


def _two_chunks(document_id):
    return [
        StoredChunkMetadata(
            chunk_id="chunk-001",
            document_id=document_id,
            chunk_index=0,
            heading_path=("Giải tích", "Đạo hàm"),
            heading_level=2,
            language="vi",
            content_text="Đạo hàm mô tả tốc độ thay đổi tức thời.",
        ),
        StoredChunkMetadata(
            chunk_id="chunk-002",
            document_id=document_id,
            chunk_index=1,
            heading_path=("Giải tích", "Đạo hàm"),
            heading_level=2,
            language="vi",
            content_text="Đạo hàm giúp phân tích xu hướng và tối ưu.",
        ),
    ]


class TestRunEnrichmentUseCase:
    def test_publishes_content_and_writes_only_ai_data_with_the_outbox_event(
        self,
        sample_document,
        sample_document_context,
        mock_metadata_store,
        mock_llm_client,
    ):
        mock_metadata_store.get_document_context.return_value = Ok(sample_document_context)
        mock_metadata_store.list_chunks.return_value = Ok(_two_chunks(sample_document.id))
        mock_llm_client.generate.side_effect = [Ok(_CARDS), Ok(_QUIZ)]
        publisher = RecordingResultPublisher()

        use_case = RunEnrichmentUseCase(
            metadata_store=mock_metadata_store,
            llm_client=mock_llm_client,
            result_publisher=publisher,
            max_section_chars=500,
        )

        result = use_case.execute(
            RunEnrichmentRequest(document_id=sample_document.id, processing_generation=4, attempt=2)
        )

        assert result.is_ok()
        response = result.unwrap()
        assert (response.concept_count, response.mention_count) == (2, 4)
        assert (response.card_count, response.quiz_count) == (1, 1)
        assert mock_llm_client.generate.call_count == 2
        _assert_no_direct_document_writes(mock_metadata_store)

        # Thứ tự: ENRICHING → học liệu; GENERATED_DRAFT để OutboxProjector báo
        # sau khi chiếu graph (có sự kiện outbox).
        assert [env["kind"] for env in publisher.envelopes] == [
            "DOCUMENT_PROCESSING_RESULT",
            "ENRICHMENT_CONTENT_RESULT",
        ]
        assert _statuses(publisher) == [
            {"value": "ENRICHING", "stage": "enrichment", "processing_generation": 4, "attempt": 2}
        ]
        assert publisher.envelopes[0]["result_id"] == (
            "DOCUMENT_PROCESSING_RESULT.doc-001.g4-enrichment2-ENRICHING"
        )

        content = _content(publisher)
        # Tất định theo lượt: job chạy lại sinh lại đúng id, BullMQ bỏ bản trùng.
        assert content["result_id"] == "ENRICHMENT_CONTENT_RESULT.doc-001.g4"
        payload = content["payload"]
        assert payload["document_id"] == sample_document.id
        assert payload["processing_generation"] == 4
        assert payload["model_id"] == "gemini-test"
        [card] = payload["lesson_cards"]
        uuid.UUID(card["card_id"])
        assert card["primary_chunk_id"] == "chunk-001"
        assert card["source_chunk_ids"] == ["chunk-001", "chunk-002"]
        assert card["heading_path"] == ["Giải tích", "Đạo hàm"]
        assert card["title"] == "Khái niệm đạo hàm"
        assert card["bullets"] == ["Mô tả tốc độ thay đổi", "Áp dụng trong tối ưu"]
        assert card["card_index"] == 0
        [item] = payload["quiz_items"]
        uuid.UUID(item["quiz_id"])
        assert item["choices"][item["correct_index"]] == "Đo tốc độ thay đổi"
        assert item["source_chunk_ids"] == ["chunk-001", "chunk-002"]

        # Phần AI + sự kiện outbox: một lần gọi = một transaction.
        persist = mock_metadata_store.persist_concept_enrichment.call_args.kwargs
        assert persist["document_id"] == sample_document.id
        assert len(persist["concepts"]) == 2
        assert len(persist["chunk_concepts"]) == 4
        assert persist["outbox_event_type"] == OutboxEventType.CONCEPT_GRAPH_PROJECT
        assert persist["outbox_payload"]["processing_generation"] == 4
        assert persist["outbox_payload"]["enrichment_attempt"] == 2
        assert "lesson_cards" not in persist and "quiz_items" not in persist

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
            result_publisher=RecordingResultPublisher(),
        )

        result = use_case.execute(RunEnrichmentRequest(document_id=sample_document.id))

        assert result.is_ok()
        assert mock_llm_client.generate.call_count == 3

    def test_document_without_headings_reports_generated_draft_itself(
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
        mock_metadata_store.persist_concept_enrichment.return_value = Ok(None)
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
        publisher = RecordingResultPublisher()

        use_case = RunEnrichmentUseCase(
            metadata_store=mock_metadata_store,
            llm_client=mock_llm_client,
            result_publisher=publisher,
        )

        result = use_case.execute(RunEnrichmentRequest(document_id=sample_document.id))

        assert result.is_ok()
        persist = mock_metadata_store.persist_concept_enrichment.call_args.kwargs
        assert persist["outbox_event_type"] is None
        payload = _content(publisher)["payload"]
        assert payload["lesson_cards"][0]["heading_path"] == ["Section 1"]
        # Không có sự kiện chiếu graph: enrichment tự báo GENERATED_DRAFT, sau học liệu.
        assert [env["kind"] for env in publisher.envelopes] == [
            "DOCUMENT_PROCESSING_RESULT",
            "ENRICHMENT_CONTENT_RESULT",
            "DOCUMENT_PROCESSING_RESULT",
        ]
        assert [s["value"] for s in _statuses(publisher)] == ["ENRICHING", "GENERATED_DRAFT"]
        _assert_no_direct_document_writes(mock_metadata_store)

    def test_ai_persist_failure_falls_back_to_indexed_and_fails_the_job(
        self,
        sample_document,
        sample_document_context,
        mock_metadata_store,
        mock_llm_client,
    ):
        mock_metadata_store.get_document_context.return_value = Ok(sample_document_context)
        mock_metadata_store.list_chunks.return_value = Ok(_two_chunks(sample_document.id))
        mock_metadata_store.persist_concept_enrichment.return_value = Err(RuntimeError("sql down"))
        mock_llm_client.generate.side_effect = [Ok(_CARDS), Ok(_QUIZ)]
        publisher = RecordingResultPublisher()

        result = RunEnrichmentUseCase(
            metadata_store=mock_metadata_store,
            llm_client=mock_llm_client,
            result_publisher=publisher,
        ).execute(RunEnrichmentRequest(document_id=sample_document.id))

        assert result.is_err()
        statuses = _statuses(publisher)
        assert [s["value"] for s in statuses] == ["ENRICHING", "INDEXED"]
        assert statuses[-1]["error"] == "sql down"
        assert statuses[-1]["stage"] == "enrichment"
        _assert_no_direct_document_writes(mock_metadata_store)

    def test_content_publish_failure_fails_the_job_before_writing_ai_data(
        self,
        sample_document,
        sample_document_context,
        mock_metadata_store,
        mock_llm_client,
    ):
        mock_metadata_store.get_document_context.return_value = Ok(sample_document_context)
        mock_metadata_store.list_chunks.return_value = Ok(_two_chunks(sample_document.id))
        mock_llm_client.generate.side_effect = [Ok(_CARDS), Ok(_QUIZ)]
        publisher = _FailOnKind("ENRICHMENT_CONTENT_RESULT", ConnectionError("redis down"))

        result = RunEnrichmentUseCase(
            metadata_store=mock_metadata_store,
            llm_client=mock_llm_client,
            result_publisher=publisher,
        ).execute(RunEnrichmentRequest(document_id=sample_document.id))

        assert result.is_err()
        assert isinstance(result.error, ConnectionError)
        mock_metadata_store.persist_concept_enrichment.assert_not_called()
        assert [s["value"] for s in _statuses(publisher)] == ["ENRICHING", "INDEXED"]

    def test_status_publish_failure_stops_before_any_llm_call(
        self,
        sample_document,
        sample_document_context,
        mock_metadata_store,
        mock_llm_client,
    ):
        mock_metadata_store.get_document_context.return_value = Ok(sample_document_context)
        publisher = RecordingResultPublisher(fail_with=TimeoutError("no ack"))

        result = RunEnrichmentUseCase(
            metadata_store=mock_metadata_store,
            llm_client=mock_llm_client,
            result_publisher=publisher,
        ).execute(RunEnrichmentRequest(document_id=sample_document.id))

        assert result.is_err()
        assert isinstance(result.error, TimeoutError)
        mock_llm_client.generate.assert_not_called()
        mock_metadata_store.persist_concept_enrichment.assert_not_called()

    def test_document_deleted_meanwhile_stops_without_publishing(
        self,
        sample_document,
        sample_document_context,
        mock_metadata_store,
        mock_llm_client,
    ):
        """Delete-race guard: trước đây là UPDATE ... AND deleted_at IS NULL trả 0 dòng."""
        mock_metadata_store.get_document_context.side_effect = [
            Ok(sample_document_context),
            Ok(None),
        ]
        publisher = RecordingResultPublisher()

        result = RunEnrichmentUseCase(
            metadata_store=mock_metadata_store,
            llm_client=mock_llm_client,
            result_publisher=publisher,
        ).execute(RunEnrichmentRequest(document_id=sample_document.id))

        assert result.is_err()
        assert isinstance(result.error, DocumentStaleError)
        assert publisher.envelopes == []
        mock_llm_client.generate.assert_not_called()

    def test_uuid_chunk_ids_from_postgres_are_sent_as_strings(
        self,
        sample_document,
        sample_document_context,
        mock_metadata_store,
        mock_llm_client,
    ):
        """list_chunks trả uuid.UUID; JSON của envelope chỉ nhận str."""
        chunk_id = uuid.uuid4()
        mock_metadata_store.get_document_context.return_value = Ok(sample_document_context)
        mock_metadata_store.list_chunks.return_value = Ok(
            [
                StoredChunkMetadata(
                    chunk_id=chunk_id,
                    document_id=sample_document.id,
                    chunk_index=0,
                    heading_path=("Giải tích",),
                    heading_level=1,
                    language="vi",
                    content_text="Đạo hàm mô tả tốc độ thay đổi tức thời.",
                )
            ]
        )
        mock_llm_client.generate.side_effect = [Ok(_CARDS), Ok(_QUIZ)]
        publisher = RecordingResultPublisher()

        result = RunEnrichmentUseCase(
            metadata_store=mock_metadata_store,
            llm_client=mock_llm_client,
            result_publisher=publisher,
        ).execute(RunEnrichmentRequest(document_id=sample_document.id))

        assert result.is_ok(), str(result.error)
        payload = _content(publisher)["payload"]
        assert payload["lesson_cards"][0]["source_chunk_ids"] == [str(chunk_id)]
        assert payload["quiz_items"][0]["primary_chunk_id"] == str(chunk_id)


class TestSectionFailureIsolation:
    """
    O1 — một section lỗi làm mất card/quiz của cả tài liệu.

    Vòng lặp cũ `return self._fail_to_done(...)` ngay khi một section lỗi, nên
    mọi thứ đã sinh cho các section trước đó bị vứt — sau khi bản nháp cũ đã bị
    soft-delete. Status vẫn là DONE nên nhìn từ ngoài như thành công.
    """

    def _chunks(self, document_id, sections):
        return [
            StoredChunkMetadata(
                chunk_id=f"chunk-{i}",
                document_id=document_id,
                chunk_index=i,
                heading_path=(heading,),
                heading_level=1,
                language="vi",
                content_text=f"Nội dung của {heading} đủ dài để sinh thẻ học.",
            )
            for i, heading in enumerate(sections)
        ]

    def _cards(self, name):
        return Ok(
            f'{{"cards":[{{"title":"{name}","bullets":["a","b"],'
            f'"key_insight":"{name} quan trọng."}}]}}'
        )

    def _quiz(self, name):
        return Ok(
            f'{{"questions":[{{"question":"{name}?","choices":["A","B","C","D"],'
            f'"correct_index":0,"explanation":"vì {name}","difficulty":"easy"}}]}}'
        )

    def test_one_failing_section_keeps_the_others(
        self, sample_document, sample_document_context, mock_metadata_store, mock_llm_client,
    ):
        mock_metadata_store.get_document_context.return_value = Ok(sample_document_context)
        mock_metadata_store.list_chunks.return_value = Ok(
            self._chunks(sample_document.id, ["Chương 1", "Chương 2", "Chương 3"])
        )
        # Section 2 hỏng cả thẻ lẫn quiz; mỗi lần hỏng tiêu hai lượt gọi
        # (bản đầu + một lần sửa JSON). Hai section còn lại bình thường.
        mock_llm_client.generate.side_effect = [
            self._cards("Chương 1"), self._quiz("Chương 1"),
            Ok("không phải JSON"), Ok("vẫn không phải JSON"),
            Ok("không phải JSON"), Ok("vẫn không phải JSON"),
            self._cards("Chương 3"), self._quiz("Chương 3"),
        ]
        publisher = RecordingResultPublisher()

        use_case = RunEnrichmentUseCase(
            metadata_store=mock_metadata_store,
            llm_client=mock_llm_client,
            result_publisher=publisher,
        )
        result = use_case.execute(RunEnrichmentRequest(document_id=sample_document.id))

        assert result.is_ok(), str(result.error)
        titles = {c["title"] for c in _content(publisher)["payload"]["lesson_cards"]}
        assert titles == {"Chương 1", "Chương 3"}
        mock_metadata_store.persist_concept_enrichment.assert_called_once()

    def test_all_sections_failing_does_not_overwrite_with_nothing(
        self, sample_document, sample_document_context, mock_metadata_store, mock_llm_client,
    ):
        """Không còn gì để lưu thì dừng lại, đừng ghi đè nội dung đang có bằng rỗng."""
        mock_metadata_store.get_document_context.return_value = Ok(sample_document_context)
        mock_metadata_store.list_chunks.return_value = Ok(
            self._chunks(sample_document.id, ["Chương 1", "Chương 2"])
        )
        mock_llm_client.generate.side_effect = [Ok("rác")] * 8
        publisher = RecordingResultPublisher()

        use_case = RunEnrichmentUseCase(
            metadata_store=mock_metadata_store,
            llm_client=mock_llm_client,
            result_publisher=publisher,
        )
        result = use_case.execute(RunEnrichmentRequest(document_id=sample_document.id))

        assert result.is_err()
        assert _content(publisher) is None
        mock_metadata_store.persist_concept_enrichment.assert_not_called()
        assert [s["value"] for s in _statuses(publisher)] == ["ENRICHING", "INDEXED"]
