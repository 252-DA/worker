"""
Tests for worker/use_cases/generate_curriculum_quiz.py.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from document_chunk.domain.ports.metadata_store import (
    StoredAssessment,
    StoredChapter,
    StoredChunkMetadata,
    StoredCourse,
    StoredLearningOutcome,
)
from document_chunk.shared.result import Err, Ok
from worker.use_cases.generate_curriculum_quiz import (
    GenerateCurriculumQuizRequest,
    GenerateCurriculumQuizUseCase,
)


class TestGenerateCurriculumQuizRequest:
    def test_defaults(self):
        req = GenerateCurriculumQuizRequest(
            course_id="course-001",
            target_kind="lo",
            target_code="L.O.1.1",
        )
        assert req.course_id == "course-001"
        assert req.target_kind == "lo"
        assert req.target_code == "L.O.1.1"
        assert req.count == 5
        assert req.style == "quiz"
        assert req.bloom_level is None

    def test_full_request(self):
        req = GenerateCurriculumQuizRequest(
            course_id="course-001",
            target_kind="chapter",
            target_code="C01",
            style="midterm",
            bloom_level="apply",
            count=10,
        )
        assert req.target_kind == "chapter"
        assert req.count == 10
        assert req.style == "midterm"
        assert req.bloom_level == "apply"


class TestGenerateCurriculumQuizUseCase:
    def test_quiz_from_lo(self, mock_metadata_store, mock_llm_client):
        lo_id = "course-001:L.O.1.1"
        mock_metadata_store.get_curriculum.return_value = Ok((
            StoredCourse(course_id="course-001", code="CS101", title_vi="Nhập môn AI"),
            [],
            [StoredLearningOutcome(
                lo_id=lo_id,
                course_id="course-001",
                code="L.O.1.1",
                parent_code=None,
                statement_vi="Hiểu khái niệm AI",
                bloom_level="understand",
            )],
            [],
        ))
        mock_metadata_store.list_chunks_for_lo.return_value = Ok([
            StoredChunkMetadata(
                chunk_id="chunk-001",
                document_id="doc-001",
                chunk_index=0,
                heading_path=("Giới thiệu",),
                content_text="AI là một lĩnh vực của khoa học máy tính.",
            )
        ])
        mock_metadata_store.persist_curriculum_quiz_items.return_value = Ok(None)
        mock_llm_client.model_id = "gemini-test"
        mock_llm_client.generate.return_value = Ok(
            '{"questions":[{"question":"AI là gì?","choices":["A","B","C","D"],'
            '"correct_index":0,"explanation":"...","difficulty":"easy","lo_alignment_rationale":"..."}]}'
        )

        use_case = GenerateCurriculumQuizUseCase(
            metadata_store=mock_metadata_store,
            llm_client=mock_llm_client,
        )

        result = use_case.execute(
            GenerateCurriculumQuizRequest(
                course_id="course-001",
                target_kind="lo",
                target_code="L.O.1.1",
                count=1,
            )
        )

        assert result.is_ok()
        resp = result.unwrap()
        assert resp.course_id == "course-001"
        assert resp.lo_id == lo_id
        assert resp.quiz_count == 1

    def test_quiz_from_chapter(self, mock_metadata_store, mock_llm_client):
        lo_id = "lo-ch01"
        mock_metadata_store.list_los_by_chapter.return_value = Ok([
            StoredLearningOutcome(
                lo_id=lo_id,
                course_id="course-001",
                code="L.O.1.1",
                parent_code=None,
                statement_vi="Hiểu cơ bản",
                bloom_level="understand",
            )
        ])
        mock_metadata_store.get_curriculum.return_value = Ok((
            StoredCourse(course_id="course-001", code="CS101", title_vi="Nhập môn AI"),
            [StoredChapter(chapter_id="ch01", course_id="course-001", code="C01", title="Chương 1")],
            [StoredLearningOutcome(
                lo_id=lo_id,
                course_id="course-001",
                code="L.O.1.1",
                parent_code=None,
                statement_vi="Hiểu cơ bản",
                bloom_level="understand",
            )],
            [],
        ))
        mock_metadata_store.list_chunks_for_lo.return_value = Ok([
            StoredChunkMetadata(
                chunk_id="chunk-chapter",
                document_id="doc-chapter",
                chunk_index=0,
                content_text="Nội dung của chương.",
            )
        ])
        mock_metadata_store.persist_curriculum_quiz_items.return_value = Ok(None)
        mock_llm_client.model_id = "gemini-test"
        mock_llm_client.generate.return_value = Ok(
            '{"questions":[{"question":"Q?","choices":["W","X","Y","Z"],'
            '"correct_index":2,"explanation":"...","difficulty":"medium","lo_alignment_rationale":"..."}]}'
        )

        use_case = GenerateCurriculumQuizUseCase(
            metadata_store=mock_metadata_store,
            llm_client=mock_llm_client,
        )

        result = use_case.execute(
            GenerateCurriculumQuizRequest(
                course_id="course-001",
                target_kind="chapter",
                target_code="C01",
            )
        )

        assert result.is_ok()

    def test_curriculum_not_found(self, mock_metadata_store, mock_llm_client):
        mock_metadata_store.get_curriculum.return_value = Ok(None)

        use_case = GenerateCurriculumQuizUseCase(
            metadata_store=mock_metadata_store,
            llm_client=mock_llm_client,
        )

        result = use_case.execute(
            GenerateCurriculumQuizRequest(
                course_id="course-nonexistent",
                target_kind="lo",
                target_code="L.O.9.9",
            )
        )

        assert result.is_err()

    def test_no_los_for_target(self, mock_metadata_store, mock_llm_client):
        mock_metadata_store.list_los_by_chapter.return_value = Ok([])

        use_case = GenerateCurriculumQuizUseCase(
            metadata_store=mock_metadata_store,
            llm_client=mock_llm_client,
        )

        result = use_case.execute(
            GenerateCurriculumQuizRequest(
                course_id="course-001",
                target_kind="chapter",
                target_code="C99",
            )
        )

        assert result.is_err()

    def test_llm_failure_propagates(self, mock_metadata_store, mock_llm_client):
        lo_id = "course-001:L.O.1.1"
        mock_metadata_store.get_curriculum.return_value = Ok((
            StoredCourse(course_id="course-001", code="CS101", title_vi="Nhập môn AI"),
            [],
            [StoredLearningOutcome(
                lo_id=lo_id, course_id="course-001", code="L.O.1.1",
                parent_code=None, statement_vi="Hiểu AI", bloom_level="understand",
            )],
            [],
        ))
        mock_metadata_store.list_chunks_for_lo.return_value = Ok([])
        mock_llm_client.generate.return_value = Err(RuntimeError("LLM timeout"))

        use_case = GenerateCurriculumQuizUseCase(
            metadata_store=mock_metadata_store,
            llm_client=mock_llm_client,
        )

        result = use_case.execute(
            GenerateCurriculumQuizRequest(
                course_id="course-001",
                target_kind="lo",
                target_code="L.O.1.1",
            )
        )

        assert result.is_err()

    def test_unknown_target_kind(self, mock_metadata_store, mock_llm_client):
        use_case = GenerateCurriculumQuizUseCase(
            metadata_store=mock_metadata_store,
            llm_client=mock_llm_client,
        )

        result = use_case.execute(
            GenerateCurriculumQuizRequest(
                course_id="course-001",
                target_kind="invalid_kind",
                target_code="X",
            )
        )

        assert result.is_err()

    def test_uses_mcp_context_and_filters_unknown_source_ids(
        self,
        mock_metadata_store,
        mock_llm_client,
    ):
        lo_id = "course-001:L.O.1.1"
        mock_metadata_store.get_curriculum.return_value = Ok((
            StoredCourse(course_id="course-001", code="CS101", title_vi="Nhập môn AI"),
            [],
            [StoredLearningOutcome(
                lo_id=lo_id,
                course_id="course-001",
                code="L.O.1.1",
                parent_code=None,
                statement_vi="Hiểu khái niệm AI",
                bloom_level="understand",
            )],
            [],
        ))
        mock_metadata_store.persist_curriculum_quiz_items.return_value = Ok(None)
        mock_llm_client.model_id = "gemini-test"
        mock_llm_client.generate.return_value = Ok(
            '{"questions":[{"question":"AI là gì?","choices":["A","B","C","D"],'
            '"correct_index":0,"explanation":"...","difficulty":"easy",'
            '"lo_alignment_rationale":"...","source_chunk_ids":'
            '["hallucinated","chunk-mcp"]}]}'
        )
        context_client = MagicMock()
        context_client.retrieve_quiz_context.return_value = SimpleNamespace(
            chunks=[
                SimpleNamespace(
                    chunk_id="chunk-mcp",
                    document_id="doc-mcp",
                    page_number=7,
                    content="Grounded MCP context",
                )
            ]
        )
        use_case = GenerateCurriculumQuizUseCase(
            metadata_store=mock_metadata_store,
            llm_client=mock_llm_client,
            context_client=context_client,
        )

        result = use_case.execute(
            GenerateCurriculumQuizRequest(
                course_id="course-001",
                target_kind="lo",
                target_code="L.O.1.1",
                count=1,
            )
        )

        assert result.is_ok()
        persist_kwargs = (
            mock_metadata_store.persist_curriculum_quiz_items.call_args.kwargs
        )
        assert persist_kwargs["lo_id"] == lo_id
        assert persist_kwargs["quiz_items"][0].source_chunk_ids == ("chunk-mcp",)
        # The verifier pass (added alongside Reflexion) also calls generate(), so
        # check every call rather than assuming the quiz prompt is the last one.
        assert any(
            "Grounded MCP context" in call.args[0]
            for call in mock_llm_client.generate.call_args_list
        )

    def test_quiz_persistence_failure_is_returned(
        self,
        mock_metadata_store,
        mock_llm_client,
    ):
        lo_id = "course-001:L.O.1.1"
        mock_metadata_store.get_curriculum.return_value = Ok((
            StoredCourse(course_id="course-001", code="CS101", title_vi="Nhập môn AI"),
            [],
            [StoredLearningOutcome(
                lo_id=lo_id,
                course_id="course-001",
                code="L.O.1.1",
                parent_code=None,
                statement_vi="Hiểu khái niệm AI",
                bloom_level="understand",
            )],
            [],
        ))
        mock_metadata_store.list_chunks_for_lo.return_value = Ok([
            StoredChunkMetadata(
                chunk_id="chunk-001",
                document_id="doc-001",
                chunk_index=0,
                content_text="AI là một lĩnh vực của khoa học máy tính.",
            )
        ])
        mock_metadata_store.persist_curriculum_quiz_items.return_value = Err(
            RuntimeError("database unavailable")
        )
        mock_llm_client.generate.return_value = Ok(
            '{"questions":[{"question":"AI là gì?","choices":["A","B","C","D"],'
            '"correct_index":0,"explanation":"...","difficulty":"easy",'
            '"lo_alignment_rationale":"..."}]}'
        )

        result = GenerateCurriculumQuizUseCase(
            metadata_store=mock_metadata_store,
            llm_client=mock_llm_client,
        ).execute(
            GenerateCurriculumQuizRequest(
                course_id="course-001",
                target_kind="lo",
                target_code="L.O.1.1",
                count=1,
            )
        )

        assert result.is_err()
        assert "database unavailable" in str(result.error)


class TestVerifyAndRefine:
    """Chain-of-Verification + Reflexion pass added between generation and persistence."""

    def _curriculum_ok(self, mock_metadata_store, lo_id, source_text="Some source text."):
        mock_metadata_store.get_curriculum.return_value = Ok((
            StoredCourse(course_id="course-001", code="CS101", title_vi="Nhập môn AI"),
            [],
            [StoredLearningOutcome(
                lo_id=lo_id,
                course_id="course-001",
                code="L.O.1.1",
                parent_code=None,
                statement_vi="Hiểu khái niệm AI",
                bloom_level="understand",
            )],
            [],
        ))
        mock_metadata_store.list_chunks_for_lo.return_value = Ok([
            StoredChunkMetadata(
                chunk_id="chunk-001",
                document_id="doc-001",
                chunk_index=0,
                content_text=source_text,
            )
        ])
        mock_metadata_store.persist_curriculum_quiz_items.return_value = Ok(None)

    def test_verifier_passes_through_when_all_supported(
        self, mock_metadata_store, mock_llm_client
    ):
        lo_id = "course-001:L.O.1.1"
        self._curriculum_ok(mock_metadata_store, lo_id)
        mock_llm_client.model_id = "gemini-test"
        mock_llm_client.generate.side_effect = [
            Ok(
                '{"questions":[{"question":"AI là gì?","choices":["A","B","C","D"],'
                '"correct_index":0,"explanation":"...","difficulty":"easy"}]}'
            ),
            Ok('{"verdicts":[{"question_index":0,"verdict":"supported","reason":"ok"}]}'),
        ]

        use_case = GenerateCurriculumQuizUseCase(
            metadata_store=mock_metadata_store,
            llm_client=mock_llm_client,
        )
        result = use_case.execute(
            GenerateCurriculumQuizRequest(
                course_id="course-001", target_kind="lo", target_code="L.O.1.1", count=1,
            )
        )

        assert result.is_ok()
        # generate() + verify() only — a supported verdict never triggers a regen call.
        assert mock_llm_client.generate.call_count == 2
        persisted = mock_metadata_store.persist_curriculum_quiz_items.call_args.kwargs[
            "quiz_items"
        ]
        assert persisted[0].question == "AI là gì?"

    def test_reflexion_regenerates_failing_question(
        self, mock_metadata_store, mock_llm_client
    ):
        lo_id = "course-001:L.O.1.1"
        self._curriculum_ok(
            mock_metadata_store, lo_id, "Dijkstra requires non-negative edge weights."
        )
        mock_llm_client.model_id = "gemini-test"
        mock_llm_client.generate.side_effect = [
            Ok(
                '{"questions":[{"question":"Does Dijkstra allow negative weights?",'
                '"choices":["Yes","No","Sometimes","Never"],"correct_index":0,'
                '"explanation":"wrong claim","difficulty":"medium"}]}'
            ),
            Ok(
                '{"verdicts":[{"question_index":0,"verdict":"contradicted",'
                '"reason":"Source says weights must be non-negative"}]}'
            ),
            Ok(
                '{"question":"What edge-weight constraint does Dijkstra require?",'
                '"choices":["Non-negative","Negative","Zero only","Any"],"correct_index":0,'
                '"explanation":"Per source, weights must be non-negative",'
                '"difficulty":"medium"}'
            ),
            Ok(
                '{"verdicts":[{"question_index":0,"verdict":"supported",'
                '"reason":"matches source"}]}'
            ),
        ]

        use_case = GenerateCurriculumQuizUseCase(
            metadata_store=mock_metadata_store,
            llm_client=mock_llm_client,
        )
        result = use_case.execute(
            GenerateCurriculumQuizRequest(
                course_id="course-001", target_kind="lo", target_code="L.O.1.1", count=1,
            )
        )

        assert result.is_ok()
        # generate, verify(fail), regenerate, verify(pass).
        assert mock_llm_client.generate.call_count == 4
        persisted = mock_metadata_store.persist_curriculum_quiz_items.call_args.kwargs[
            "quiz_items"
        ]
        assert persisted[0].question == "What edge-weight constraint does Dijkstra require?"

    def test_verifier_error_does_not_block_persistence(
        self, mock_metadata_store, mock_llm_client
    ):
        """A verifier that can't be parsed is inconclusive, not a rejection —
        the original question still reaches the human reviewer."""
        lo_id = "course-001:L.O.1.1"
        self._curriculum_ok(mock_metadata_store, lo_id)
        mock_llm_client.model_id = "gemini-test"
        mock_llm_client.generate.side_effect = [
            Ok(
                '{"questions":[{"question":"AI là gì?","choices":["A","B","C","D"],'
                '"correct_index":0,"explanation":"...","difficulty":"easy"}]}'
            ),
            Ok("not valid json at all"),
            Ok("still not valid json"),
        ]

        use_case = GenerateCurriculumQuizUseCase(
            metadata_store=mock_metadata_store,
            llm_client=mock_llm_client,
        )
        result = use_case.execute(
            GenerateCurriculumQuizRequest(
                course_id="course-001", target_kind="lo", target_code="L.O.1.1", count=1,
            )
        )

        assert result.is_ok()
        persisted = mock_metadata_store.persist_curriculum_quiz_items.call_args.kwargs[
            "quiz_items"
        ]
        assert persisted[0].question == "AI là gì?"

    def test_verify_exhausts_retries_and_drops_the_question(
        self, mock_metadata_store, mock_llm_client
    ):
        """
        V2: hết lượt sửa mà câu vẫn không bám được nguồn thì **không lưu**.

        Chính sách cũ lưu lần thử cuối kèm một dòng log, tức là đẩy việc phát
        hiện câu sai sang người duyệt — ngược với mục đích của vòng kiểm tra.
        Ở đây chỉ có một câu và nó bị loại, nên không còn gì để lưu và request
        báo lỗi thay vì lưu một câu đã biết là sai.
        """
        lo_id = "course-001:L.O.1.1"
        self._curriculum_ok(mock_metadata_store, lo_id)
        mock_llm_client.model_id = "gemini-test"
        always_contradicted = Ok(
            '{"verdicts":[{"question_index":0,"verdict":"contradicted","reason":"still wrong"}]}'
        )
        mock_llm_client.generate.side_effect = [
            Ok(
                '{"questions":[{"question":"Q0","choices":["A","B","C","D"],'
                '"correct_index":0,"explanation":"e0","difficulty":"easy"}]}'
            ),
            always_contradicted,
            Ok(
                '{"question":"Q1","choices":["A","B","C","D"],"correct_index":0,'
                '"explanation":"e1","difficulty":"easy"}'
            ),
            always_contradicted,
            Ok(
                '{"question":"Q2","choices":["A","B","C","D"],"correct_index":0,'
                '"explanation":"e2","difficulty":"easy"}'
            ),
            always_contradicted,
        ]

        use_case = GenerateCurriculumQuizUseCase(
            metadata_store=mock_metadata_store,
            llm_client=mock_llm_client,
            max_verification_retries=2,
        )
        result = use_case.execute(
            GenerateCurriculumQuizRequest(
                course_id="course-001", target_kind="lo", target_code="L.O.1.1", count=1,
            )
        )

        assert result.is_err()
        mock_metadata_store.persist_curriculum_quiz_items.assert_not_called()

    def test_usage_is_logged_for_generate_and_verify(
        self, mock_metadata_store, mock_llm_client
    ):
        lo_id = "course-001:L.O.1.1"
        self._curriculum_ok(mock_metadata_store, lo_id)
        mock_llm_client.model_id = "gemini-test"
        mock_llm_client.generate.side_effect = [
            Ok(
                '{"questions":[{"question":"AI là gì?","choices":["A","B","C","D"],'
                '"correct_index":0,"explanation":"...","difficulty":"easy"}]}'
            ),
            Ok('{"verdicts":[{"question_index":0,"verdict":"supported","reason":"ok"}]}'),
        ]

        use_case = GenerateCurriculumQuizUseCase(
            metadata_store=mock_metadata_store,
            llm_client=mock_llm_client,
            llm_provider="gemini",
        )
        result = use_case.execute(
            GenerateCurriculumQuizRequest(
                course_id="course-001", target_kind="lo", target_code="L.O.1.1", count=1,
            )
        )

        assert result.is_ok()
        use_cases_logged = [
            call.kwargs["use_case"]
            for call in mock_metadata_store.record_llm_usage.call_args_list
        ]
        assert use_cases_logged == ["quiz_generation", "quiz_verification"]
        trace_ids = {
            call.kwargs["trace_id"]
            for call in mock_metadata_store.record_llm_usage.call_args_list
        }
        assert len(trace_ids) == 1  # one correlation id per execute() call
        assert all(
            call.kwargs["provider"] == "gemini"
            for call in mock_metadata_store.record_llm_usage.call_args_list
        )
