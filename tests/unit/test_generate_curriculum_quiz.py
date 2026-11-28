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
from document_chunk.domain.ports.llm_client import LLMUsage
from document_chunk.shared.result import Err, Ok
from worker.use_cases.generate_curriculum_quiz import (
    GenerateCurriculumQuizRequest,
    GenerateCurriculumQuizUseCase,
)
from worker.use_cases.shared.llm_role import LlmRole


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
        resp = result.unwrap()
        assert resp.lo_id == lo_id
        assert resp.items[0].source_chunk_ids == ("chunk-mcp",)
        # The verifier pass (added alongside Reflexion) also calls generate(), so
        # check every call rather than assuming the quiz prompt is the last one.
        assert any(
            "Grounded MCP context" in call.args[0]
            for call in mock_llm_client.generate.call_args_list
        )

    def test_use_case_returns_items_without_writing_quiz_items(
        self,
        mock_metadata_store,
        mock_llm_client,
    ):
        """
        Worker tính, core-api ghi: use case không còn gọi hàm ghi quiz_items
        (trước đây lỗi DB ở bước đó làm hỏng cả lần sinh); câu đi về core-api
        trong kết quả CONTENT_GENERATION_RESULT do runner gửi.
        """
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

        assert result.is_ok()
        assert result.unwrap().quiz_count == 1


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
        [item] = result.unwrap().items
        assert item.question == "AI là gì?"
        assert item.verification.status == "VERIFIED"
        assert item.verification.judge_reason == "ok"
        assert item.verification.regeneration_attempts == 0

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
        [item] = result.unwrap().items
        assert item.question == "What edge-weight constraint does Dijkstra require?"
        # Hồ sơ là của bản sinh lại: judge xác nhận bản mới, kèm lượt bị loại trước đó.
        assert item.verification.status == "VERIFIED"
        assert item.verification.judge_reason == "matches source"
        assert item.verification.regeneration_attempts == 1
        assert item.verification.regenerations_applied == 1
        assert [h.verdict for h in item.verification.history] == ["contradicted", "supported"]

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
        [item] = result.unwrap().items
        assert item.question == "AI là gì?"
        assert item.verification.status == "INCONCLUSIVE"
        assert item.verification.inconclusive_reason == "judge_failed"
        assert item.verification.judge_verdict is None

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

    def test_rejected_question_is_counted_not_returned(
        self, mock_metadata_store, mock_llm_client
    ):
        lo_id = "course-001:L.O.1.1"
        self._curriculum_ok(mock_metadata_store, lo_id)
        mock_llm_client.model_id = "gemini-test"
        verdicts = Ok(
            '{"verdicts":[{"question_index":0,"verdict":"supported","reason":"ok"},'
            '{"question_index":1,"verdict":"contradicted","reason":"still wrong"}]}'
        )
        regen = Ok(
            '{"question":"Q1 sửa","choices":["A","B","C","D"],"correct_index":0,'
            '"explanation":"e","difficulty":"easy"}'
        )
        mock_llm_client.generate.side_effect = [
            Ok(
                '{"questions":[{"question":"Q0","choices":["A","B","C","D"],'
                '"correct_index":0,"explanation":"e0","difficulty":"easy"},'
                '{"question":"Q1","choices":["A","B","C","D"],'
                '"correct_index":0,"explanation":"e1","difficulty":"easy"}]}'
            ),
            verdicts, regen, verdicts, regen, verdicts,
        ]

        result = GenerateCurriculumQuizUseCase(
            metadata_store=mock_metadata_store, llm_client=mock_llm_client,
        ).execute(GenerateCurriculumQuizRequest(
            course_id="course-001", target_kind="lo", target_code="L.O.1.1", count=2,
        ))

        resp = result.unwrap()
        assert [i.question for i in resp.items] == ["Q0"]
        assert [(r.reason, r.regeneration_attempts) for r in resp.rejected] == [("still wrong", 2)]

    def test_judge_failing_after_a_rewrite_leaves_the_rewrite_inconclusive(
        self, mock_metadata_store, mock_llm_client
    ):
        """Bản sinh lại chưa được chấm thì không được mang kết luận của bản cũ."""
        lo_id = "course-001:L.O.1.1"
        self._curriculum_ok(mock_metadata_store, lo_id)
        mock_llm_client.model_id = "gemini-test"
        mock_llm_client.generate.side_effect = [
            Ok(
                '{"questions":[{"question":"Q0","choices":["A","B","C","D"],'
                '"correct_index":0,"explanation":"e0","difficulty":"easy"},'
                '{"question":"Q1","choices":["A","B","C","D"],'
                '"correct_index":0,"explanation":"e1","difficulty":"easy"}]}'
            ),
            Ok(
                '{"verdicts":[{"question_index":0,"verdict":"supported","reason":"ok"},'
                '{"question_index":1,"verdict":"contradicted","reason":"sai"}]}'
            ),
            Ok(
                '{"question":"Q1 sửa","choices":["A","B","C","D"],"correct_index":0,'
                '"explanation":"e","difficulty":"easy"}'
            ),
            Ok("judge hỏng"),
            Ok("vẫn hỏng"),
        ]

        resp = GenerateCurriculumQuizUseCase(
            metadata_store=mock_metadata_store, llm_client=mock_llm_client,
        ).execute(GenerateCurriculumQuizRequest(
            course_id="course-001", target_kind="lo", target_code="L.O.1.1", count=2,
        )).unwrap()

        by_question = {i.question: i.verification for i in resp.items}
        assert by_question["Q0"].status == "VERIFIED"
        assert by_question["Q1 sửa"].status == "INCONCLUSIVE"
        assert by_question["Q1 sửa"].inconclusive_reason == "judge_failed"
        assert by_question["Q1 sửa"].regeneration_attempts == 1

    def test_mcq_with_duplicate_choices_is_dropped_and_counted(
        self, mock_metadata_store, mock_llm_client
    ):
        """
        core-api từ chối cả kết quả nếu một câu sai dạng (normalizeAnswerKey);
        lọc trước để một câu có phương án trùng không làm mất cả mẻ.
        """
        lo_id = "course-001:L.O.1.1"
        self._curriculum_ok(mock_metadata_store, lo_id)
        mock_llm_client.model_id = "gemini-test"
        mock_llm_client.generate.side_effect = [
            Ok(
                '{"questions":[{"question":"Q0","choices":["A","B","C","D"],'
                '"correct_index":0,"explanation":"e0","difficulty":"easy"},'
                '{"question":"Q1","choices":["A","a","C","D"],'
                '"correct_index":0,"explanation":"e1","difficulty":"easy"}]}'
            ),
            Ok(
                '{"verdicts":[{"question_index":0,"verdict":"supported","reason":"ok"},'
                '{"question_index":1,"verdict":"supported","reason":"ok"}]}'
            ),
        ]

        resp = GenerateCurriculumQuizUseCase(
            metadata_store=mock_metadata_store, llm_client=mock_llm_client,
        ).execute(GenerateCurriculumQuizRequest(
            course_id="course-001", target_kind="lo", target_code="L.O.1.1", count=2,
        )).unwrap()

        assert [i.question for i in resp.items] == ["Q0"]
        [rejected] = resp.rejected
        assert "distinct" in rejected.reason

    def test_payload_carries_the_verification_record(
        self, mock_metadata_store, mock_llm_client
    ):
        lo_id = "11111111-1111-4111-8111-111111111111"
        self._curriculum_ok(mock_metadata_store, lo_id)
        mock_llm_client.model_id = "writer-x"
        judge = MagicMock()
        judge.model_id = "judge-y"
        judge.generate_with_usage.return_value = Ok((
            '{"verdicts":[{"question_index":0,"verdict":"supported","reason":"Khớp nguồn"}]}',
            LLMUsage(),
        ))
        mock_llm_client.generate.side_effect = [
            Ok(
                '{"questions":[{"question":"AI là gì?","choices":["A","B","C","D"],'
                '"correct_index":0,"explanation":"...","difficulty":"easy",'
                '"source_chunk_ids":["chunk-001"]}]}'
            ),
        ]

        resp = GenerateCurriculumQuizUseCase(
            metadata_store=mock_metadata_store,
            llm_client=mock_llm_client,
            verifier=LlmRole(client=judge, provider="gemini", profile="judge"),
        ).execute(GenerateCurriculumQuizRequest(
            course_id="course-001", target_kind="lo", target_code="L.O.1.1",
            count=1, bloom_level="apply",
        )).unwrap()

        payload = resp.items[0].to_payload()
        assert payload["verification_status"] == "VERIFIED"
        assert payload["bloom_level"] == 3
        evidence = payload["verification_evidence"]
        assert evidence["criterion"] == "source_grounding"
        assert evidence["judge_verdict"] == "supported"
        assert evidence["judge_reason"] == "Khớp nguồn"
        assert evidence["writer_model"] == "writer-x"
        assert evidence["judge_model"] == "judge-y"
        assert evidence["judge_is_writer"] is False
        # Mức Bloom là mức được yêu cầu, không phải mức judge đánh giá.
        assert evidence["requested_bloom_level"] == 3
        assert "bloom_level" not in evidence
        assert evidence["source_chunk_ids"] == payload["source_chunk_ids"]
