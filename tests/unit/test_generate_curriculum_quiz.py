"""
Tests for worker/use_cases/generate_curriculum_quiz.py.
"""
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
        mock_metadata_store.persist_enrichment_batch.return_value = Ok(None)
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
        mock_metadata_store.list_chunks_for_lo.return_value = Ok([])
        mock_metadata_store.persist_enrichment_batch.return_value = Ok(None)
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
