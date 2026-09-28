"""
Regression cho Giai đoạn 0 của quiz-generation-redesign.md.

Mỗi test khoá lại một lỗi cụ thể ở §1.2 của tài liệu, đặt tên theo mã lỗi đó để
đọc doc và đọc test khớp nhau.
"""
from unittest.mock import MagicMock

import pytest

from tests.support import as_llm_client

from document_chunk.application.dto.generation_dto import GenerateCurriculumQuizRequest
from document_chunk.application.services.answer_layout import (
    balanced_positions,
    shuffle_choices,
)
from document_chunk.application.services.structured_output import generate_structured_payload
from document_chunk.domain.bloom import bloom_to_level, bloom_to_name
from document_chunk.domain.ports.llm_client import LLMUsage
from document_chunk.domain.ports.metadata_store import (
    StoredAssessment,
    StoredChapter,
    StoredChunkMetadata,
    StoredCourse,
    StoredLearningOutcome,
)
from document_chunk.shared.result import Ok
from worker.use_cases.generate_curriculum_quiz import GenerateCurriculumQuizUseCase

_LO_ID = "course-001:L.O.1.1"


def _curriculum(store, *, bloom_level):
    """LO với bloom là INT, đúng như Postgres trả về."""
    store.get_curriculum.return_value = Ok((
        StoredCourse(course_id="course-001", code="CO3011", title_vi="Quản lý Dự án Phần mềm"),
        [StoredChapter(chapter_id="ch-1", course_id="course-001", code="1", title="Giới thiệu")],
        [StoredLearningOutcome(
            lo_id=_LO_ID, course_id="course-001", code="L.O.1.1", parent_code="L.O.1",
            statement_vi="Phân tích rủi ro dự án", bloom_level=bloom_level,
        )],
        [StoredAssessment(assessment_id="ao-1", course_id="course-001",
                          code="A.O.1", name_vi="Project nhóm")],
    ))
    store.list_chunks_for_lo.return_value = Ok([
        StoredChunkMetadata(
            chunk_id="11111111-1111-7111-8111-111111111111",
            document_id="22222222-2222-7222-8222-222222222222",
            chunk_index=0, heading_path=("Chương 1",), heading_level=1,
            page_number=1, content_length=120, language="vi",
            content_text="Rủi ro dự án được đánh giá theo xác suất và tác động.",
        )
    ])
    store.persist_curriculum_quiz_items.return_value = Ok(None)
    store.record_llm_usage.return_value = Ok(None)
    return store


def _questions_json(n: int) -> str:
    items = ",".join(
        f'{{"question":"Câu {i}?","choices":["Đúng-{i}","B","C","D"],'
        f'"correct_index":0,"explanation":"vì vậy","difficulty":"easy"}}'
        for i in range(n)
    )
    return f'{{"questions":[{items}]}}'


def _all_supported(n: int) -> str:
    items = ",".join(
        f'{{"question_index":{i},"verdict":"supported","reason":"ok"}}' for i in range(n)
    )
    return f'{{"verdicts":[{items}]}}'


@pytest.fixture
def store():
    return MagicMock()


@pytest.fixture
def llm():
    client = as_llm_client(MagicMock())
    client.model_id = "gemini-test"
    return client


def _run(store, llm, *, count=1, bloom_level=4, request_bloom=None):
    _curriculum(store, bloom_level=bloom_level)
    llm.generate.side_effect = [Ok(_questions_json(count)), Ok(_all_supported(count))]
    use_case = GenerateCurriculumQuizUseCase(
        metadata_store=store, llm_client=llm, llm_provider="gemini",
    )
    return use_case.execute(GenerateCurriculumQuizRequest(
        course_id="course-001", target_kind="lo", target_code="L.O.1.1",
        count=count, bloom_level=request_bloom,
    ))


# ---------------------------------------------------------------------------
# D4 — mọi câu đều lưu bloom_level = 2
# ---------------------------------------------------------------------------

class TestBloomLevel:
    def test_int_bloom_from_postgres_survives(self, store, llm):
        """LO mức 4 phải lưu 4. Trước đây str(4) không khớp bảng tên → 2."""
        assert _run(store, llm, bloom_level=4).is_ok()
        assert store.persist_curriculum_quiz_items.call_args.kwargs["bloom_level"] == 4

    def test_request_bloom_overrides_lo(self, store, llm):
        assert _run(store, llm, bloom_level=4, request_bloom="remember").is_ok()
        assert store.persist_curriculum_quiz_items.call_args.kwargs["bloom_level"] == 1

    def test_lo_without_bloom_falls_back_to_understand(self, store, llm):
        assert _run(store, llm, bloom_level=None).is_ok()
        assert store.persist_curriculum_quiz_items.call_args.kwargs["bloom_level"] == 2

    @pytest.mark.parametrize("value,level", [
        (3, 3), ("3", 3), ("analyze", 4), ("Phân tích", 4), (None, None), ("", None), (9, None),
    ])
    def test_conversion_accepts_every_shape_seen_in_the_system(self, value, level):
        assert bloom_to_level(value) == level

    def test_name_and_level_agree(self):
        assert bloom_to_name(4) == "analyze"
        assert bloom_to_level(bloom_to_name(4)) == 4


# ---------------------------------------------------------------------------
# O2 — llm_usage_logs CHECK từ chối mọi dòng
# ---------------------------------------------------------------------------

class TestUsageLogging:
    def test_status_is_uppercase(self, store, llm):
        """CHECK chỉ nhận 'OK'|'RATE_LIMITED'|'ERROR', phân biệt hoa thường."""
        assert _run(store, llm).is_ok()
        statuses = {
            c.kwargs["status"] for c in store.record_llm_usage.call_args_list
        }
        assert statuses <= {"OK", "RATE_LIMITED", "ERROR"}
        assert "OK" in statuses

    def test_tokens_reach_the_usage_row(self, store, llm):
        """ai_runtime/Gemini trả token; trước đây chúng bị log rồi bỏ đi."""
        _curriculum(store, bloom_level=4)
        llm.generate_with_usage.side_effect = [
            Ok((_questions_json(1), LLMUsage(prompt_tokens=120, completion_tokens=45))),
            Ok((_all_supported(1), LLMUsage(prompt_tokens=80, completion_tokens=12))),
        ]
        use_case = GenerateCurriculumQuizUseCase(
            metadata_store=store, llm_client=llm, llm_provider="gemini",
        )
        assert use_case.execute(GenerateCurriculumQuizRequest(
            course_id="course-001", target_kind="lo", target_code="L.O.1.1", count=1,
        )).is_ok()

        by_use_case = {c.kwargs["use_case"]: c.kwargs for c in store.record_llm_usage.call_args_list}
        assert by_use_case["quiz_generation"]["prompt_tokens"] == 120
        assert by_use_case["quiz_generation"]["completion_tokens"] == 45
        assert by_use_case["quiz_verification"]["prompt_tokens"] == 80

    def test_repair_call_is_also_billed(self):
        """Lần sửa JSON cũng tốn tiền nên không được nằm ngoài sổ chi phí."""
        from pydantic import BaseModel

        class _Payload(BaseModel):
            name: str

        llm = as_llm_client(MagicMock())
        llm.generate_with_usage.side_effect = [
            Ok(("not json", LLMUsage(prompt_tokens=10, completion_tokens=5))),
            Ok(('{"name":"ok"}', LLMUsage(prompt_tokens=20, completion_tokens=7))),
        ]
        seen: list[LLMUsage] = []
        result = generate_structured_payload(
            llm_client=llm, prompt="p", system=None,
            schema_model=_Payload, on_usage=seen.append,
        )
        assert result.is_ok()
        assert sum(u.prompt_tokens for u in seen) == 30
        assert sum(u.completion_tokens for u in seen) == 12


# ---------------------------------------------------------------------------
# G3 — không xáo đáp án
# ---------------------------------------------------------------------------

class TestAnswerShuffling:
    def test_correct_answer_moves_and_stays_correct(self, store, llm):
        """LLM luôn đặt đáp án đúng ở vị trí 0; sau khi xáo phải khác đi."""
        assert _run(store, llm, count=4).is_ok()
        items = store.persist_curriculum_quiz_items.call_args.kwargs["quiz_items"]
        for item in items:
            assert item.choices[item.correct_index].startswith("Đúng-")
        assert {item.correct_index for item in items} != {0}

    def test_positions_are_balanced_across_a_batch(self):
        positions = balanced_positions(8, options=4, seed="lo-x")
        assert sorted(positions) == [0, 0, 1, 1, 2, 2, 3, 3]

    def test_shuffle_is_deterministic(self):
        a = shuffle_choices(["dung", "b", "c", "d"], 0, seed="q", target_index=2)
        b = shuffle_choices(["dung", "b", "c", "d"], 0, seed="q", target_index=2)
        assert a == b
        assert a[0][a[1]] == "dung"

    def test_broken_correct_index_is_left_alone(self):
        """Dữ liệu hỏng phải đi tiếp để validator bắt, không im lặng thành câu khác."""
        assert shuffle_choices(["a", "b"], 9, seed="q", target_index=0) == (["a", "b"], 9)


# ---------------------------------------------------------------------------
# O3 — quiz_id ngẫu nhiên nên retry nhân đôi bộ câu
# ---------------------------------------------------------------------------

class TestDeterministicQuizId:
    def test_same_content_gives_same_id_across_runs(self, llm):
        ids = []
        for _ in range(2):
            store = MagicMock()
            client = as_llm_client(MagicMock())
            client.model_id = "gemini-test"
            assert _run(store, client, count=2).is_ok()
            items = store.persist_curriculum_quiz_items.call_args.kwargs["quiz_items"]
            ids.append([i.question_id for i in items])
        assert ids[0] == ids[1]

    def test_different_questions_give_different_ids(self, store, llm):
        assert _run(store, llm, count=3).is_ok()
        items = store.persist_curriculum_quiz_items.call_args.kwargs["quiz_items"]
        assert len({i.question_id for i in items}) == 3
