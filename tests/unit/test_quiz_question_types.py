"""Sinh quiz theo từng kiểu câu: prompt, schema đầu ra và dạng đáp án lưu xuống DB."""
import json

import pytest

from document_chunk.domain.ports.metadata_store import (
    StoredChunkMetadata,
    StoredCourse,
    StoredLearningOutcome,
)
from document_chunk.shared.result import Ok
from worker.prompts import curriculum_quiz as prompts
from worker.use_cases.generate_curriculum_quiz import (
    GenerateCurriculumQuizRequest,
    GenerateCurriculumQuizUseCase,
)

LO_ID = "course-001:L.O.1.1"
_TAIL = {"explanation": "Theo slide.", "difficulty": "easy", "lo_alignment_rationale": "r",
         "source_chunk_ids": ["chunk-001"]}

# Kiểu câu → (một câu model trả về, kỳ vọng options/correct_answer lưu xuống).
CASES = {
    "MCQ_MULTI": (
        {"question": "Chọn tất cả tính chất của cây.", "choices": ["Liên thông", "Có chu trình",
         "Không chu trình", "Có đúng 2 đỉnh"], "correct_indices": [0, 2]},
        lambda item: (
            sorted(item.choices) == sorted(["Liên thông", "Có chu trình", "Không chu trình", "Có đúng 2 đỉnh"])
            and sorted(item.correct_answer) == ["Không chu trình", "Liên thông"]
            # Đáp án đúng theo thứ tự đã xáo của phương án.
            and item.correct_answer == [c for c in item.choices if c in item.correct_answer]
        ),
    ),
    "TRUE_FALSE": (
        {"question": "Cây n đỉnh có n cạnh.", "is_true": False},
        lambda item: item.choices == ("Đúng", "Sai") and item.correct_answer == "Sai",
    ),
    "FILL_BLANK": (
        {"question": "Cây n đỉnh có ____ cạnh.", "accepted_answers": ["n-1", " n − 1 "]},
        lambda item: item.choices == () and item.correct_answer == ["n-1", "n − 1"],
    ),
    "SHORT_ANSWER": (
        {"question": "Vì sao cây n đỉnh có n-1 cạnh?", "model_answer": "Vì thêm mỗi đỉnh cần đúng một cạnh.",
         "rubric": ["Nêu được quy nạp theo số đỉnh", "Nêu được mỗi đỉnh mới thêm một cạnh"]},
        lambda item: item.choices == () and item.correct_answer == {
            "model_answer": "Vì thêm mỗi đỉnh cần đúng một cạnh.",
            "rubric": ["Nêu được quy nạp theo số đỉnh", "Nêu được mỗi đỉnh mới thêm một cạnh"],
        },
    ),
}


def _store(mock_metadata_store):
    mock_metadata_store.get_curriculum.return_value = Ok((
        StoredCourse(course_id="course-001", code="CS101", title_vi="Toán rời rạc"),
        [],
        [StoredLearningOutcome(lo_id=LO_ID, course_id="course-001", code="L.O.1.1", parent_code=None,
                               statement_vi="Hiểu cây", bloom_level="understand")],
        [],
    ))
    mock_metadata_store.list_chunks_for_lo.return_value = Ok([
        StoredChunkMetadata(chunk_id="chunk-001", document_id="doc-001", chunk_index=0,
                            content_text="Cây là đồ thị liên thông không chu trình, n đỉnh có n-1 cạnh.")
    ])
    mock_metadata_store.persist_curriculum_quiz_items.return_value = Ok(None)
    return mock_metadata_store


def _run(store, llm, question_type, item):
    llm.generate.side_effect = [
        Ok(json.dumps({"questions": [{**item, **_TAIL}]}, ensure_ascii=False)),
        Ok('{"verdicts":[{"question_index":0,"verdict":"supported","reason":""}]}'),
    ]
    return GenerateCurriculumQuizUseCase(metadata_store=store, llm_client=llm).execute(
        GenerateCurriculumQuizRequest(course_id="course-001", target_kind="lo", target_code="L.O.1.1",
                                      count=1, question_type=question_type)
    )


@pytest.mark.parametrize("question_type", sorted(CASES))
def test_each_type_is_generated_verified_and_stored(question_type, mock_metadata_store, mock_llm_client):
    store = _store(mock_metadata_store)
    item, expected = CASES[question_type]

    result = _run(store, mock_llm_client, question_type, item)

    assert result.is_ok(), result
    [stored] = store.persist_curriculum_quiz_items.call_args.kwargs["quiz_items"]
    assert stored.question_type == question_type
    assert expected(stored), stored
    generation_prompt, verification_prompt = (c.args[0] for c in mock_llm_client.generate.call_args_list)
    assert prompts._KINDS[question_type].rules in generation_prompt
    assert prompts._KINDS[question_type].check_subject in verification_prompt


def test_quiz_id_includes_type_so_types_do_not_overwrite_each_other(mock_metadata_store, mock_llm_client):
    store = _store(mock_metadata_store)
    ids = []
    for question_type, item in (("TRUE_FALSE", {"question": "Q", "is_true": True}),
                                ("SHORT_ANSWER", {"question": "Q", "model_answer": "M", "rubric": ["R"]})):
        assert _run(store, mock_llm_client, question_type, item).is_ok()
        ids.append(store.persist_curriculum_quiz_items.call_args.kwargs["quiz_items"][0].question_id)
    assert ids[0] != ids[1]


def test_fill_blank_without_a_blank_is_rejected_by_the_schema(mock_metadata_store, mock_llm_client):
    store = _store(mock_metadata_store)
    bad = Ok(json.dumps({"questions": [{"question": "Không có chỗ trống", "accepted_answers": ["x"], **_TAIL}]}))
    # Lần sinh, rồi lần sửa JSON cũng trả lại đúng nội dung hỏng.
    mock_llm_client.generate.side_effect = [bad, bad]

    result = GenerateCurriculumQuizUseCase(metadata_store=store, llm_client=mock_llm_client).execute(
        GenerateCurriculumQuizRequest(course_id="course-001", target_kind="lo", target_code="L.O.1.1",
                                      count=1, question_type="FILL_BLANK")
    )

    assert result.is_err()
    store.persist_curriculum_quiz_items.assert_not_called()


def test_unknown_type_fails_before_calling_the_model(mock_metadata_store, mock_llm_client):
    result = GenerateCurriculumQuizUseCase(metadata_store=_store(mock_metadata_store),
                                           llm_client=mock_llm_client).execute(
        GenerateCurriculumQuizRequest(course_id="course-001", target_kind="lo", target_code="L.O.1.1",
                                      question_type="ESSAY")
    )
    assert result.is_err()
    mock_llm_client.generate.assert_not_called()


def test_multi_answer_must_leave_a_wrong_choice(mock_metadata_store, mock_llm_client):
    store = _store(mock_metadata_store)
    all_correct = Ok(json.dumps({"questions": [{"question": "Q", "choices": ["a", "b", "c", "d"],
                                                "correct_indices": [0, 1, 2, 3], **_TAIL}]}))
    mock_llm_client.generate.side_effect = [all_correct, all_correct]

    result = GenerateCurriculumQuizUseCase(metadata_store=store, llm_client=mock_llm_client).execute(
        GenerateCurriculumQuizRequest(course_id="course-001", target_kind="lo", target_code="L.O.1.1",
                                      count=1, question_type="MCQ_MULTI")
    )

    assert result.is_err()
