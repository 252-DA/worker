"""Trích quan hệ: trích dẫn phải nằm nguyên văn trong chunk, lô lỗi không làm mất lô đã xong."""
import json
from unittest.mock import MagicMock

import pytest

from document_chunk.domain.ports.course_knowledge_store import KnowledgeChunk
from document_chunk.shared.result import Err, Ok
from worker.llm_role import LlmRole
from worker.prompts import relations as prompts
from worker.use_cases.extract_relations import (
    ExtractRelationsRequest,
    ExtractRelationsUseCase,
    _RawPassage,
    _RawRelation,
    clean_relations,
)

TEXT_A = (
    "Deadlock xảy ra khi bốn điều kiện cùng đúng. Điều kiện giữ và chờ (hold and wait) "
    "là một trong bốn điều kiện cần của deadlock, cùng với loại trừ tương hỗ."
)
TEXT_B = (
    "Starvation khác deadlock: trong starvation một tiến trình bị trì hoãn vô hạn "
    "dù hệ thống vẫn tiến triển, còn deadlock thì không ai tiến triển được."
)


def _chunk(i: int, text: str, page: int = 3) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=f"c{i}", document_id="d1", sort_order=i,
        heading_path=("Chương 5", "Deadlock"), page_number=page, content=text,
    )


def _rel(source, relation, target, quote) -> _RawRelation:
    return _RawRelation(source=source, relation=relation, target=target, evidence_quote=quote)


# ---------------------------------------------------------------- clean_relations

def test_keeps_a_relation_whose_quote_is_in_the_chunk():
    raw = _RawPassage(index=0, relations=[
        _rel("Hold and wait", "part_of", "Điều kiện deadlock",
             "giữ và chờ (hold and wait) là một trong bốn điều kiện cần của deadlock"),
    ])
    kept, dropped = clean_relations(raw, _chunk(1, TEXT_A))
    assert dropped == {}
    assert len(kept) == 1
    assert kept[0].source_key == "hold and wait"
    assert kept[0].target_key == "dieu kien deadlock"
    assert kept[0].relation_type == "part_of"
    assert kept[0].page_number == 3
    assert kept[0].chunk_id == "c1"


def test_drops_each_kind_of_bad_relation_with_its_reason():
    quote = "bốn điều kiện cùng đúng"
    raw = _RawPassage(index=0, relations=[
        _rel("A", "causes", "B", quote),                       # loại lạ
        _rel("", "cause", "B", quote),                         # nhãn rỗng
        _rel("Deadlock", "cause", "deadlock!", quote),         # tự nối
        _rel("A", "cause", "B", "một câu mô hình bịa ra hoàn toàn"),  # trích dẫn không có trong chunk
        _rel("Điều kiện", "cause", "Deadlock", quote),         # hợp lệ
        _rel("điều kiện", "cause", "DEADLOCK", quote),         # trùng sau chuẩn hóa
    ])
    kept, dropped = clean_relations(raw, _chunk(1, TEXT_A))
    assert [r.source_label for r in kept] == ["Điều kiện"]
    assert dropped == {
        "unknown_type": 1, "empty_label": 1, "self_loop": 1,
        "quote_not_in_chunk": 1, "duplicate": 1,
    }


def test_quote_from_another_chunk_is_rejected():
    raw = _RawPassage(index=0, relations=[
        _rel("Starvation", "contrast", "Deadlock", "một tiến trình bị trì hoãn vô hạn"),
    ])
    kept, dropped = clean_relations(raw, _chunk(1, TEXT_A))  # trích dẫn thuộc TEXT_B
    assert kept == []
    assert dropped == {"quote_not_in_chunk": 1}


def test_caps_relations_per_chunk():
    quote = "bốn điều kiện cùng đúng"
    raw = _RawPassage(index=0, relations=[_rel(f"Ý {i}", "cause", "Deadlock", quote) for i in range(9)])
    kept, dropped = clean_relations(raw, _chunk(1, TEXT_A))
    assert len(kept) == 6
    assert dropped == {"over_limit": 3}


def test_missing_passage_means_no_relations():
    assert clean_relations(None, _chunk(1, TEXT_A)) == ([], {})


# ---------------------------------------------------------------- use case

@pytest.fixture
def knowledge():
    store = MagicMock()
    store.list_document_chunks.return_value = Ok([_chunk(1, TEXT_A), _chunk(2, TEXT_B), _chunk(3, "ngắn quá")])
    store.replace_relations.side_effect = lambda **kw: Ok(len(kw["relations"]))
    return store


@pytest.fixture
def metadata():
    store = MagicMock()
    store.record_llm_usage.return_value = Ok(None)
    return store


def _llm(mock_llm_client, *replies):
    mock_llm_client.generate.side_effect = [Ok(r if isinstance(r, str) else json.dumps(r)) for r in replies]
    return LlmRole(client=mock_llm_client, provider="test", profile="deepseek-chat")


GOOD_REPLY = {
    "passages": [
        {"index": 0, "relations": [
            {"source": "Hold and wait", "relation": "part_of", "target": "Điều kiện deadlock",
             "evidence_quote": "giữ và chờ (hold and wait) là một trong bốn điều kiện cần của deadlock"},
        ]},
        {"index": 1, "relations": [
            {"source": "Starvation", "relation": "contrast", "target": "Deadlock",
             "evidence_quote": "Starvation khác deadlock"},
            {"source": "Starvation", "relation": "cause", "target": "Deadlock",
             "evidence_quote": "câu này không có trong tài liệu"},
        ]},
    ]
}


def test_extracts_validates_and_saves_per_batch(knowledge, metadata, mock_llm_client):
    use_case = ExtractRelationsUseCase(knowledge, metadata, _llm(mock_llm_client, GOOD_REPLY))
    response = use_case.execute(ExtractRelationsRequest(course_id="CO3137", document_ids=["d1"])).unwrap()

    assert response.chunks_seen == 3
    assert response.chunks_processed == 2          # chunk "ngắn quá" bị lọc
    assert response.relations_written == 2
    assert response.relations_dropped == 1
    assert response.dropped_reasons == {"quote_not_in_chunk": 1}

    call = knowledge.replace_relations.call_args.kwargs
    assert call["course_id"] == "CO3137"
    assert call["chunk_ids"] == ["c1", "c2"]
    assert call["extraction_version"] == prompts.EXTRACTION_VERSION
    assert call["model"] == "gemini-test"
    assert {r.relation_type for r in call["relations"]} == {"part_of", "contrast"}


def test_usage_is_logged_under_relation_extraction(knowledge, metadata, mock_llm_client):
    use_case = ExtractRelationsUseCase(knowledge, metadata, _llm(mock_llm_client, GOOD_REPLY))
    use_case.execute(ExtractRelationsRequest(course_id="CO3137", document_ids=["d1"]))
    kwargs = metadata.record_llm_usage.call_args.kwargs
    assert kwargs["use_case"] == "relation_extraction"
    assert kwargs["status"] == "OK"
    assert kwargs["provider"] == "test"


def test_a_failed_batch_does_not_lose_finished_batches(knowledge, metadata, mock_llm_client):
    chunks = [_chunk(i, TEXT_A) for i in range(1, 5)]            # 4 chunk, 3 mỗi lô → 2 lô
    knowledge.list_document_chunks.return_value = Ok(chunks)
    ok_reply = {"passages": [{"index": 0, "relations": [
        {"source": "Điều kiện", "relation": "cause", "target": "Deadlock", "evidence_quote": "bốn điều kiện cùng đúng"}]}]}
    # Lô 1 thành công; lô 2 trả rác, cả lần sửa JSON cũng hỏng.
    role = _llm(mock_llm_client, ok_reply, "không phải json", "vẫn không phải json")
    response = ExtractRelationsUseCase(knowledge, metadata, role).execute(
        ExtractRelationsRequest(course_id="CO3137", document_ids=["d1"])
    ).unwrap()
    assert response.failed_batches == 1
    assert response.chunks_processed == 3
    assert knowledge.replace_relations.call_count == 1


def test_all_batches_failing_is_an_error(knowledge, metadata, mock_llm_client):
    role = _llm(mock_llm_client, "rác", "rác")
    result = ExtractRelationsUseCase(knowledge, metadata, role).execute(
        ExtractRelationsRequest(course_id="CO3137", document_ids=["d1"])
    )
    assert result.is_err()
    knowledge.replace_relations.assert_not_called()


def test_save_failure_counts_as_a_failed_batch(knowledge, metadata, mock_llm_client):
    knowledge.replace_relations.side_effect = lambda **kw: Err(RuntimeError("db down"))
    result = ExtractRelationsUseCase(knowledge, metadata, _llm(mock_llm_client, GOOD_REPLY)).execute(
        ExtractRelationsRequest(course_id="CO3137", document_ids=["d1"])
    )
    assert result.is_err()


def test_requires_documents_and_text(knowledge, metadata, mock_llm_client):
    use_case = ExtractRelationsUseCase(knowledge, metadata, _llm(mock_llm_client))
    assert use_case.execute(ExtractRelationsRequest(course_id="c", document_ids=[])).is_err()
    knowledge.list_document_chunks.return_value = Ok([_chunk(1, "ngắn")])
    assert use_case.execute(ExtractRelationsRequest(course_id="c", document_ids=["d1"])).is_err()


def test_stops_at_the_chunk_budget(knowledge, metadata, mock_llm_client):
    knowledge.list_document_chunks.return_value = Ok([_chunk(i, TEXT_A) for i in range(1, 11)])
    role = _llm(mock_llm_client, {"passages": []}, {"passages": []})
    response = ExtractRelationsUseCase(knowledge, metadata, role, max_chunks=4).execute(
        ExtractRelationsRequest(course_id="c", document_ids=["d1"])
    ).unwrap()
    assert response.chunks_seen == 10
    assert response.chunks_processed == 4
