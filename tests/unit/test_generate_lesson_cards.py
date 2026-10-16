"""
Tests for worker/use_cases/generate_lesson_cards.py.
"""
import json

import pytest
from pydantic import ValidationError

from document_chunk.domain.ports.metadata_store import (
    StoredChunkMetadata,
    StoredCourse,
    StoredLearningOutcome,
)
from document_chunk.shared.result import Err, Ok
from worker.use_cases.generate_lesson_cards import (
    GenerateLessonCardsRequest,
    GenerateLessonCardsUseCase,
    _LessonCard,
)

LO_ID = "11111111-1111-4111-8111-111111111111"


def _card(title: str, **overrides) -> dict:
    card = {
        "title": title,
        "key_insight": f"{title} là ý chính.",
        "bullets": ["Ý một", "Ý hai"],
        "example": "Ví dụ trong slide.",
        "misconception": None,
        "check_question": f"{title} là gì?",
        "check_answer": f"{title} là ý chính.",
        "source_chunk_ids": ["chunk-001"],
    }
    card.update(overrides)
    return card


def _cards_json(*cards: dict) -> str:
    return json.dumps({"cards": list(cards)}, ensure_ascii=False)


def _verdicts(*verdicts: str) -> str:
    return json.dumps(
        {
            "verdicts": [
                {"card_index": i, "verdict": v, "reason": "r"} for i, v in enumerate(verdicts)
            ]
        }
    )


def _request(count: int = 4) -> GenerateLessonCardsRequest:
    return GenerateLessonCardsRequest(
        course_id="course-001", target_kind="lo", target_code=LO_ID, count=count
    )


@pytest.fixture
def store(mock_metadata_store):
    mock_metadata_store.get_curriculum.return_value = Ok((
        StoredCourse(course_id="course-001", code="CS101", title_vi="Nhập môn AI"),
        [],
        [StoredLearningOutcome(
            lo_id=LO_ID,
            course_id="course-001",
            code="L.O.1.1",
            parent_code=None,
            statement_vi="Hiểu khái niệm AI",
            bloom_level=2,
        )],
        [],
    ))
    mock_metadata_store.list_chunks_for_lo.return_value = Ok([
        StoredChunkMetadata(
            chunk_id="chunk-001",
            document_id="doc-001",
            chunk_index=0,
            content_text="AI là một lĩnh vực của khoa học máy tính.",
        ),
        StoredChunkMetadata(
            chunk_id="chunk-002",
            document_id="doc-001",
            chunk_index=1,
            content_text="Học máy là một nhánh của AI.",
        ),
    ])
    return mock_metadata_store


def _use_case(store, llm) -> GenerateLessonCardsUseCase:
    return GenerateLessonCardsUseCase(metadata_store=store, llm_client=llm)


def _persisted(store):
    return store.persist_curriculum_lesson_cards.call_args.kwargs["lesson_cards"]


class TestGenerateLessonCards:
    def test_persists_verified_cards_in_order_with_all_fields(self, store, mock_llm_client):
        mock_llm_client.generate.side_effect = [
            Ok(_cards_json(
                _card("AI", misconception="null"),
                _card("Học máy", source_chunk_ids=["chunk-002", "chunk-bịa"]),
            )),
            Ok(_verdicts("supported", "supported")),
        ]

        result = _use_case(store, mock_llm_client).execute(_request(count=2))

        assert result.is_ok()
        assert result.unwrap().card_count == 2
        assert store.persist_curriculum_lesson_cards.call_args.kwargs["lo_id"] == LO_ID
        first, second = _persisted(store)
        assert [first.card_index, second.card_index] == [0, 1]
        assert first.title == "AI"
        assert first.check_question == "AI là gì?"
        assert first.check_answer == "AI là ý chính."
        assert first.example == "Ví dụ trong slide."
        # Model trả chuỗi "null" theo prompt "... or null": phải thành None thật.
        assert first.misconception is None
        # chunk_id model bịa ra bị bỏ, chỉ giữ chunk thật sự đã đưa vào prompt.
        assert second.source_chunk_ids == ("chunk-002",)
        assert second.primary_chunk_id == "chunk-002"

    def test_card_id_is_stable_for_the_same_lo_and_title(self, store, mock_llm_client):
        mock_llm_client.generate.side_effect = [
            Ok(_cards_json(_card("AI"))),
            Ok(_verdicts("supported")),
            Ok(_cards_json(_card("  ai  "))),
            Ok(_verdicts("supported")),
        ]
        use_case = _use_case(store, mock_llm_client)

        first = use_case.execute(_request(count=1)).unwrap().card_ids
        second = use_case.execute(_request(count=1)).unwrap().card_ids

        assert first == second

    def test_repeated_titles_are_dropped_before_verification(self, store, mock_llm_client):
        mock_llm_client.generate.side_effect = [
            Ok(_cards_json(_card("AI"), _card("ai"), _card("Học máy"))),
            Ok(_verdicts("supported", "supported")),
        ]

        result = _use_case(store, mock_llm_client).execute(_request(count=3))

        assert result.unwrap().card_count == 2
        assert [card.title for card in _persisted(store)] == ["AI", "Học máy"]

    def test_only_count_cards_are_kept(self, store, mock_llm_client):
        mock_llm_client.generate.side_effect = [
            Ok(_cards_json(_card("A"), _card("B"), _card("C"))),
            Ok(_verdicts("supported", "supported")),
        ]

        result = _use_case(store, mock_llm_client).execute(_request(count=2))

        assert result.unwrap().card_count == 2

    def test_failing_card_is_rewritten_then_saved(self, store, mock_llm_client):
        mock_llm_client.generate.side_effect = [
            Ok(_cards_json(_card("AI"), _card("Sai"))),
            Ok(_verdicts("supported", "contradicted")),
            Ok(json.dumps(_card("Học máy"), ensure_ascii=False)),
            Ok(_verdicts("supported", "supported")),
        ]

        result = _use_case(store, mock_llm_client).execute(_request(count=2))

        assert result.is_ok()
        assert mock_llm_client.generate.call_count == 4
        assert [card.title for card in _persisted(store)] == ["AI", "Học máy"]

    def test_card_still_failing_after_retries_is_not_saved(self, store, mock_llm_client):
        mock_llm_client.generate.side_effect = [
            Ok(_cards_json(_card("AI"), _card("Sai"))),
            Ok(_verdicts("supported", "not_enough_evidence")),
            Ok(json.dumps(_card("Sai nữa"), ensure_ascii=False)),
            Ok(_verdicts("supported", "not_enough_evidence")),
            Ok(json.dumps(_card("Vẫn sai"), ensure_ascii=False)),
            Ok(_verdicts("supported", "contradicted")),
        ]

        result = _use_case(store, mock_llm_client).execute(_request(count=2))

        assert result.is_ok()
        assert [card.title for card in _persisted(store)] == ["AI"]

    def test_nothing_saved_when_every_card_fails(self, store, mock_llm_client):
        use_case = GenerateLessonCardsUseCase(
            metadata_store=store, llm_client=mock_llm_client, max_verification_retries=0
        )
        mock_llm_client.generate.side_effect = [
            Ok(_cards_json(_card("AI"))),
            Ok(_verdicts("contradicted")),
        ]

        result = use_case.execute(_request(count=1))

        assert result.is_err()
        store.persist_curriculum_lesson_cards.assert_not_called()

    def test_broken_verifier_does_not_block_saving(self, store, mock_llm_client):
        """Chấm hỏng là không kết luận được; người duyệt vẫn là cổng cuối."""
        mock_llm_client.generate.side_effect = [
            Ok(_cards_json(_card("AI"))),
            Ok("không phải JSON"),
            Ok("vẫn không phải JSON"),
        ]

        result = _use_case(store, mock_llm_client).execute(_request(count=1))

        assert result.is_ok()
        assert [card.title for card in _persisted(store)] == ["AI"]

    def test_no_source_chunks_fails_without_calling_the_model(self, store, mock_llm_client):
        store.list_chunks_for_lo.return_value = Ok([])

        result = _use_case(store, mock_llm_client).execute(_request())

        assert result.is_err()
        assert "No grounded source chunks" in str(result.error)
        mock_llm_client.generate.assert_not_called()

    def test_only_selected_documents_are_used(self, store, mock_llm_client):
        request = GenerateLessonCardsRequest(
            course_id="course-001",
            target_kind="lo",
            target_code=LO_ID,
            count=1,
            source_document_ids=["doc-khác"],
        )

        result = _use_case(store, mock_llm_client).execute(request)

        assert result.is_err()
        mock_llm_client.generate.assert_not_called()

    def test_count_out_of_range_is_rejected(self, store, mock_llm_client):
        result = _use_case(store, mock_llm_client).execute(_request(count=7))

        assert result.is_err()
        mock_llm_client.generate.assert_not_called()

    def test_persist_error_is_returned(self, store, mock_llm_client):
        store.persist_curriculum_lesson_cards.return_value = Err(RuntimeError("db down"))
        mock_llm_client.generate.side_effect = [
            Ok(_cards_json(_card("AI"))),
            Ok(_verdicts("supported")),
        ]

        result = _use_case(store, mock_llm_client).execute(_request(count=1))

        assert result.is_err()
        assert "db down" in str(result.error)

    def test_usage_is_logged_per_step(self, store, mock_llm_client):
        mock_llm_client.generate.side_effect = [
            Ok(_cards_json(_card("AI"))),
            Ok(_verdicts("contradicted")),
            Ok(json.dumps(_card("AI đúng"), ensure_ascii=False)),
            Ok(_verdicts("supported")),
        ]

        _use_case(store, mock_llm_client).execute(_request(count=1))

        use_cases = [c.kwargs["use_case"] for c in store.record_llm_usage.call_args_list]
        assert use_cases == [
            "card_generation",
            "card_verification",
            "card_regeneration",
            "card_verification",
        ]
        assert all(c.kwargs["status"] == "OK" for c in store.record_llm_usage.call_args_list)


class TestLessonCardModel:
    def test_blank_optional_fields_become_none(self):
        card = _LessonCard.model_validate(_card("AI", example="  ", misconception="N/A"))

        assert card.example is None
        assert card.misconception is None

    def test_too_many_bullets_is_rejected(self):
        with pytest.raises(ValidationError):
            _LessonCard.model_validate(_card("AI", bullets=["a", "b", "c", "d", "e"]))

    def test_paragraph_instead_of_insight_is_rejected(self):
        with pytest.raises(ValidationError):
            _LessonCard.model_validate(_card("AI", key_insight="x" * 401))

    def test_missing_check_pair_is_rejected(self):
        card = _card("AI")
        del card["check_answer"]
        with pytest.raises(ValidationError):
            _LessonCard.model_validate(card)
