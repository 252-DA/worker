import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from bullmq.custom_errors import UnrecoverableError

from document_chunk.domain.ports.metadata_store import (
    StoredChunkMetadata,
    StoredCourse,
    StoredLearningOutcome,
)
from document_chunk.shared.result import Err, Ok
from tests.support import RecordingResultPublisher, assert_store_made_no_writes
from worker.runners.content_generation_worker import (
    _card_request,
    _parse_job,
    _process_job,
    _quiz_request,
    _relations_request,
)
from worker.use_cases.extract_relations import ExtractRelationsResponse
from worker.use_cases.generate_curriculum_quiz import GenerateCurriculumQuizResponse
from worker.use_cases.generate_lesson_cards import (
    DEFAULT_CARD_COUNT,
    GenerateLessonCardsResponse,
)
from worker.use_cases.generated_content import (
    GeneratedLessonCard,
    GeneratedQuizItem,
    RejectedDraft,
)
from worker.use_cases.verification import Verification


def _job_data():
    return {
        "event_id": "event-1",
        "event_type": "CONTENT_GENERATION_REQUESTED",
        "payload": {
            "request_id": "request-1",
            "course_id": "course-1",
            "type": "quiz",
            "scope": {
                "target_kind": "lo",
                "target_code": "L.O.3.1",
                "count": 3,
                "bloom": "analyze",
            },
        },
    }


def test_parse_job_and_scope():
    job = _parse_job(_job_data())
    request = _quiz_request(job)

    assert job.request_id == "request-1"
    assert request.target_kind == "lo"
    assert request.target_code == "L.O.3.1"
    assert request.count == 3
    assert request.bloom_level == "analyze"
    # Request cũ không có question_type: vẫn là trắc nghiệm một đáp án.
    assert request.question_type == "MCQ_SINGLE"


def test_quiz_request_reads_question_type():
    data = _job_data()
    data["payload"]["scope"]["question_type"] = "FILL_BLANK"

    assert _quiz_request(_parse_job(data)).question_type == "FILL_BLANK"


def _verification(status="VERIFIED"):
    return Verification(
        status=status,
        judge_verdict="supported" if status == "VERIFIED" else None,
        judge_reason="ok" if status == "VERIFIED" else None,
        inconclusive_reason=None if status == "VERIFIED" else "judge_failed",
        regeneration_attempts=0,
        regenerations_applied=0,
    )


def _quiz_item(question_id, status="VERIFIED"):
    return GeneratedQuizItem(
        question_id=question_id,
        question_type="MCQ_SINGLE",
        question=f"Câu {question_id}?",
        choices=("A", "B", "C", "D"),
        correct_index=1,
        correct_answer="B",
        explanation="Theo slide.",
        bloom_level=3,
        source_chunk_ids=("chunk-1",),
        model_id="writer-model",
        judge_model="judge-model",
        verification=_verification(status),
    )


def _quiz_response(*ids, rejected=()):
    return GenerateCurriculumQuizResponse(
        course_id="course-1",
        lo_id="lo-1",
        bloom_level=3,
        items=tuple(_quiz_item(i) for i in ids),
        rejected=tuple(rejected),
    )


def _container(publisher=None):
    container = MagicMock()
    container.result_publisher = publisher or RecordingResultPublisher()
    return container


def _statuses(container):
    return [e["payload"]["status"] for e in container.result_publisher.envelopes]


def _assert_no_direct_writes(container):
    """Worker không còn ghi bảng của core-api: mọi thay đổi đi qua worker_results."""
    assert_store_made_no_writes(container.metadata_store)


@pytest.mark.asyncio
async def test_process_job_publishes_request_lifecycle():
    container = _container()
    container.generate_curriculum_quiz_use_case.execute.return_value = Ok(
        _quiz_response("q1", "q2", "q3", rejected=[RejectedDraft("không bám nguồn", 2)])
    )
    bull_job = SimpleNamespace(data=_job_data())

    result = await _process_job(container, bull_job, None)

    assert result["quiz_count"] == 3
    assert result["rejected_count"] == 1
    running, succeeded = container.result_publisher.envelopes
    assert running["kind"] == succeeded["kind"] == "CONTENT_GENERATION_RESULT"
    # result_id tất định theo event_id của lệnh + giai đoạn.
    assert running["result_id"] == "CONTENT_GENERATION_RESULT.event-1.RUNNING"
    assert succeeded["result_id"] == "CONTENT_GENERATION_RESULT.event-1.SUCCEEDED"
    assert running["source_event_id"] == succeeded["source_event_id"] == "event-1"
    assert running["payload"] == {
        "request_id": "request-1",
        "course_id": "course-1",
        "content_type": "quiz",
        "status": "RUNNING",
    }
    payload = succeeded["payload"]
    assert payload["status"] == "SUCCEEDED"
    assert payload["lo_id"] == "lo-1"
    assert payload["generated_count"] == 3
    assert [i["quiz_id"] for i in payload["quiz_items"]] == [
        _quiz_item(q).quiz_id for q in ("q1", "q2", "q3")
    ]
    first = payload["quiz_items"][0]
    assert first["options"] == ["A", "B", "C", "D"]
    assert first["correct_answer"] == "B"
    assert first["bloom_level"] == 3
    assert first["verification_status"] == "VERIFIED"
    assert first["verification_evidence"]["judge_verdict"] == "supported"
    assert first["verification_evidence"]["requested_bloom_level"] == 3
    assert payload["rejected"] == [{"reason": "không bám nguồn", "regeneration_attempts": 2}]
    _assert_no_direct_writes(container)


@pytest.mark.asyncio
async def test_retried_job_reuses_the_same_result_ids():
    """Job gốc retry sau khi đã gửi: cùng result_id nên BullMQ bỏ bản trùng."""
    ids = []
    for _ in range(2):
        container = _container()
        container.generate_curriculum_quiz_use_case.execute.return_value = Ok(_quiz_response("q1"))
        await _process_job(
            container, SimpleNamespace(id="event-1", data=_job_data(), attemptsMade=1), None
        )
        ids.append([e["result_id"] for e in container.result_publisher.envelopes])
    assert ids[0] == ids[1]


@pytest.mark.asyncio
async def test_failure_before_last_attempt_publishes_nothing_terminal():
    """
    FAILED là trạng thái kết thúc: báo sớm thì lần thử sau không còn đưa
    request về SUCCEEDED được. Giữa các lần thử request vẫn RUNNING.
    """
    container = _container()
    container.generate_curriculum_quiz_use_case.execute.return_value = Err(RuntimeError("LLM timeout"))
    bull_job = SimpleNamespace(data=_job_data(), opts={"attempts": 5}, attemptsMade=1)

    with pytest.raises(RuntimeError, match="LLM timeout"):
        await _process_job(container, bull_job, None)

    assert _statuses(container) == ["RUNNING"]


@pytest.mark.asyncio
async def test_failure_on_last_attempt_publishes_failed_with_the_error():
    container = _container()
    container.generate_curriculum_quiz_use_case.execute.return_value = Err(RuntimeError("LLM timeout"))
    bull_job = SimpleNamespace(data=_job_data(), opts={"attempts": 5}, attemptsMade=4)

    with pytest.raises(RuntimeError, match="LLM timeout"):
        await _process_job(container, bull_job, None)

    assert _statuses(container) == ["RUNNING", "FAILED"]
    failed = container.result_publisher.envelopes[-1]
    assert failed["result_id"] == "CONTENT_GENERATION_RESULT.event-1.FAILED"
    assert failed["payload"]["error"] == "LLM timeout"
    _assert_no_direct_writes(container)


@pytest.mark.asyncio
async def test_unexpected_exception_on_last_attempt_still_fails_the_request():
    """Trước đây exception (không phải Err) để request treo RUNNING mãi."""
    container = _container()
    container.generate_curriculum_quiz_use_case.execute.side_effect = KeyError("boom")

    with pytest.raises(KeyError):
        await _process_job(container, SimpleNamespace(data=_job_data()), None)

    assert _statuses(container) == ["RUNNING", "FAILED"]


@pytest.mark.asyncio
async def test_publish_failure_at_start_fails_the_job_before_generating():
    container = _container(RecordingResultPublisher(fail_with=ConnectionError("redis down")))

    with pytest.raises(ConnectionError):
        await _process_job(container, SimpleNamespace(data=_job_data()), None)

    container.generate_curriculum_quiz_use_case.execute.assert_not_called()


@pytest.mark.asyncio
async def test_publish_failure_of_the_result_fails_the_job_so_it_is_retried():
    """Kết quả chưa tới Redis thì job gốc không được hoàn tất."""

    class FailOnSucceeded(RecordingResultPublisher):
        def publish_sync(self, kind, payload, **kwargs):
            if payload["status"] == "SUCCEEDED":
                raise TimeoutError("redis did not ack")
            return super().publish_sync(kind, payload, **kwargs)

    container = _container(FailOnSucceeded())
    container.generate_curriculum_quiz_use_case.execute.return_value = Ok(_quiz_response("q1"))

    with pytest.raises(TimeoutError):
        await _process_job(container, SimpleNamespace(data=_job_data()), None)

    assert _statuses(container) == ["RUNNING"]


def _card_job_data(**scope):
    data = _job_data()
    data["payload"]["type"] = "card"
    data["payload"]["scope"] = {"target_kind": "lo", "target_code": "lo-uuid", **scope}
    return data


def test_card_request_reads_target_and_defaults_count():
    request = _card_request(_parse_job(_card_job_data(source_document_ids=["doc-1"])))

    assert request.target_kind == "lo"
    assert request.target_code == "lo-uuid"
    assert request.count == DEFAULT_CARD_COUNT
    assert request.source_document_ids == ["doc-1"]


def test_card_request_rejects_count_out_of_range():
    with pytest.raises(ValueError, match="card count"):
        _card_request(_parse_job(_card_job_data(count=50)))


@pytest.mark.asyncio
async def test_card_job_runs_card_use_case_and_publishes_cards():
    container = _container()
    cards = tuple(
        GeneratedLessonCard(
            card_id=card_id,
            title=f"Card {card_id}",
            bullets=("Ý một",),
            key_insight="Ý chính.",
            example=None,
            misconception=None,
            check_question="Hỏi?",
            check_answer="Đáp.",
            card_index=index,
            primary_chunk_id="chunk-1",
            source_chunk_ids=("chunk-1",),
            model_id="writer-model",
            judge_model="judge-model",
            verification=_verification("INCONCLUSIVE"),
        )
        for index, card_id in enumerate(("c1", "c2"))
    )
    container.generate_lesson_cards_use_case.execute.return_value = Ok(
        GenerateLessonCardsResponse(course_id="course-1", lo_id="lo-1", cards=cards)
    )

    result = await _process_job(container, SimpleNamespace(data=_card_job_data(count=2)), None)

    assert result["card_ids"] == ["c1", "c2"]
    container.generate_curriculum_quiz_use_case.execute.assert_not_called()
    payload = container.result_publisher.envelopes[-1]["payload"]
    assert payload["status"] == "SUCCEEDED"
    assert payload["content_type"] == "card"
    assert payload["generated_count"] == 2
    assert "quiz_items" not in payload
    assert [c["title"] for c in payload["lesson_cards"]] == ["Card c1", "Card c2"]
    assert payload["lesson_cards"][0]["content"]["check_answer"] == "Đáp."
    assert payload["lesson_cards"][0]["verification_status"] == "INCONCLUSIVE"
    assert payload["lesson_cards"][0]["verification_evidence"]["inconclusive_reason"] == "judge_failed"
    _assert_no_direct_writes(container)


def _relations_job_data(**scope):
    data = _job_data()
    data["payload"]["type"] = "relations"
    data["payload"]["scope"] = {"source_document_ids": ["doc-1", "doc-2"], **scope}
    return data


def test_relations_request_reads_documents():
    request = _relations_request(_parse_job(_relations_job_data()))

    assert request.course_id == "course-1"
    assert request.document_ids == ["doc-1", "doc-2"]


@pytest.mark.parametrize("scope", [{"source_document_ids": []}, {"source_document_ids": "doc-1"}, {"source_document_ids": [1]}])
def test_relations_request_needs_a_document_list(scope):
    data = _relations_job_data()
    data["payload"]["scope"] = scope
    with pytest.raises(ValueError, match="source_document_ids"):
        _relations_request(_parse_job(data))


@pytest.mark.asyncio
async def test_relations_job_runs_extraction_and_publishes_written_count():
    container = _container()
    container.extract_relations_use_case.execute.return_value = Ok(
        ExtractRelationsResponse(
            course_id="course-1", documents=2, chunks_seen=30, chunks_processed=28,
            relations_written=41, relations_dropped=9, failed_batches=1,
        )
    )

    result = await _process_job(container, SimpleNamespace(data=_relations_job_data()), None)

    assert result["relations_written"] == 41
    assert result["lo_id"] is None
    container.generate_curriculum_quiz_use_case.execute.assert_not_called()
    assert _statuses(container) == ["RUNNING", "SUCCEEDED"]
    assert container.result_publisher.envelopes[-1]["payload"] == {
        "request_id": "request-1",
        "course_id": "course-1",
        "content_type": "relations",
        "status": "SUCCEEDED",
        "generated_count": 41,
    }
    _assert_no_direct_writes(container)


@pytest.mark.asyncio
async def test_unsupported_content_type_marks_request_failed():
    container = _container()
    data = _job_data()
    data["payload"]["type"] = "summary"

    with pytest.raises(UnrecoverableError):
        await _process_job(container, SimpleNamespace(data=data), None)

    [failed] = container.result_publisher.envelopes
    assert failed["payload"]["request_id"] == "request-1"
    assert failed["payload"]["status"] == "FAILED"
    # Loại lạ không phải giá trị của content_generation_requests.type.
    assert failed["payload"]["content_type"] is None
    _assert_no_direct_writes(container)


@pytest.mark.asyncio
async def test_unparseable_job_still_marks_request_failed():
    """
    Job hỏng mà không đánh dấu FAILED thì dòng nằm mãi ở QUEUED, và FE chặn
    mọi yêu cầu sinh quiz sau đó của khoá học bằng 409.
    """
    container = _container()
    data = _job_data()
    data["payload"]["scope"] = "không-phải-object"

    with pytest.raises(UnrecoverableError):
        await _process_job(container, SimpleNamespace(data=data), None)

    [failed] = container.result_publisher.envelopes
    assert failed["result_id"] == "CONTENT_GENERATION_RESULT.event-1.FAILED"
    assert failed["payload"]["request_id"] == "request-1"
    assert failed["payload"]["course_id"] == "course-1"
    assert failed["payload"]["status"] == "FAILED"
    assert "scope" in failed["payload"]["error"]
    _assert_no_direct_writes(container)


@pytest.mark.asyncio
async def test_request_id_recovered_from_aggregate_id_when_payload_is_broken():
    """Outbox đặt aggregate_id = request_id, nên payload hỏng hẳn vẫn cứu được."""
    container = _container()
    data = {
        "event_id": "event-1",
        "event_type": "CONTENT_GENERATION_REQUESTED",
        "aggregate_id": "request-42",
        "payload": None,
    }

    with pytest.raises(UnrecoverableError):
        await _process_job(container, SimpleNamespace(data=data), None)

    [failed] = container.result_publisher.envelopes
    assert failed["payload"]["request_id"] == "request-42"
    assert failed["payload"]["course_id"] is None
    assert failed["payload"]["status"] == "FAILED"


@pytest.mark.asyncio
async def test_job_without_any_request_id_does_not_crash():
    container = _container()

    with pytest.raises(UnrecoverableError):
        await _process_job(container, SimpleNamespace(data={"payload": None}), None)

    assert container.result_publisher.envelopes == []


@pytest.mark.asyncio
async def test_failing_to_publish_failed_makes_the_job_retry_instead_of_dying():
    """
    Trước đây ghi FAILED lỗi thì vẫn UnrecoverableError và dòng treo QUEUED.
    Nay job lỗi thường để BullMQ thử lại việc gửi; lỗi gốc vẫn nằm trong
    thông báo để không bị che.
    """
    container = _container(RecordingResultPublisher(fail_with=ConnectionError("redis down")))
    data = _job_data()
    data["payload"]["scope"] = 123

    with pytest.raises(RuntimeError) as raised:
        await _process_job(container, SimpleNamespace(data=data), None)

    assert not isinstance(raised.value, UnrecoverableError)
    assert "scope" in str(raised.value)
    assert isinstance(raised.value.__cause__, ConnectionError)


@pytest.mark.asyncio
async def test_content_generation_offloads_sync_work_from_event_loop(monkeypatch):
    calls = []
    real_to_thread = __import__("asyncio").to_thread

    async def tracked_to_thread(func, *args, **kwargs):
        calls.append(func.__name__)
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr("worker.runners.content_generation_worker.asyncio.to_thread", tracked_to_thread)

    container = _container()
    container.generate_curriculum_quiz_use_case.execute.return_value = Ok(_quiz_response("q1"))

    await _process_job(container, SimpleNamespace(data=_job_data()), None)

    # Dựng use case (đọc llm_model_bindings) và chạy use case đều ngoài loop;
    # chỉ việc gửi kết quả (async) chạy trên loop.
    assert calls == ["_plan", "_run_job_sync"]


@pytest.mark.asyncio
async def test_enrichment_worker_offloads_sync_work_from_event_loop(monkeypatch):
    from worker.runners.enrichment_worker import _process_job as process_enrichment_job
    from worker.use_cases.run_enrichment import RunEnrichmentResponse

    calls = []
    real_to_thread = __import__("asyncio").to_thread

    async def tracked_to_thread(func, *args, **kwargs):
        calls.append(func.__name__)
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr("worker.runners.enrichment_worker.asyncio.to_thread", tracked_to_thread)

    container = MagicMock()
    container.run_enrichment_use_case.execute.return_value = Ok(
        RunEnrichmentResponse(
            document_id="doc-1",
            concept_count=3,
            mention_count=5,
            card_count=2,
            quiz_count=4,
        )
    )
    job = SimpleNamespace(id="1", data={"document_id": "doc-1"})

    result = await process_enrichment_job(container, job, None)

    assert calls == ["_run_enrichment_sync"]
    assert result["quiz_count"] == 4



# ---------------------------------------------------------------------------
# Hợp đồng: envelope worker gửi đúng contracts/.../content-generation-result.json
# ---------------------------------------------------------------------------

_CONTRACTS = Path(__file__).resolve().parents[3] / "contracts" / "integration-events" / "v1"
_EVENT = "0193f2a1-7c4e-7d10-9a3b-5e6f7a8b9c01"
_REQUEST = "0193f2a1-7c4e-7d10-9a3b-5e6f7a8b9c02"
_COURSE = "0193f2a1-7c4e-7d10-9a3b-5e6f7a8b9c03"
_LO = "0193f2a1-7c4e-7d10-9a3b-5e6f7a8b9c04"
_CHUNK = "0193f2a1-7c4e-7d10-9a3b-5e6f7a8b9c05"


def _contract_validator():
    jsonschema = pytest.importorskip("jsonschema")
    if not _CONTRACTS.exists():
        pytest.skip("contracts/ không có trong image này")
    store = {}
    for path in _CONTRACTS.glob("*.json"):
        schema = json.loads(path.read_text(encoding="utf-8"))
        store[schema["$id"]] = schema
    schema = store["https://da-platform.dev/contracts/integration-events/v1/content-generation-result.json"]
    return jsonschema.Draft7Validator(
        schema,
        resolver=jsonschema.RefResolver.from_schema(schema, store=store),
        format_checker=jsonschema.Draft7Validator.FORMAT_CHECKER,
    )


def _real_use_case_container(content_type, question_type="MCQ_SINGLE"):
    from tests.support import as_llm_client
    from worker.use_cases.generate_curriculum_quiz import GenerateCurriculumQuizUseCase
    from worker.use_cases.generate_lesson_cards import GenerateLessonCardsUseCase

    store = MagicMock()
    store.get_curriculum.return_value = Ok((
        StoredCourse(course_id=_COURSE, code="CO3011", title_vi="Toán rời rạc"),
        [],
        [StoredLearningOutcome(lo_id=_LO, course_id=_COURSE, code="L.O.1.1", parent_code=None,
                               statement_vi="Hiểu cây", bloom_level=3)],
        [],
    ))
    store.list_chunks_for_lo.return_value = Ok([
        StoredChunkMetadata(chunk_id=_CHUNK, document_id=_COURSE, chunk_index=0,
                            content_text="Cây n đỉnh có n-1 cạnh; Dijkstra cần trọng số không âm.")
    ])
    llm = as_llm_client(MagicMock())
    llm.model_id = "writer-model"
    tail = {"explanation": "Theo slide.", "difficulty": "easy", "source_chunk_ids": [_CHUNK]}
    questions = {
        "MCQ_SINGLE": {"question": "Dijkstra cần gì?", "choices": ["Không âm", "Âm", "Bằng 0", "Tuỳ ý"],
                       "correct_index": 0},
        "MCQ_MULTI": {"question": "Chọn tính chất của cây.", "choices": ["Liên thông", "Có chu trình",
                      "Không chu trình", "Có 2 đỉnh"], "correct_indices": [0, 2]},
        "TRUE_FALSE": {"question": "Cây n đỉnh có n cạnh.", "is_true": False},
        "FILL_BLANK": {"question": "Cây n đỉnh có ____ cạnh.", "accepted_answers": ["n-1"]},
        "SHORT_ANSWER": {"question": "Vì sao cây n đỉnh có n-1 cạnh?", "model_answer": "Quy nạp.",
                         "rubric": ["Nêu quy nạp"]},
    }
    if content_type == "quiz":
        llm.generate.side_effect = [
            Ok(json.dumps({"questions": [{**questions[question_type], **tail}]}, ensure_ascii=False)),
            Ok('{"verdicts":[{"question_index":0,"verdict":"supported","reason":"Khớp nguồn"}]}'),
        ]
    else:
        card = {"title": "Cây", "key_insight": "Cây n đỉnh có n-1 cạnh.", "bullets": ["Liên thông"],
                "example": None, "misconception": None, "check_question": "Bao nhiêu cạnh?",
                "check_answer": "n-1.", "source_chunk_ids": [_CHUNK]}
        llm.generate.side_effect = [
            Ok(json.dumps({"cards": [card]}, ensure_ascii=False)),
            Ok("judge hỏng"),
            Ok("vẫn hỏng"),
        ]
    container = _container()
    container.generate_curriculum_quiz_use_case = GenerateCurriculumQuizUseCase(
        metadata_store=store, llm_client=llm
    )
    container.generate_lesson_cards_use_case = GenerateLessonCardsUseCase(
        metadata_store=store, llm_client=llm
    )
    return container, store


def _uuid_job(content_type, **scope):
    return SimpleNamespace(id=_EVENT, data={
        "event_id": _EVENT,
        "event_type": "CONTENT_GENERATION_REQUESTED",
        "aggregate_id": _REQUEST,
        "payload": {"request_id": _REQUEST, "course_id": _COURSE, "type": content_type,
                    "scope": {"target_kind": "lo", "target_code": _LO, "count": 1, **scope}},
    })


@pytest.mark.asyncio
@pytest.mark.parametrize("question_type", ["MCQ_SINGLE", "MCQ_MULTI", "TRUE_FALSE", "FILL_BLANK", "SHORT_ANSWER"])
async def test_quiz_envelopes_match_the_contract(question_type):
    validator = _contract_validator()
    container, store = _real_use_case_container("quiz", question_type)

    await _process_job(container, _uuid_job("quiz", question_type=question_type), None)

    for envelope in container.result_publisher.envelopes:
        errors = [e.message for e in validator.iter_errors(envelope)]
        assert not errors, (envelope["payload"]["status"], errors)
    [item] = container.result_publisher.envelopes[-1]["payload"]["quiz_items"]
    assert item["type"] == question_type
    assert item["verification_status"] == "VERIFIED"
    assert_store_made_no_writes(store)


@pytest.mark.asyncio
async def test_card_envelopes_match_the_contract():
    validator = _contract_validator()
    container, store = _real_use_case_container("card")

    await _process_job(container, _uuid_job("card"), None)

    statuses = []
    for envelope in container.result_publisher.envelopes:
        errors = [e.message for e in validator.iter_errors(envelope)]
        assert not errors, (envelope["payload"]["status"], errors)
        statuses.append(envelope["payload"]["status"])
    assert statuses == ["RUNNING", "SUCCEEDED"]
    [card] = container.result_publisher.envelopes[-1]["payload"]["lesson_cards"]
    assert card["verification_status"] == "INCONCLUSIVE"
    assert_store_made_no_writes(store)


@pytest.mark.asyncio
async def test_failed_envelope_matches_the_contract():
    validator = _contract_validator()
    container = _container()
    job = _uuid_job("quiz")
    job.data["payload"]["scope"] = "hỏng"

    with pytest.raises(UnrecoverableError):
        await _process_job(container, job, None)

    [envelope] = container.result_publisher.envelopes
    assert not [e.message for e in validator.iter_errors(envelope)]
