"""Unit & contract tests for B2/B3/B5 micro-content generation and TTS normalization."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from jsonschema import Draft7Validator

from document_chunk.domain.ports.metadata_store import (
    StoredChunkMetadata,
    StoredCourse,
    StoredLearningOutcome,
)
from document_chunk.shared.result import Ok
from worker.llm_role import LlmRole
from worker.runners.content_generation_worker import _process_job
from worker.services.micro_content_subtitles import normalize_for_tts
from worker.use_cases.generate_micro_outline import (
    GenerateMicroOutlineRequest,
    GenerateMicroOutlineUseCase,
)
from worker.use_cases.generate_micro_quiz import (
    GenerateMicroQuizRequest,
    GenerateMicroQuizUseCase,
)
from worker.use_cases.generate_video_script import (
    GenerateVideoScriptRequest,
    GenerateVideoScriptUseCase,
)

from tests.support import RecordingResultPublisher, as_llm_client, assert_store_made_no_writes

_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "contracts"
    / "integration-events"
    / "v1"
    / "content-generation-result.json"
)


def _valid_narration(words: int = 50) -> str:
    base = (
        "Spark RDD là tập dữ liệu phân tán bất biến được chia thành nhiều phân vùng "
        "trên các nút tính toán trong cụm để xử lý song song và tự động phục hồi "
        "khi có sự cố nhờ đồ thị lineage lưu lại chuỗi phép biến đổi trước đó"
    )
    tokens = base.split()
    out: list[str] = []
    while len(out) < words:
        out.extend(tokens)
    return " ".join(out[:words]) + "."


def test_normalize_for_tts_cleans_markdown_urls_and_abbreviations():
    raw = "Xem `SparkContext` **tại** https://example.com [1, 2] — VD: tỷ lệ 80% & v.v."
    cleaned = normalize_for_tts(raw)
    assert "https://" not in cleaned
    assert "`" not in cleaned
    assert "**" not in cleaned
    assert "[1, 2]" not in cleaned
    assert "Ví dụ:" in cleaned
    assert "80 phần trăm" in cleaned
    assert "và" in cleaned
    assert "vân vân" in cleaned


class TestGenerateMicroOutlineUseCase:
    def test_splits_pages_flags_low_text_and_filters_invalid_lo(self, mock_metadata_store):
        course_id = "course-1"
        doc_id = "doc-spark"
        lo_valid = "0193f2a1-0000-0000-0000-000000000001"
        lo_parent = "0193f2a1-0000-0000-0000-000000000000"

        mock_metadata_store.list_chunks.return_value = Ok(
            [
                StoredChunkMetadata(
                    chunk_id="c1",
                    document_id=doc_id,
                    chunk_index=0,
                    heading_path=("RDD Basics",),
                    page_number=1,
                    content_text="A" * 240,
                ),
                StoredChunkMetadata(
                    chunk_id="c2",
                    document_id=doc_id,
                    chunk_index=1,
                    heading_path=("RDD Diagram",),
                    page_number=4,
                    content_text="Chỉ là chú thích ảnh",
                ),
            ]
        )
        mock_metadata_store.get_curriculum.return_value = Ok(
            (
                StoredCourse(course_id=course_id, code="CO3137", title_vi="Dữ liệu lớn"),
                [],
                [
                    StoredLearningOutcome(
                        lo_id=lo_parent,
                        course_id=course_id,
                        code="LO1",
                        parent_code=None,
                        statement_vi="Tổng quan",
                    ),
                    StoredLearningOutcome(
                        lo_id=lo_valid,
                        course_id=course_id,
                        code="LO1.1",
                        parent_code="LO1",
                        statement_vi="Hiểu kiến trúc RDD trong Apache Spark",
                    ),
                ],
                [],
            )
        )
        mock_metadata_store.list_los_by_chapter.return_value = Ok(
            [
                StoredLearningOutcome(
                    lo_id=lo_valid,
                    course_id=course_id,
                    code="LO1.1",
                    parent_code="LO1",
                    statement_vi="Hiểu kiến trúc RDD trong Apache Spark",
                )
            ]
        )

        writer = as_llm_client(MagicMock())
        writer.model_id = "deepseek-chat"
        writer.generate.return_value = Ok(
            json.dumps(
                {
                    "sections": [
                        {
                            "title": "Spark RDD cơ bản",
                            "summary": "Giới thiệu RDD và phân vùng.",
                            "page_from": 1,
                            "page_to": 3,
                            "suggested_lo_id": lo_valid,
                            "lo_rationale": "Trùng khớp trực tiếp LO1.1.",
                            "low_text": False,
                        },
                        {
                            "title": "Biểu diễn DAG",
                            "summary": "Sơ đồ khối DAG.",
                            "page_from": 3,  # lệch 1 trang -> phải đẩy lên 4
                            "page_to": 6,
                            "suggested_lo_id": "invalid-lo-id",
                            "low_text": False,
                        },
                    ]
                }
            )
        )

        uc = GenerateMicroOutlineUseCase(
            metadata_store=mock_metadata_store,
            llm_client=writer,
            llm_provider="deepseek",
            llm_profile="writer",
        )
        res = uc.execute(
            GenerateMicroOutlineRequest(
                course_id=course_id,
                document_id=doc_id,
                document_title="Spark.pdf",
                chapter_code="4",
                total_pages=6,
            )
        )

        assert res.is_ok()
        resp = res.unwrap()
        assert resp.section_count == 2
        s1, s2 = resp.sections
        assert (s1.page_from, s1.page_to) == (1, 3)
        assert s1.suggested_lo_id == lo_valid
        assert s1.low_text is False
        assert (s2.page_from, s2.page_to) == (4, 6)
        assert s2.suggested_lo_id is None
        assert s2.low_text is True  # tổng ký tự trang 4..6 < 200
        assert "[ƯU TIÊN]" in writer.generate.call_args.args[0]
        assert_store_made_no_writes(mock_metadata_store)


class TestGenerateVideoScriptUseCase:
    def _pages(self):
        return (
            {
                "page_number": 2,
                "content": "RDD là cấu trúc dữ liệu phân tán bất biến trên cụm Spark.",
                "chunk_ids": ["chunk-p2"],
            },
            {
                "page_number": 3,
                "content": "Transformations tạo RDD mới theo kiểu lazy và lưu lineage.",
                "chunk_ids": ["chunk-p3"],
            },
        )

    def test_retains_flagged_scenes_in_order_and_enforces_code_guards(self, mock_metadata_store):
        writer = as_llm_client(MagicMock())
        writer.model_id = "deepseek-chat"
        verifier = as_llm_client(MagicMock())
        verifier.model_id = "gemini-2.5-flash"

        # 4 cảnh ban đầu: cảnh 0 ổn, cảnh 1 bị lỗi cross-ref ("bài trước"), cảnh 2-3 ổn.
        initial_scenes = [
            {
                "title": "Khái niệm RDD",
                "slide_page": 2,
                "bullets": ["Tập dữ liệu bất biến", "Chia phân vùng trên cụm"],
                "narration": _valid_narration(52),
                "visual": {"type": "bullet_reveal", "motion": "none"},
            },
            {
                "title": "Lineage khôi phục",
                "slide_page": 3,
                "bullets": ["Ghi lại chuỗi biến đổi"],
                "narration": "Như bài trước đã nêu, " + _valid_narration(48),
            },
            {
                "title": "Lazy evaluation",
                "slide_page": 3,
                "bullets": ["Chờ action mới thực thi"],
                "narration": _valid_narration(50),
                "visual": {
                    "type": "manim",
                    "diagram": "flow",
                    "items": ["Transformation", "Action"],
                    "motion": "none",
                },
            },
            {
                "title": "Tổng kết dải trang",
                "slide_page": 9,  # ngoài dải [2, 3], code guard ép sửa hoặc clamp
                "bullets": ["Gắn kết transformation và action"],
                "narration": _valid_narration(46),
            },
        ]
        # Regeneration của cảnh 1 vẫn vi phạm cross-ref (đủ 40+ từ để kiểm tra nhánh cross-ref),
        # cảnh 3 hết vi phạm -> cảnh 1 giữ nguyên vị trí với FLAGGED, cảnh 3 VERIFIED.
        bad_regen = {
            "title": "Lineage khôi phục",
            "slide_page": 3,
            "bullets": ["Ghi lại chuỗi biến đổi"],
            "narration": "Như bài trước đã trình bày chi tiết, " + _valid_narration(46),
        }
        good_regen_3 = {
            "title": "Tổng kết dải trang",
            "slide_page": 3,
            "bullets": ["Gắn kết transformation và action"],
            "narration": _valid_narration(46),
        }
        writer.generate.side_effect = [
            Ok(json.dumps({"scenes": initial_scenes})),
            Ok(json.dumps(bad_regen)),
            Ok(json.dumps(good_regen_3)),
        ]
        verifier.generate.side_effect = [
            Ok(
                json.dumps(
                    {
                        "verdicts": [
                            {"scene_index": 0, "verdict": "supported", "reason": "Khớp trang 2."},
                            {"scene_index": 1, "verdict": "supported", "reason": "Ổn ngữ nghĩa."},
                            {"scene_index": 2, "verdict": "supported", "reason": "Khớp trang 3."},
                            {"scene_index": 3, "verdict": "supported", "reason": "Khớp trang 3."},
                        ]
                    }
                )
            ),
            Ok(
                json.dumps(
                    {
                        "verdicts": [
                            {"scene_index": 0, "verdict": "supported", "reason": "OK"},
                            {"scene_index": 1, "verdict": "supported", "reason": "OK"},
                            {"scene_index": 2, "verdict": "supported", "reason": "OK"},
                            {"scene_index": 3, "verdict": "supported", "reason": "OK"},
                        ]
                    }
                )
            ),
        ]

        uc = GenerateVideoScriptUseCase(
            metadata_store=mock_metadata_store,
            llm_client=writer,
            llm_provider="deepseek",
            verifier=LlmRole(client=verifier, provider="gemini"),
            max_verification_retries=1,
        )
        res = uc.execute(
            GenerateVideoScriptRequest(
                course_id="course-1",
                micro_content_id="mc-1",
                document_id="doc-spark",
                page_from=2,
                page_to=3,
                topic_title="Spark RDD cơ bản",
                pages=self._pages(),
            )
        )

        assert res.is_ok()
        resp = res.unwrap()
        scenes = resp.script["scenes"]
        assert len(scenes) == 4
        assert [s["title"] for s in scenes] == [
            "Khái niệm RDD",
            "Lineage khôi phục",
            "Lazy evaluation",
            "Tổng kết dải trang",
        ]
        assert scenes[0]["check"]["status"] == "VERIFIED"
        assert scenes[0]["visual"] == {"type": "bullet_reveal", "motion": "none"}
        assert scenes[1]["check"]["status"] == "FLAGGED"
        assert "tham chiếu ngoài bài" in scenes[1]["check"]["reason"]
        assert scenes[2]["check"]["status"] == "VERIFIED"
        assert scenes[2]["visual"] == {
            "type": "manim",
            "motion": "none",
            "diagram": "flow",
            "items": ["Transformation", "Action"],
        }
        assert scenes[3]["check"]["status"] == "VERIFIED"
        assert scenes[3]["slide_page"] == 3
        assert len(resp.rejected) == 1
        assert_store_made_no_writes(mock_metadata_store)


class TestGenerateMicroQuizUseCase:
    def test_scopes_chunks_to_page_range_and_uses_script_context(self, mock_metadata_store):
        mock_metadata_store.get_curriculum.return_value = Ok(None)
        mock_metadata_store.list_chunks.return_value = Ok(
            [
                StoredChunkMetadata(
                    chunk_id="chunk-outside",
                    document_id="doc-spark",
                    chunk_index=0,
                    page_number=1,
                    content_text="Trang 1 nói về Hadoop MapReduce cũ.",
                ),
                StoredChunkMetadata(
                    chunk_id="chunk-p2",
                    document_id="doc-spark",
                    chunk_index=1,
                    page_number=2,
                    content_text="RDD là tập dữ liệu phân tán bất biến trong Spark.",
                ),
            ]
        )

        writer = as_llm_client(MagicMock())
        writer.model_id = "deepseek-chat"
        verifier = as_llm_client(MagicMock())
        verifier.model_id = "gemini-2.5-flash"

        writer.generate.side_effect = [
            Ok(
                json.dumps(
                    {
                        "questions": [
                            {
                                "question": "Tính chất cốt lõi nào mô tả đúng RDD?",
                                "choices": [
                                    "Phân tán và bất biến",
                                    "Chỉ chạy một luồng",
                                    "Lưu toàn bộ trên đĩa cứng",
                                    "Không hỗ trợ phân vùng",
                                ],
                                "correct_index": 0,
                                "explanation": "Theo slide trang 2, RDD là tập dữ liệu phân tán bất biến.",
                                "difficulty": "easy",
                                "lo_alignment_rationale": "Kiểm tra hiểu khái niệm RDD.",
                                "source_chunk_ids": ["chunk-p2"],
                            }
                        ]
                    }
                )
            ),
            Ok(
                json.dumps(
                    {
                        "questions": [
                            {
                                "question": "RDD có thể thay đổi trực tiếp tại chỗ sau khi khởi tạo.",
                                "is_true": False,
                                "explanation": "RDD có tính bất biến.",
                                "difficulty": "easy",
                                "lo_alignment_rationale": "Củng cố tính bất biến của RDD.",
                                "source_chunk_ids": ["chunk-p2"],
                            }
                        ]
                    }
                )
            ),
        ]
        verifier.generate.side_effect = [
            Ok(json.dumps({"verdicts": [{"question_index": 0, "verdict": "supported", "reason": "OK"}]})),
            Ok(json.dumps({"verdicts": [{"question_index": 0, "verdict": "supported", "reason": "OK"}]})),
        ]

        uc = GenerateMicroQuizUseCase(
            metadata_store=mock_metadata_store,
            llm_client=writer,
            llm_provider="deepseek",
            verifier=LlmRole(client=verifier, provider="gemini"),
        )
        res = uc.execute(
            GenerateMicroQuizRequest(
                course_id="course-1",
                micro_content_id="mc-1",
                lo_id="0193f2a1-0000-0000-0000-000000000001",
                document_id="doc-spark",
                page_from=2,
                page_to=4,
                count=2,
                question_types=("MCQ_SINGLE", "TRUE_FALSE"),
                script={"scenes": [{"title": "RDD", "slide_page": 2, "narration": _valid_narration(45)}]},
            )
        )

        assert res.is_ok()
        resp = res.unwrap()
        assert resp.quiz_count == 2
        assert {item.question_type for item in resp.items} == {"MCQ_SINGLE", "TRUE_FALSE"}
        first_prompt = writer.generate.call_args_list[0].args[0]
        assert "chunk-p2" in first_prompt
        assert "Hadoop MapReduce" not in first_prompt
        assert "Kịch bản video đã soạn" in first_prompt
        assert_store_made_no_writes(mock_metadata_store)


@pytest.mark.asyncio
async def test_content_generation_worker_micro_types_match_contract_schema():
    import jsonschema

    contracts_dir = _CONTRACT_PATH.parent
    store = {}
    for path in contracts_dir.glob("*.json"):
        s = json.loads(path.read_text(encoding="utf-8"))
        store[s["$id"]] = s
    schema = store["https://da-platform.dev/contracts/integration-events/v1/content-generation-result.json"]
    validator = Draft7Validator(
        schema,
        resolver=jsonschema.RefResolver.from_schema(schema, store=store),
        format_checker=Draft7Validator.FORMAT_CHECKER,
    )

    from worker.use_cases.generate_micro_outline import GenerateMicroOutlineResponse, OutlineSection
    from worker.use_cases.generate_micro_quiz import GenerateMicroQuizResponse
    from worker.use_cases.generate_video_script import GenerateVideoScriptResponse
    from worker.use_cases.generated_content import GeneratedQuizItem
    from worker.use_cases.verification import Verification

    # 1. micro_outline
    c1 = MagicMock()
    c1.result_publisher = RecordingResultPublisher()
    c1.generate_micro_outline_use_case.execute.return_value = Ok(
        GenerateMicroOutlineResponse(
            course_id="0193f2a1-7c4e-7d10-9a3b-5e6f7a8b9c13",
            document_id="0193f2a1-7c4e-7d10-9a3b-5e6f7a8b9c14",
            sections=(
                OutlineSection(
                    title="Spark RDD",
                    summary="Giới thiệu RDD.",
                    page_from=1,
                    page_to=5,
                    suggested_lo_id="0193f2a1-7c4e-7d10-9a3b-5e6f7a8b9c15",
                    low_text=False,
                    lo_rationale="Khớp trực tiếp LO.",
                ),
            ),
        )
    )
    job1 = SimpleNamespace(
        data={
            "event_id": "evt-outline-1",
            "payload": {
                "request_id": "0193f2a1-7c4e-7d10-9a3b-5e6f7a8b9c12",
                "course_id": "0193f2a1-7c4e-7d10-9a3b-5e6f7a8b9c13",
                "type": "micro_outline",
                "scope": {"document_id": "0193f2a1-7c4e-7d10-9a3b-5e6f7a8b9c14", "chapter_code": "4"},
            },
        }
    )
    summary1 = await _process_job(c1, job1, None)
    assert summary1["section_count"] == 1
    assert validator.is_valid(c1.result_publisher.envelopes[-1])
    assert_store_made_no_writes(c1.metadata_store)

    # 2. video_script
    c2 = MagicMock()
    c2.result_publisher = RecordingResultPublisher()
    c2.generate_video_script_use_case.execute.return_value = Ok(
        GenerateVideoScriptResponse(
            course_id="0193f2a1-7c4e-7d10-9a3b-5e6f7a8b9c23",
            micro_content_id="0193f2a1-7c4e-7d10-9a3b-5e6f7a8b9c24",
            script={
                "voice": "vi-VN-HoaiMyNeural",
                "source_document_id": "0193f2a1-7c4e-7d10-9a3b-5e6f7a8b9c25",
                "source_page_from": 1,
                "source_page_to": 5,
                "scenes": [
                    {
                        "title": "RDD",
                        "bullets": ["Bất biến"],
                        "narration": _valid_narration(45),
                        "slide_page": 2,
                        "duration_sec": 18,
                        "check": {
                            "status": "VERIFIED",
                            "verdict": "supported",
                            "reason": "Khớp trang 2",
                            "writer_model": "deepseek-chat",
                            "judge_model": "gemini-2.5-flash",
                            "source_chunk_ids": ["0193f2a1-7c4e-7d10-9a3b-5e6f7a8b9c26"],
                        },
                    }
                ],
            },
        )
    )
    job2 = SimpleNamespace(
        data={
            "event_id": "evt-script-1",
            "payload": {
                "request_id": "0193f2a1-7c4e-7d10-9a3b-5e6f7a8b9c22",
                "course_id": "0193f2a1-7c4e-7d10-9a3b-5e6f7a8b9c23",
                "type": "video_script",
                "scope": {
                    "micro_content_id": "0193f2a1-7c4e-7d10-9a3b-5e6f7a8b9c24",
                    "source_document_id": "0193f2a1-7c4e-7d10-9a3b-5e6f7a8b9c25",
                    "source_page_from": 1,
                    "source_page_to": 5,
                },
            },
        }
    )
    summary2 = await _process_job(c2, job2, None)
    assert summary2["scene_count"] == 1
    assert list(validator.iter_errors(c2.result_publisher.envelopes[-1])) == []
    assert_store_made_no_writes(c2.metadata_store)

    # 3. micro_quiz
    c3 = MagicMock()
    c3.result_publisher = RecordingResultPublisher()
    c3.generate_micro_quiz_use_case.execute.return_value = Ok(
        GenerateMicroQuizResponse(
            course_id="0193f2a1-7c4e-7d10-9a3b-5e6f7a8b9c33",
            micro_content_id="0193f2a1-7c4e-7d10-9a3b-5e6f7a8b9c34",
            lo_id="0193f2a1-7c4e-7d10-9a3b-5e6f7a8b9c35",
            items=(
                GeneratedQuizItem(
                    question_id="8d3c6a52-5b1e-5c7a-9f0e-2a4b6c8d0e99",
                    question_type="MCQ_SINGLE",
                    question="Đặc tính của RDD?",
                    choices=("Bất biến", "Chỉ đơn luồng", "Không phân vùng", "Chỉ đọc ghi file cục bộ"),
                    correct_index=0,
                    correct_answer="Bất biến",
                    explanation="Slide trang 2 nêu RDD bất biến.",
                    bloom_level=2,
                    source_chunk_ids=("0193f2a1-7c4e-7d10-9a3b-5e6f7a8b9c36",),
                    model_id="deepseek-chat",
                    judge_model="gemini-2.5-flash",
                    verification=Verification(
                        status="VERIFIED",
                        judge_verdict="supported",
                        judge_reason="Khớp trang 2.",
                        inconclusive_reason=None,
                        regeneration_attempts=0,
                        regenerations_applied=0,
                    ),
                ),
            ),
        )
    )
    job3 = SimpleNamespace(
        data={
            "event_id": "evt-quiz-1",
            "payload": {
                "request_id": "0193f2a1-7c4e-7d10-9a3b-5e6f7a8b9c32",
                "course_id": "0193f2a1-7c4e-7d10-9a3b-5e6f7a8b9c33",
                "type": "micro_quiz",
                "scope": {
                    "micro_content_id": "0193f2a1-7c4e-7d10-9a3b-5e6f7a8b9c34",
                    "source_document_id": "0193f2a1-7c4e-7d10-9a3b-5e6f7a8b9c36",
                    "source_page_from": 1,
                    "source_page_to": 5,
                },
            },
        }
    )
    summary3 = await _process_job(c3, job3, None)
    assert summary3["quiz_count"] == 1
    assert list(validator.iter_errors(c3.result_publisher.envelopes[-1])) == []
    assert_store_made_no_writes(c3.metadata_store)
