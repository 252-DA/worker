"""
Unit test cho các sửa lỗi đường ống micro-content:
- Priority 2: gộp outline theo lượng chữ & bỏ trang bìa/kết thúc, giữ heading_path,
  bỏ fallback chunks[:8], co giãn số cảnh, truyền instructor_prompt & course_title,
  không gắn nhầm cờ not_enough_evidence cho cảnh dài hợp lệ.
- Priority 3: kết xuất trang slide PDF gốc thành ảnh 1920x1080 kèm fallback thẻ chữ,
  và tách phụ đề câu dài thành các cue ngắn gọn.
- Priority 4: prompt micro-quiz bám đúng dải trang/kịch bản & lọc sạch chunk_id/Bloom
  khỏi lời giải thích cho người học.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pymupdf as fitz
import pytest
from PIL import Image

from document_chunk.domain.ports.llm_client import ILLMClient, LLMUsage
from document_chunk.domain.ports.metadata_store import (
    StoredChapter,
    StoredChunkMetadata,
    StoredCourse,
    StoredLearningOutcome,
)
from document_chunk.shared.result import Err, Ok
from worker.runners.content_generation_worker import ContentGenerationJob, _video_script_request
from worker.runners.micro_content_video_worker import parse_render_job
from worker.services.micro_content_subtitles import SubtitleCue, build_webvtt_and_timeline
from worker.services.slide_generator import SlideGenerator, render_pdf_slide_page
from worker.use_cases.generate_micro_outline import GenerateMicroOutlineRequest, GenerateMicroOutlineUseCase
from worker.use_cases.generate_micro_quiz import (
    GenerateMicroQuizRequest,
    GenerateMicroQuizUseCase,
    _sanitize_explanation,
)
from worker.use_cases.generate_video_script import GenerateVideoScriptRequest, GenerateVideoScriptUseCase
from worker.use_cases.render_micro_content_video import (
    RenderMicroContentVideoRequest,
    RenderMicroContentVideoUseCase,
)


def _chunk(cid: str, page: int | None, text: str, heading: tuple[str, ...] = ()) -> StoredChunkMetadata:
    return StoredChunkMetadata(
        chunk_id=cid,
        document_id="doc-1",
        chunk_index=0,
        heading_path=heading,
        page_number=page,
        content_length=len(text),
        content_text=text,
    )


def _fake_metadata_store(chunks: list[StoredChunkMetadata]):
    store = MagicMock()
    store.list_chunks.return_value = Ok(chunks)
    course = StoredCourse(
        course_id="course-uuid-1",
        code="CS301",
        title_vi="Dữ liệu lớn",
        title_en="Big Data",
    )
    los = [
        StoredLearningOutcome(
            lo_id="lo-1",
            course_id="course-uuid-1",
            code="LO1",
            parent_code="CL1",
            statement_vi="Giải thích khái niệm và đặc trưng 4V của Big Data",
            statement_en=None,
        )
    ]
    chapters = [StoredChapter(chapter_id="ch-1", course_id="course-uuid-1", code="1", title="Chương 1")]
    store.get_curriculum.return_value = Ok((course, chapters, los, []))
    store.list_los_by_chapter.return_value = Ok(los)
    store.record_llm_usage.return_value = Ok(None)
    return store


def _fake_llm(*json_payloads: dict):
    calls: list[dict] = []
    iterator = iter(json_payloads)

    class _FakeLlm(ILLMClient):
        @property
        def model_id(self) -> str:
            return "fake-model"

        def generate(self, prompt: str, system: str | None = None):
            calls.append({"prompt": prompt, "system": system})
            payload = next(iterator)
            return Ok(json.dumps(payload, ensure_ascii=False))

        def generate_with_usage(self, prompt: str, system: str | None = None):
            calls.append({"prompt": prompt, "system": system})
            payload = next(iterator)
            return Ok(
                (
                    json.dumps(payload, ensure_ascii=False),
                    LLMUsage(prompt_tokens=10, completion_tokens=10),
                )
            )

    return _FakeLlm(), calls


# =====================================================================
# Priority 2: Outline splitting, heading recovery, no fallback, script
# =====================================================================


def test_outline_skips_cover_and_closing_and_merges_sparse_sections():
    chunks = [
        _chunk("c1", 1, "Big Data Intro\nLecturer: Dr. Nguyen\nAI Lab", ("Cover",)),
        _chunk("c2", 2, " ".join(["Khái niệm dữ liệu lớn và bối cảnh ứng dụng."] * 14), ("Giới thiệu",)),
        _chunk("c3", 3, " ".join(["Thách thức lưu trữ truyền thống."] * 14), ("Thách thức",)),
        _chunk("c4", 4, " ".join(["Xử lý phân tán trên cụm máy."] * 14), ("Xử lý phân tán",)),
        _chunk("c5", 5, " ".join(["Tổng hợp luồng batch và stream."] * 14), ("Kiến trúc",)),
        _chunk("c6", 6, "Hình minh họa", ("Sơ đồ",)),  # rất ít chữ
        _chunk("c7", 7, "Tóm tắt ngắn gọn.", ("Tóm tắt",)),
        _chunk("c8", 8, "Thank you! Q&A", ("Kết thúc",)),
    ]
    store = _fake_metadata_store(chunks)
    llm, calls = _fake_llm(
        {
            "sections": [
                {"title": "Trang bìa và mở đầu", "summary": "Mở đầu", "page_from": 1, "page_to": 3, "suggested_lo_id": "lo-1"},
                {"title": "Kiến trúc xử lý", "summary": "Xử lý", "page_from": 4, "page_to": 5, "suggested_lo_id": "lo-1"},
                {"title": "Sơ đồ ngắn", "summary": "Sơ đồ", "page_from": 6, "page_to": 7, "suggested_lo_id": "lo-1"},
                {"title": "Cảm ơn", "summary": "Q&A", "page_from": 8, "page_to": 8, "suggested_lo_id": None},
            ]
        }
    )

    uc = GenerateMicroOutlineUseCase(metadata_store=store, llm_client=llm)
    res = uc.execute(GenerateMicroOutlineRequest(course_id="course-uuid-1", document_id="doc-1"))
    assert res.is_ok()
    sections = res.unwrap().sections

    # Bỏ trang 1 bìa và trang 8 Q&A; mục thưa chữ trang 6–7 được gộp vào mục liền kề
    assert sections[0].page_from == 2
    assert sections[-1].page_to == 7
    assert len(sections) == 2
    assert "Dữ liệu lớn" in calls[0]["prompt"]
    assert "course-uuid-1" not in calls[0]["prompt"]


def test_outline_long_deck_more_than_20_sections_is_merged_not_failed():
    # Tái hiện Ch2_LSH_lecture.pdf (101 trang) trên dev: model trả 21 mục, schema
    # chặn max 20 nên cả yêu cầu FAILED. Nay gộp bớt về <= 20, không mất trang cuối.
    total_pages = 100
    chunks = [
        _chunk(f"c{p}", p, f"Trang {p}: " + " ".join([f"Nội dung kiến thức số {p}."] * 30), (f"Mục {p}",))
        for p in range(1, total_pages + 1)
    ]
    store = _fake_metadata_store(chunks)
    raw_sections = [
        {
            "title": f"Nội dung {i + 1}",
            "summary": f"Tóm tắt {i + 1}",
            "page_from": i * 4 + 1,
            "page_to": i * 4 + 4,
            "suggested_lo_id": "lo-1",
        }
        for i in range(25)
    ]
    llm, calls = _fake_llm({"sections": raw_sections})

    uc = GenerateMicroOutlineUseCase(metadata_store=store, llm_client=llm)
    res = uc.execute(GenerateMicroOutlineRequest(course_id="course-uuid-1", document_id="doc-1"))

    assert res.is_ok(), res
    sections = res.unwrap().sections
    assert len(sections) == 20
    assert sections[0].page_from == 1
    assert sections[-1].page_to == total_pages
    for prev, cur in zip(sections, sections[1:]):
        assert cur.page_from == prev.page_to + 1
    assert len(calls) == 1  # không tốn lượt "repair" vì vượt trần
    assert "TỐI ĐA 20 nội dung" in calls[0]["prompt"]


def test_outline_prompt_tells_model_to_grow_sections_for_long_deck():
    from worker.prompts.micro_content import outline_suggestion

    def _pages(total_chars_per_page: int, n: int):
        return [
            {"page_number": p, "text_snippet": "x", "char_count": total_chars_per_page}
            for p in range(1, n + 1)
        ]

    long_deck = outline_suggestion(
        course_title="C",
        document_title="D",
        chapter_code=None,
        total_pages=100,
        page_summaries=_pages(600, 100),  # 60.000 ký tự -> >= 3.000 ký tự/mục
        candidate_los=[],
    )
    assert "3,000 ký tự chữ trở lên" in long_deck.user
    assert "TỐI ĐA 20 nội dung" in long_deck.user

    short_deck = outline_suggestion(
        course_title="C",
        document_title="D",
        chapter_code=None,
        total_pages=10,
        page_summaries=_pages(300, 10),
        candidate_los=[],
    )
    assert "1.000–2.500 ký tự" in short_deck.user


def test_outline_raw_pages_are_not_counted_twice():
    store = _fake_metadata_store([])
    text = "Nội dung trang hai về kiến trúc lambda."
    llm, calls = _fake_llm(
        {"sections": [{"title": "A", "summary": "a", "page_from": 1, "page_to": 2, "suggested_lo_id": "lo-1"}]}
    )
    uc = GenerateMicroOutlineUseCase(metadata_store=store, llm_client=llm)
    res = uc.execute(
        GenerateMicroOutlineRequest(
            course_id="course-uuid-1",
            document_id="doc-1",
            pages=[
                {"page_number": 1, "text": "Trang một."},
                {"page_number": 2, "text": text},
            ],
        )
    )
    assert res.is_ok(), res
    assert f"[Trang 2/2 | {len(text)} ký tự]\n{text}\n" in calls[0]["prompt"]
    assert f"{text}\n{text}" not in calls[0]["prompt"]


def test_video_script_preserves_heading_path_and_never_borrows_page_1():
    chunks = [
        _chunk("c1", 1, "Trang 1 tổng quan về Big Data", ("Trang bìa",)),
        _chunk(
            "c2",
            2,
            "to capture, manage, and process large datasets.",
            ("What is Big Data?", "new architectures, techniques, algorithms"),
        ),
    ]
    store = _fake_metadata_store(chunks)

    # Dải trang 5-6 chỉ là slide ảnh (không có chunk trong khoảng) -> phải báo lỗi, KHÔNG lấy trang 1
    llm_fail, _ = _fake_llm()
    uc_fail = GenerateVideoScriptUseCase(metadata_store=store, llm_client=llm_fail)
    missing = uc_fail.execute(
        GenerateVideoScriptRequest(
            course_id="course-uuid-1",
            micro_content_id="mc-1",
            document_id="doc-1",
            page_from=5,
            page_to=6,
            topic_title="Trang ảnh",
        )
    )
    assert missing.is_err()
    assert "5–6" in str(missing.error)

    # Trang 2 với nguồn thưa -> cho phép 2 cảnh (~28 từ), giữ đủ cụm từ bị đẩy lên heading_path, truyền instructor_prompt
    narration = " ".join(["Dữ liệu lớn đòi hỏi kiến trúc kỹ thuật và thuật toán mới để thu thập xử lý dữ liệu."] * 3)
    writer, writer_calls = _fake_llm(
        {
            "scenes": [
                {"title": "Định nghĩa Big Data", "slide_page": 2, "bullets": ["Kiến trúc mới"], "narration": narration},
                {"title": "Kết luận ngắn", "slide_page": 2, "bullets": ["Thuật toán xử lý"], "narration": narration},
            ]
        }
    )
    verifier, _ = _fake_llm(
        {
            "verdicts": [
                {"scene_index": 0, "verdict": "supported", "reason": "Bám slide"},
                {"scene_index": 1, "verdict": "supported", "reason": "Bám slide"},
            ]
        }
    )
    uc_ok = GenerateVideoScriptUseCase(
        metadata_store=store,
        llm_client=writer,
        verifier=SimpleNamespace(client=verifier, provider="fake", profile="judge"),  # type: ignore[arg-type]
    )
    res = uc_ok.execute(
        GenerateVideoScriptRequest(
            course_id="course-uuid-1",
            micro_content_id="mc-1",
            document_id="doc-1",
            page_from=2,
            page_to=2,
            topic_title="Định nghĩa dữ liệu lớn",
            instructor_prompt="Nhấn mạnh định nghĩa số 2",
        )
    )
    assert res.is_ok()
    assert res.unwrap().scene_count == 2
    sent_prompt = writer_calls[0]["prompt"]
    assert "new architectures, techniques, algorithms" in sent_prompt
    assert "Dữ liệu lớn" in sent_prompt
    assert "Nhấn mạnh định nghĩa số 2" in sent_prompt
    assert "Trang 1 tổng quan" not in sent_prompt


def test_video_script_120_word_grounded_scene_is_verified_not_flagged():
    chunks = [_chunk("c2", 2, "Big Data định nghĩa hệ thống thu thập lưu trữ phân tích dữ liệu lớn.", ("Định nghĩa",))]
    store = _fake_metadata_store(chunks)
    words_118 = " ".join(["Hệ thống xử lý dữ liệu quy mô lớn gồm bước thu thập lưu trữ và phân tích."] * 7)[:700]
    # Bảo đảm >= 35 từ và nằm trong (110, 135] từ
    narration_118 = " ".join(["word"] * 118)
    narration_45 = " ".join(["word"] * 45)
    writer, _ = _fake_llm(
        {
            "scenes": [
                {"title": "Cảnh 1", "slide_page": 2, "bullets": ["Ý chính"], "narration": narration_118},
                {"title": "Cảnh 2", "slide_page": 2, "bullets": ["Ý phụ"], "narration": narration_45},
            ]
        }
    )
    verifier, judge_calls = _fake_llm(
        {
            "verdicts": [
                {"scene_index": 0, "verdict": "supported", "reason": "Hợp lệ"},
                {"scene_index": 1, "verdict": "supported", "reason": "Hợp lệ"},
            ]
        }
    )
    uc = GenerateVideoScriptUseCase(
        metadata_store=store,
        llm_client=writer,
        verifier=SimpleNamespace(client=verifier, provider="fake", profile="judge"),  # type: ignore[arg-type]
    )
    res = uc.execute(
        GenerateVideoScriptRequest(
            course_id="course-uuid-1",
            micro_content_id="mc-1",
            document_id="doc-1",
            page_from=2,
            page_to=2,
            topic_title="Big Data",
        )
    )
    assert res.is_ok()
    assert all(sc["check"]["status"] == "VERIFIED" for sc in res.unwrap().script["scenes"])
    assert "40 từ hoặc > 110 từ" not in judge_calls[0]["prompt"]
    assert words_118  # keep helper unused lint calm if any


def test_video_script_request_extracts_instructor_prompt_from_scope():
    job = ContentGenerationJob(
        request_id="req-1",
        course_id="course-1",
        content_type="video_script",
        scope={
            "micro_content_id": "mc-1",
            "document_id": "doc-1",
            "page_from": 2,
            "page_to": 4,
            "instructor_prompt": "Tập trung ví dụ thực tế",
            "low_text": True,
        },
    )
    req = _video_script_request(job)
    assert req.instructor_prompt == "Tập trung ví dụ thực tế"
    assert req.low_text is True


# =====================================================================
# Priority 3: PDF slide rendering & long subtitle cue splitting
# =====================================================================


def test_parse_render_job_and_render_pdf_slide_pages_with_fallback(tmp_path):
    pdf_path = tmp_path / "deck.pdf"
    doc = fitz.open()
    p1 = doc.new_page(width=800, height=600)
    p1.insert_text((72, 100), "Slide 1 Cover", fontsize=24)
    p2 = doc.new_page(width=800, height=600)
    p2.insert_text((72, 100), "Slide 2 Architecture Diagram", fontsize=24)
    doc.save(str(pdf_path))
    doc.close()

    # Check single helper render_pdf_slide_page dimension & file creation
    single_out = tmp_path / "single.png"
    render_pdf_slide_page(str(pdf_path), 2, str(single_out))
    with Image.open(single_out) as img:
        assert img.size == (1920, 1080)

    # Check parse_render_job carries source_document_key & mime
    parsed = parse_render_job(
        {
            "event_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaac1",
            "event_type": "MICRO_CONTENT_RENDER_REQUESTED",
            "aggregate_type": "micro_content",
            "aggregate_id": "22222222-2222-4222-8222-222222222222",
            "payload": {
                "micro_content_id": "22222222-2222-4222-8222-222222222222",
                "course_id": "33333333-3333-4333-8333-333333333333",
                "asset_id": "55555555-5555-4555-8555-555555555555",
                "render_id": "66666666-6666-4666-8666-666666666666",
                "source_document_key": "courses/c1/deck.pdf",
                "source_document_mime_type": "application/pdf",
                "script": {"scenes": [{"title": "S2", "slide_page": 2, "narration": "Test"}]},
            },
        }
    )
    assert parsed.source_document_key == "courses/c1/deck.pdf"
    assert parsed.source_document_mime_type == "application/pdf"


@pytest.mark.asyncio
async def test_render_use_case_uses_pdf_slide_when_available_and_falls_back_on_bad_page(tmp_path):
    pdf_path = tmp_path / "deck.pdf"
    doc = fitz.open()
    doc.new_page(width=800, height=600).insert_text((72, 100), "Slide 1", fontsize=24)
    doc.new_page(width=800, height=600).insert_text((72, 100), "Slide 2", fontsize=24)
    pdf_bytes = doc.tobytes()
    doc.close()

    storage = MagicMock()
    def _download(_key: str, local_path: Path):
        local_path.write_bytes(pdf_bytes)
        return Ok(None)
    storage.download.side_effect = _download
    storage.upload.return_value = Ok("uploaded")

    slide_gen = MagicMock(spec=SlideGenerator)
    slide_gen.width = 1280
    slide_gen.height = 720
    def _gen_slide_states(**kwargs):
        paths = [
            f"{kwargs['output_prefix']}_{idx:02d}.png"
            for idx in range(len(kwargs["bullets"]) + 1)
        ]
        for path in paths:
            Path(path).write_bytes(b"fallback-png")
        return paths
    slide_gen.generate_slide_states.side_effect = _gen_slide_states

    class _FakeTts:
        async def synthesize(self, text: str, output_path: str, voice: str) -> float:
            Path(output_path).write_bytes(b"mp3")
            return 3.0

    class _FakeComposer:
        async def compose_video(self, scenes, work_dir: str, output_mp4_path: str) -> float:
            Path(output_mp4_path).write_bytes(b"mp4-bytes")
            return 6.0

    class _FakePublisher:
        async def publish(self, *args, **kwargs) -> str:
            return "msg-1"

    uc = RenderMicroContentVideoUseCase(
        result_publisher=_FakePublisher(),  # type: ignore[arg-type]
        file_storage=storage,
        tts_client=_FakeTts(),  # type: ignore[arg-type]
        slide_generator=slide_gen,
        video_composer=_FakeComposer(),  # type: ignore[arg-type]
    )

    res = await uc.execute(
        RenderMicroContentVideoRequest(
            micro_content_id="mc-1",
            course_id="course-1",
            asset_id="asset-1",
            render_id="render-1",
            source_document_key="courses/c1/deck.pdf",
            source_document_mime_type="application/pdf",
            script={
                "scenes": [
                    {"title": "Trang hợp lệ", "slide_page": 2, "bullets": ["A"], "narration": "Cảnh một."},
                    {"title": "Trang thiếu", "slide_page": 99, "bullets": ["B"], "narration": "Cảnh hai."},
                ]
            },
        )
    )
    assert res.is_ok()
    # Cảnh 1 render trực tiếp từ PDF (không gọi thẻ chữ), Cảnh 2 vượt tổng trang nên fallback sang thẻ chữ!
    assert slide_gen.generate_slide_states.call_count == 1
    assert slide_gen.generate_slide_states.call_args.kwargs["scene_index"] == 2


def test_subtitle_builder_splits_long_45_word_sentence_into_short_cues():
    long_sentence = (
        "Dữ liệu lớn là tập hợp dữ liệu có kích thước vượt quá khả năng thu thập, "
        "quản lý và xử lý của các công cụ cơ sở dữ liệu truyền thống, "
        "do đó chúng ta cần những kiến trúc phân tán mới và các thuật toán song song hiệu quả hơn."
    )
    vtt, _, timeline = build_webvtt_and_timeline(
        scenes=[{"title": "Khái niệm", "narration": long_sentence, "slide_page": 2}],
        scene_durations_ms=[12000],
        scene_cues=[[SubtitleCue(start_ms=0, end_ms=11500, text=long_sentence)]],
    )
    assert len(timeline) == 1
    cue_lines = [
        line for line in vtt.splitlines()
        if line and line != "WEBVTT" and "-->" not in line
    ]
    assert len(cue_lines) >= 3
    for line in cue_lines:
        assert len(line.split()) <= 14
        assert len(line) <= 84


# =====================================================================
# Priority 4: Micro-quiz scope alignment & clean learner explanations
# =====================================================================


def test_micro_quiz_scopes_to_slides_and_sanitizes_explanation():
    clean_ex = _sanitize_explanation(
        "Nguồn (chunk 29127832) nêu Big Data cần kiến trúc mới [1]. Đây là mức độ hiểu (understand) vì giải thích khái niệm."
    )
    assert "big data cần kiến trúc mới" in clean_ex.lower()
    assert "chunk" not in clean_ex.lower()
    assert "29127832" not in clean_ex
    assert "understand" not in clean_ex.lower()
    assert "mức độ hiểu" not in clean_ex.lower()
    assert "[1]" not in clean_ex

    chunks = [
        _chunk("c1", 1, "Bìa học phần Dữ liệu lớn", ("Trang bìa",)),
        _chunk("c2", 2, "vượt quá khả năng công cụ truyền thống.", ("What is Big Data?", "new architectures")),
    ]
    store = _fake_metadata_store(chunks)

    writer, gen_calls = _fake_llm(
        {
            "questions": [
                {
                    "question": "Đặc điểm cốt lõi của Big Data theo định nghĩa trong slide là gì?",
                    "choices": [
                        "Vượt quá khả năng xử lý của công cụ truyền thống",
                        "Chỉ dùng bảng tính Excel",
                        "Không cần lưu trữ",
                        "Chỉ có văn bản",
                    ],
                    "correct_index": 0,
                    "explanation": "Nguồn (chunk c2) nêu vượt quá khả năng công cụ truyền thống. Đây là mức độ hiểu (understand) vì tóm tắt khái niệm.",
                    "difficulty": "easy",
                    "lo_alignment_rationale": "Kiểm tra khái niệm trang 2",
                    "source_chunk_ids": ["c2"],
                }
            ]
        }
    )
    verifier, _ = _fake_llm(
        {"verdicts": [{"question_index": 0, "verdict": "supported", "reason": "Bám slide trang 2"}]}
    )
    uc = GenerateMicroQuizUseCase(
        metadata_store=store,
        llm_client=writer,
        verifier=SimpleNamespace(client=verifier, provider="fake", profile="judge"),  # type: ignore[arg-type]
    )
    res = uc.execute(
        GenerateMicroQuizRequest(
            course_id="course-uuid-1",
            micro_content_id="mc-1",
            lo_id="lo-1",
            lo_code="LO1",
            lo_statement_vi="Giải thích khái niệm và đặc trưng 4V (Volume, Velocity, Variety, Veracity)",
            document_id="doc-1",
            page_from=2,
            page_to=2,
            topic_title="Giới thiệu Big Data",
            count=1,
            question_types=("MCQ_SINGLE",),
        )
    )
    assert res.is_ok()
    item = res.unwrap().items[0]
    # Tên học phần thực tế được dùng, không lộ UUID; giữ text từ heading_path; explanation sạch thông tin nội bộ
    assert "Dữ liệu lớn" in gen_calls[0]["prompt"]
    assert "course-uuid-1" not in gen_calls[0]["prompt"]
    assert "new architectures" in gen_calls[0]["prompt"]
    assert "TUYỆT ĐỐI KHÔNG đặt câu hỏi về khái niệm" in gen_calls[0]["prompt"]
    assert "chunk" not in item.explanation.lower()
    assert "understand" not in item.explanation.lower()
    assert "mức độ hiểu" not in item.explanation.lower()


def test_slide_generator_builds_progressive_1080p_states(tmp_path):
    generator = SlideGenerator()
    paths = generator.generate_slide_states(
        title="RDD là gì?",
        bullets=["Phân tán trên cụm", "Bất biến", "Theo dõi lineage"],
        output_prefix=str(tmp_path / "scene"),
        scene_index=1,
        total_scenes=2,
    )

    assert len(paths) == 4
    pixel_frames = []
    for path in paths:
        with Image.open(path) as image:
            assert image.size == (1920, 1080)
            pixel_frames.append(image.tobytes())
    assert len(set(pixel_frames)) == 4
