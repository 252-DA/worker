"""Unit tests for Bước 1.5 — ManimGenerator (AST Guard, Templates, MANIM_AUTO self-repair, Fallback)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from document_chunk.shared.result import Err, Ok

from worker.services.manim_generator import (
    ManimGenerationRequest,
    ManimGenerationResult,
    ManimGenerator,
    classify_runtime_error,
    extract_python_code,
    render_insertion_sort_template,
    render_merge_sort_template,
    validate_manim_source_code,
)
from worker.use_cases.render_micro_content_video import (
    RenderMicroContentVideoRequest,
    RenderMicroContentVideoUseCase,
)


VALID_SCENE_CODE = """from manim import *


class GeneratedScene(Scene):
    def construct(self):
        self.camera.background_color = "#0f172a"
        txt = Text("Merge Sort", font_size=32, color="#f8fafc")
        self.play(FadeIn(txt), run_time=1.0)
        self.wait(1.0)
"""


def test_extract_python_code_from_markdown():
    raw = f"Here is the animation:\n```python\n{VALID_SCENE_CODE}\n```\nEnjoy!"
    assert extract_python_code(raw) == VALID_SCENE_CODE.strip()


def test_ast_guard_allows_safe_scene():
    res = validate_manim_source_code(VALID_SCENE_CODE)
    assert res.valid is True
    assert res.scene_class_name == "GeneratedScene"


@pytest.mark.parametrize(
    "bad_code,expected_cat",
    [
        ("import os\nclass GeneratedScene(Scene):\n    pass", "AST_SECURITY_VIOLATION"),
        ("from subprocess import run\nclass GeneratedScene(Scene):\n    pass", "AST_SECURITY_VIOLATION"),
        ("from manim import *\nclass GeneratedScene(Scene):\n    def construct(self):\n        eval('1+1')", "AST_SECURITY_VIOLATION"),
        ("from manim import *\nclass GeneratedScene(Scene):\n    def construct(self):\n        self.add(MathTex('x^2'))", "LATEX_FORBIDDEN"),
        ("from manim import *\ndef construct(self): pass", "MISSING_SCENE_CLASS"),
        ("from manim import *\nclass Broken(Scene):\n    def construct(self)\n        pass", "SYNTAX_ERROR"),
    ],
)
def test_ast_guard_rejects_unsafe_or_invalid_code(bad_code: str, expected_cat: str):
    res = validate_manim_source_code(bad_code)
    assert res.valid is False
    assert res.error_category == expected_cat


def test_templates_pass_ast_guard():
    ins_code = render_insertion_sort_template({"array": [10, 5, 2, 8]})
    merge_code = render_merge_sort_template({"array": [38, 27, 43, 3, 9, 82]})
    assert validate_manim_source_code(ins_code).valid is True
    assert validate_manim_source_code(merge_code).valid is True
    assert "[10, 5, 2, 8]" in ins_code
    assert "[38, 27, 43, 3, 9, 82]" in merge_code


def test_classify_runtime_error():
    assert classify_runtime_error("AttributeError: 'Text' object has no attribute 'foo'") == "MANIM_API_ERROR"
    assert classify_runtime_error("RuntimeError: latex failed but did not produce a log") == "LATEX_FORBIDDEN"
    assert classify_runtime_error("ModuleNotFoundError: No module named 'manim'") == "MANIM_UNAVAILABLE"


@pytest.mark.asyncio
async def test_manim_auto_self_repairs_on_attempt_2(tmp_path: Path):
    """Kiểm chứng vòng lặp Coder <-> Error Fixer: lần 1 lỗi MathTex, lần 2 tự sửa thành công (Pass@2)."""
    llm = MagicMock()
    llm.generate.side_effect = [
        Ok("```python\nfrom manim import *\nclass GeneratedScene(Scene):\n    def construct(self):\n        self.add(MathTex('O(n)'))\n```"),
        Ok(f"```python\n{VALID_SCENE_CODE}\n```"),
    ]

    gen = ManimGenerator(llm_client=llm)

    async def fake_render(code, scene_class_name, request, output_mp4_path):
        Path(output_mp4_path).write_bytes(b"FAKE_MP4_DATA")
        return True, None, None

    gen._render_manim_subprocess = fake_render  # type: ignore[method-assign]

    out_mp4 = str(tmp_path / "merge_sort.mp4")
    req = ManimGenerationRequest(
        title="Merge Sort",
        narration="Thuật toán chia mảng làm đôi rồi trộn.",
        bullets=["Chia đôi", "Sắp xếp đệ quy", "Trộn hai nửa"],
        target_duration_seconds=16.0,
    )
    res = await gen.generate_scene_clip(req, out_mp4)

    assert res.success is True
    assert res.tier_used == "AUTO_LLM"
    assert res.attempts == 2
    assert res.clip_path == out_mp4
    assert Path(out_mp4).exists()
    # Kiểm tra prompt lần 2 có chứa phản hồi lỗi MathTex
    second_call_prompt = llm.generate.call_args_list[1].args[0]
    assert "MathTex" in second_call_prompt


@pytest.mark.asyncio
async def test_manim_auto_exhausts_attempts_and_returns_failure(tmp_path: Path):
    """Thất bại sau 3 lần thử phải trả về success=False kèm error_category rõ ràng."""
    llm = MagicMock()
    llm.generate.return_value = Ok(f"```python\n{VALID_SCENE_CODE}\n```")

    gen = ManimGenerator(llm_client=llm)
    gen._render_manim_subprocess = AsyncMock(  # type: ignore[method-assign]
        return_value=(False, "AttributeError: bad arg", "MANIM_API_ERROR")
    )

    out_mp4 = str(tmp_path / "fail.mp4")
    req = ManimGenerationRequest(title="Dijkstra", narration="Tìm đường đi ngắn nhất", max_attempts=3)
    res = await gen.generate_scene_clip(req, out_mp4)

    assert res.success is False
    assert res.attempts == 3
    assert res.error_category == "MANIM_API_ERROR"
    assert llm.generate.call_count == 3


@pytest.mark.asyncio
async def test_render_use_case_triggers_fallback_when_manim_auto_fails(tmp_path: Path):
    """Tích hợp Bước 1.5 -> Bước 1: khi ManimGenerator thất bại, tự động fallback về SLIDE_STATIC và gắn cờ fallback_used."""
    publisher = AsyncMock()
    publisher.publish.return_value = "res-1"

    storage = MagicMock()
    storage.upload.return_value = Ok("micro-content/c1/m1/ai-r1.mp4")

    tts = AsyncMock()
    tts.synthesize.return_value = 5.0

    slide_gen = MagicMock()
    slide_gen.render_slide.side_effect = lambda title, bullets, output_path, **kw: Path(output_path).write_bytes(b"PNG")

    composer = AsyncMock()
    composer.compose_video.return_value = 5.0

    failing_manim = AsyncMock(spec=ManimGenerator)
    failing_manim.generate_scene_clip.return_value = ManimGenerationResult(
        success=False,
        tier_used="AUTO_LLM",
        attempts=3,
        error="AttributeError: bad API",
        error_category="MANIM_API_ERROR",
    )

    use_case = RenderMicroContentVideoUseCase(
        result_publisher=publisher,
        file_storage=storage,
        tts_client=tts,
        slide_generator=slide_gen,
        video_composer=composer,
        manim_generator=failing_manim,
    )

    req = RenderMicroContentVideoRequest(
        micro_content_id="mc-1",
        course_id="c-1",
        asset_id="a-1",
        render_id="r-1",
        script={
            "voice": "vi-VN-HoaiMyNeural",
            "scenes": [
                {
                    "title": "Merge Sort Trace",
                    "narration": "Chia mảng và trộn hai nửa.",
                    "bullets": ["Divide", "Conquer", "Merge"],
                    "visual_mode": "MANIM_AUTO",
                }
            ],
        },
    )

    # Mock os.path.getsize cho final_mp4_path giả
    async def fake_compose(scenes, work_dir, output_mp4_path):
        Path(output_mp4_path).write_bytes(b"FINAL_MP4")
        # Đảm bảo SceneMedia đã fallback về None (không truyền clip lỗi)
        assert scenes[0].video_clip_path is None
        return 5.4

    composer.compose_video.side_effect = fake_compose

    result = await use_case.execute(req)
    assert result.is_ok()
    resp = result.unwrap()
    assert resp.fallback_used is True
    assert resp.fallbacks is not None and len(resp.fallbacks) == 1
    assert resp.fallbacks[0]["manim_attempts"] == 3
    assert resp.fallbacks[0]["error_category"] == "MANIM_API_ERROR"
