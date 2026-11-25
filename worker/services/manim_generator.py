"""
Bước 1.5 — Engine sinh hoạt họa Manim tự động (2 tầng an toàn):
- Tầng 1 (Parameterized Templates): Sinh code Manim tất định cho các thuật toán
  kinh điển (Insertion Sort, Merge Sort) theo tham số JSON trích xuất từ slide.
- Tầng 2 (MANIM_AUTO Agent): Gọi LLM (Gemini / ILLMClient) sinh code Manim tổng
  quát cho thuật toán/khái niệm mới, kết hợp AST Security Guard và vòng lặp tự
  sửa lỗi (Coder <-> Error Fixer, tối đa 3 lần thử) trong tiến trình cô lập.
- Nếu thất bại sau số lần thử cho phép, trả về ManimGenerationResult(success=False)
  để RenderMicroContentVideoUseCase kích hoạt cơ chế Fallback an toàn về Slide tĩnh.
"""
from __future__ import annotations

import ast
import asyncio
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from document_chunk.shared.logger import get_logger
from document_chunk.shared.result import Result

logger = get_logger(__name__)

# Tập module được phép import trong code Manim do LLM sinh ra
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "manim",
        "numpy",
        "math",
        "random",
        "typing",
        "itertools",
        "collections",
        "dataclasses",
        "functools",
    }
)

# Các hàm builtin nguy hiểm bị cấm tuyệt đối
FORBIDDEN_CALLS: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "breakpoint",
    }
)

# Các lớp LaTeX bị cấm vì container worker không cài texlive
FORBIDDEN_TEX_NAMES: frozenset[str] = frozenset(
    {
        "MathTex",
        "Tex",
        "SingleStringMathTex",
        "TexTemplate",
        "BulletedList",
    }
)

MANIM_SYSTEM_PROMPT = """Bạn là chuyên gia lập trình hoạt họa sư phạm bằng Manim Community (v0.18–v0.22).
Nhiệm vụ: Viết MỘT file Python Manim độc lập minh họa rõ ràng, trực quan nội dung cảnh bài giảng.

QUY TẮC BẮT BUỘC (VI PHẠM SẼ BỊ TỪ CHỐI):
1. CHỈ import từ: `from manim import *`, `import numpy as np`, `import math`. CẤM import `os`, `sys`, `subprocess`, `pathlib`, v.v.
2. CẤM TUYỆT ĐỐI dùng `MathTex`, `Tex`, `TexTemplate`, `BulletedList` (môi trường không có LaTeX). Chỉ dùng `Text(...)` cho mọi chữ, số và ký hiệu toán học (dùng Unicode như ≤, ≥, →, ×, O(n log n)).
3. Bảng màu chuẩn LMS (Dark Theme):
   - Nền: `self.camera.background_color = "#0f172a"`
   - Ô mặc định: fill `#1e293b`, stroke `#64748b`
   - Ô đang xét (active): fill `#78350f`, stroke `#f59e0b`
   - Ô đã sắp xếp / kết quả (sorted): fill `#064e3b`, stroke `#10b981`
   - Điểm nhấn phụ (cyan): `#38bdf8`, chữ chính `#f8fafc`, chữ phụ `#94a3b8`
4. QUY TẮC VGROUP KHÔNG LỆCH TỌA ĐỘ:
   - Mỗi ô mảng là `cell = VGroup(RoundedRectangle(...), Text(...))`.
   - TUYỆT ĐỐI KHÔNG gọi `cell.animate.move_to(...)` và `cell[0].animate.set_fill(...)` trong CÙNG MỘT lệnh `self.play(...)` (Manim sẽ giữ nguyên hộp con ở vị trí cũ gây lệch khỏi chữ).
   - Hãy gọi `cell[0].set_fill(...).set_stroke(...)` TRƯỚC lệnh `self.play(cell.animate.move_to(...))` hoặc tách thành 2 lệnh `self.play` riêng biệt.
5. Bố cục an toàn 16:9 (khung 1280x720):
   - Tiêu đề ở cạnh trên (`to_edge(UP, buff=0.35)`), `font_size` từ 28–34, giới hạn độ dài để không tràn biên (hoặc gọi `.scale_to_fit_width(min(obj.width, 12.0))`).
   - Không để chữ chồng đè lên nhau; dùng `ReplacementTransform` khi đổi dòng trạng thái.
6. Thời lượng mục tiêu: Tổng `run_time` của các `self.play(...)` và `self.wait(...)` phải xấp xỉ thời lượng audio mục tiêu (sai lệch trong ±20%).
7. Định nghĩa duy nhất một lớp kế thừa `Scene` tên là `GeneratedScene`.
8. CHỈ trả về code Python trong duy nhất một khối ```python ... ```, không giải thích dài dòng bên ngoài.
"""


class SimpleLLMClient(Protocol):
    def generate(self, prompt: str, system: str | None = None) -> Result[str, Exception]: ...


@dataclass(frozen=True)
class ManimGenerationRequest:
    title: str
    narration: str
    bullets: list[str] = field(default_factory=list)
    target_duration_seconds: float = 15.0
    visual_mode: str = "MANIM_AUTO"
    template_name: str | None = None
    template_params: dict[str, Any] | None = None
    custom_source_code: str | None = None
    max_attempts: int = 3
    timeout_seconds: int = 60
    resolution: tuple[int, int] = (1280, 720)
    fps: int = 25


@dataclass(frozen=True)
class ManimGenerationResult:
    success: bool
    clip_path: str | None = None
    source_code: str | None = None
    tier_used: str = "NONE"  # "TEMPLATE" | "AUTO_LLM" | "CUSTOM" | "NONE"
    attempts: int = 0
    error: str | None = None
    error_category: str | None = None
    duration_seconds: float | None = None


@dataclass(frozen=True)
class CodeValidationResult:
    valid: bool
    scene_class_name: str | None = None
    error: str | None = None
    error_category: str | None = None


def extract_python_code(raw_text: str) -> str:
    """Trích xuất mã Python từ khối markdown ```python ... ``` hoặc trả về text đã strip."""
    if not raw_text:
        return ""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", raw_text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return raw_text.strip()


def validate_manim_source_code(code: str) -> CodeValidationResult:
    """
    Kiểm tra an toàn tĩnh (AST Guard) trước khi chạy Manim trong subprocess:
    1. Kiểm tra cú pháp Python hợp lệ.
    2. Chặn import ngoài whitelist (os, sys, subprocess, ...).
    3. Chặn gọi hàm nguy hiểm (eval, exec, open, __import__, ...).
    4. Chặn MathTex / Tex (không có LaTeX trong container).
    5. Xác nhận tồn tại ít nhất 1 lớp kế thừa Scene và trả về tên lớp đó.
    """
    if not code or not code.strip():
        return CodeValidationResult(
            valid=False,
            error="Mã nguồn Manim trống",
            error_category="EMPTY_CODE",
        )

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return CodeValidationResult(
            valid=False,
            error=f"SyntaxError tại dòng {exc.lineno}: {exc.msg}",
            error_category="SYNTAX_ERROR",
        )

    scene_classes: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                if root_pkg not in ALLOWED_IMPORTS:
                    return CodeValidationResult(
                        valid=False,
                        error=f"Import bị cấm vì lý do bảo mật: '{alias.name}'",
                        error_category="AST_SECURITY_VIOLATION",
                    )
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod not in ALLOWED_IMPORTS:
                return CodeValidationResult(
                    valid=False,
                    error=f"Import bị cấm vì lý do bảo mật: 'from {node.module} import ...'",
                    error_category="AST_SECURITY_VIOLATION",
                )
        elif isinstance(node, ast.Call):
            func_name: str | None = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            if func_name in FORBIDDEN_CALLS:
                return CodeValidationResult(
                    valid=False,
                    error=f"Lời gọi hàm bị cấm vì lý do bảo mật: '{func_name}()'",
                    error_category="AST_SECURITY_VIOLATION",
                )
            if func_name in FORBIDDEN_TEX_NAMES:
                return CodeValidationResult(
                    valid=False,
                    error=(
                        f"Không được sử dụng '{func_name}' (môi trường không cài LaTeX/texlive). "
                        "Hãy thay bằng Text(...) với ký tự Unicode."
                    ),
                    error_category="LATEX_FORBIDDEN",
                )
        elif isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_name = (
                    base.id
                    if isinstance(base, ast.Name)
                    else base.attr
                    if isinstance(base, ast.Attribute)
                    else ""
                )
                if base_name in ("Scene", "MovingCameraScene", "ThreeDScene", "ZoomedScene"):
                    scene_classes.append(node.name)

    if not scene_classes:
        return CodeValidationResult(
            valid=False,
            error="Không tìm thấy lớp kế thừa Scene (ví dụ: class GeneratedScene(Scene):).",
            error_category="MISSING_SCENE_CLASS",
        )

    chosen = "GeneratedScene" if "GeneratedScene" in scene_classes else scene_classes[0]
    return CodeValidationResult(valid=True, scene_class_name=chosen)


def classify_runtime_error(stderr_text: str) -> str:
    """Phân loại lỗi thực thi Manim phục vụ Error Analysis ở Chương 5."""
    low = (stderr_text or "").lower()
    if "latex" in low or "dvisvgm" in low or "pdflatex" in low:
        return "LATEX_FORBIDDEN"
    if "attributeerror" in low or "typeerror" in low or "got an unexpected keyword" in low:
        return "MANIM_API_ERROR"
    if "nameerror" in low or "unboundlocalerror" in low:
        return "NAME_ERROR"
    if "indexerror" in low or "keyerror" in low or "valueerror" in low:
        return "LOGIC_ERROR"
    if "syntaxerror" in low or "indentationerror" in low:
        return "SYNTAX_ERROR"
    if "no module named 'manim'" in low or "no module named \"manim\"" in low:
        return "MANIM_UNAVAILABLE"
    return "RUNTIME_ERROR"


# =============================================================================
# TẦNG 1: PARAMETERIZED TEMPLATES (TẤT ĐỊNH, AN TOÀN 100%)
# =============================================================================

def _sanitize_int_array(raw: Any, default: list[int], min_len: int = 3, max_len: int = 8) -> list[int]:
    if isinstance(raw, list):
        cleaned: list[int] = []
        for item in raw:
            try:
                cleaned.append(int(item))
            except (TypeError, ValueError):
                continue
        if min_len <= len(cleaned) <= max_len:
            return cleaned
    return list(default)


def render_insertion_sort_template(params: dict[str, Any] | None = None) -> str:
    """Sinh code Manim tham số hóa cho Straight Insertion Sort với mảng đầu vào bất kỳ."""
    params = params or {}
    arr = _sanitize_int_array(params.get("array"), default=[23, 78, 45, 8, 32, 56])
    title_text = str(params.get("title") or "Straight Insertion Sort — Array Trace")[:60]

    return f'''from manim import *
import numpy as np


class GeneratedScene(Scene):
    def construct(self):
        self.camera.background_color = "#0f172a"
        BG_DEFAULT = "#1e293b"
        STROKE_DEFAULT = "#64748b"
        BG_SORTED = "#064e3b"
        STROKE_SORTED = "#10b981"
        BG_ACTIVE = "#78350f"
        STROKE_ACTIVE = "#f59e0b"
        TEXT_WHITE = "#f8fafc"
        TEXT_MUTED = "#94a3b8"

        values = {arr!r}
        n = len(values)
        box_w = min(1.2, 9.0 / max(n, 1))
        box_h = box_w
        gap = 0.28
        step_x = box_w + gap
        start_x = -((n - 1) * step_x) / 2.0
        base_y = -0.35
        temp_y = base_y + box_h + 0.65

        def slot_pos(idx: int) -> np.ndarray:
            return np.array([start_x + idx * step_x, base_y, 0.0])

        title = Text({title_text!r}, font_size=32, weight=BOLD, color=TEXT_WHITE)
        title.to_edge(UP, buff=0.4)
        if title.width > 12.0:
            title.scale_to_fit_width(12.0)

        banner = Text(f"Initial array: {{values}}", font_size=24, color="#38bdf8")
        banner.next_to(title, DOWN, buff=0.3)

        cells = []
        idx_labels = VGroup()
        for i, v in enumerate(values):
            rect = RoundedRectangle(
                width=box_w, height=box_h, corner_radius=0.14,
                fill_color=BG_DEFAULT, fill_opacity=1.0,
                stroke_color=STROKE_DEFAULT, stroke_width=2.5,
            )
            num = Text(str(v), font_size=30, weight=BOLD, color=TEXT_WHITE)
            cell = VGroup(rect, num).move_to(slot_pos(i))
            cells.append(cell)
            lbl = Text(str(i), font_size=18, color=TEXT_MUTED)
            lbl.next_to(slot_pos(i), DOWN, buff=box_h / 2 + 0.22)
            idx_labels.add(lbl)

        self.play(FadeIn(title), FadeIn(banner), FadeIn(VGroup(*cells)), FadeIn(idx_labels), run_time=0.9)
        cells[0][0].set_fill(BG_SORTED).set_stroke(STROKE_SORTED, width=3.0)
        self.wait(0.6)

        arr_state = list(values)
        for current in range(1, n):
            key_val = arr_state[current]
            key_cell = cells[current]
            b_pick = Text(f"Pass {{current}}: Extract temp = {{key_val}}", font_size=24, color=STROKE_ACTIVE)
            b_pick.next_to(title, DOWN, buff=0.3)

            key_cell[0].set_fill(BG_ACTIVE).set_stroke(STROKE_ACTIVE, width=4.0)
            self.play(
                ReplacementTransform(banner, b_pick),
                key_cell.animate.move_to(np.array([slot_pos(current)[0], temp_y, 0.0])),
                run_time=0.55,
            )
            banner = b_pick

            walker = current - 1
            shifts = []
            while walker >= 0 and arr_state[walker] > key_val:
                arr_state[walker + 1] = arr_state[walker]
                cells[walker + 1] = cells[walker]
                shifts.append(cells[walker].animate.move_to(slot_pos(walker + 1)))
                walker -= 1

            target_idx = walker + 1
            arr_state[target_idx] = key_val
            cells[target_idx] = key_cell

            if shifts:
                shifts.append(key_cell.animate.move_to(np.array([slot_pos(target_idx)[0], temp_y, 0.0])))
                self.play(*shifts, run_time=0.65)

            b_ins = Text(f"Insert {{key_val}} at index [{{target_idx}}] -> {{arr_state}}", font_size=22, color=STROKE_SORTED)
            b_ins.next_to(title, DOWN, buff=0.3)
            if b_ins.width > 12.0:
                b_ins.scale_to_fit_width(12.0)

            key_cell[0].set_fill(BG_SORTED).set_stroke(STROKE_SORTED, width=3.0)
            self.play(
                ReplacementTransform(banner, b_ins),
                key_cell.animate.move_to(slot_pos(target_idx)),
                run_time=0.6,
            )
            banner = b_ins
            self.wait(0.35)

        b_done = Text(f"Sorted Complete: {{arr_state}}", font_size=26, weight=BOLD, color=STROKE_SORTED)
        b_done.next_to(title, DOWN, buff=0.3)
        self.play(ReplacementTransform(banner, b_done), run_time=0.5)
        self.play(*[Indicate(c, color=STROKE_SORTED, scale_factor=1.06) for c in cells], run_time=0.5)
        self.wait(1.2)
'''


def render_merge_sort_template(params: dict[str, Any] | None = None) -> str:
    """Sinh code Manim tham số hóa cho Merge Sort (Divide & Conquer) với mảng bất kỳ."""
    params = params or {}
    arr = _sanitize_int_array(params.get("array"), default=[38, 27, 43, 3, 9, 82])
    title_text = str(params.get("title") or "Merge Sort — Divide & Conquer Trace")[:60]

    return f'''from manim import *
import numpy as np


class GeneratedScene(Scene):
    def construct(self):
        self.camera.background_color = "#0f172a"
        BG_DEFAULT = "#1e293b"
        STROKE_DEFAULT = "#64748b"
        BG_LEFT = "#1e3a5f"
        STROKE_LEFT = "#38bdf8"
        BG_RIGHT = "#4c1d95"
        STROKE_RIGHT = "#a78bfa"
        BG_SORTED = "#064e3b"
        STROKE_SORTED = "#10b981"
        TEXT_WHITE = "#f8fafc"

        values = {arr!r}
        n = len(values)
        box_w = min(1.05, 9.2 / max(n, 1))
        box_h = 0.85
        gap = 0.22

        def make_array_row(vals, y_pos, fill_c=BG_DEFAULT, stroke_c=STROKE_DEFAULT, x_center=0.0):
            grp = VGroup()
            step = box_w + gap
            sx = x_center - ((len(vals) - 1) * step) / 2.0
            for i, v in enumerate(vals):
                rect = RoundedRectangle(
                    width=box_w, height=box_h, corner_radius=0.12,
                    fill_color=fill_c, fill_opacity=1.0,
                    stroke_color=stroke_c, stroke_width=2.5,
                )
                txt = Text(str(v), font_size=26, weight=BOLD, color=TEXT_WHITE)
                cell = VGroup(rect, txt).move_to(np.array([sx + i * step, y_pos, 0.0]))
                grp.add(cell)
            return grp

        title = Text({title_text!r}, font_size=32, weight=BOLD, color=TEXT_WHITE)
        title.to_edge(UP, buff=0.35)
        if title.width > 12.0:
            title.scale_to_fit_width(12.0)

        banner = Text(f"Input array (n = {{n}}): {{values}}", font_size=24, color=STROKE_LEFT)
        banner.next_to(title, DOWN, buff=0.25)

        root_row = make_array_row(values, y_pos=1.25)
        self.play(FadeIn(title), FadeIn(banner), FadeIn(root_row), run_time=1.0)
        self.wait(0.8)

        # Phase 1: Divide into Left & Right halves
        mid = n // 2
        left_vals = values[:mid]
        right_vals = values[mid:]
        b_div = Text(f"Divide: Left {{left_vals}}  |  Right {{right_vals}}", font_size=23, color="#f59e0b")
        b_div.next_to(title, DOWN, buff=0.25)

        left_row = make_array_row(left_vals, y_pos=-0.1, fill_c=BG_LEFT, stroke_c=STROKE_LEFT, x_center=-3.1)
        right_row = make_array_row(right_vals, y_pos=-0.1, fill_c=BG_RIGHT, stroke_c=STROKE_RIGHT, x_center=3.1)
        arr_l = Arrow(root_row.get_bottom() + LEFT * 0.5, left_row.get_top(), color=STROKE_LEFT, buff=0.15)
        arr_r = Arrow(root_row.get_bottom() + RIGHT * 0.5, right_row.get_top(), color=STROKE_RIGHT, buff=0.15)

        self.play(
            ReplacementTransform(banner, b_div),
            GrowArrow(arr_l), GrowArrow(arr_r),
            FadeIn(left_row), FadeIn(right_row),
            run_time=1.3,
        )
        banner = b_div
        self.wait(0.9)

        # Phase 2: Sort each half recursively
        left_sorted = sorted(left_vals)
        right_sorted = sorted(right_vals)
        b_sub = Text(f"Conquer (Recursive Sort): {{left_sorted}}  &  {{right_sorted}}", font_size=23, color=STROKE_LEFT)
        b_sub.next_to(title, DOWN, buff=0.25)

        left_s_row = make_array_row(left_sorted, y_pos=-0.1, fill_c=BG_LEFT, stroke_c=STROKE_LEFT, x_center=-3.1)
        right_s_row = make_array_row(right_sorted, y_pos=-0.1, fill_c=BG_RIGHT, stroke_c=STROKE_RIGHT, x_center=3.1)

        self.play(
            ReplacementTransform(banner, b_sub),
            ReplacementTransform(left_row, left_s_row),
            ReplacementTransform(right_row, right_s_row),
            run_time=1.2,
        )
        banner = b_sub
        self.wait(0.9)

        # Phase 3: Merge two sorted halves element-by-element
        merged_vals = sorted(values)
        b_merge = Text("Merge: Compare front elements & place smaller value", font_size=23, color=STROKE_SORTED)
        b_merge.next_to(title, DOWN, buff=0.25)
        self.play(ReplacementTransform(banner, b_merge), run_time=0.6)
        banner = b_merge

        merged_row = make_array_row(merged_vals, y_pos=-1.9, fill_c=BG_SORTED, stroke_c=STROKE_SORTED, x_center=0.0)
        merge_arr_l = Arrow(left_s_row.get_bottom(), merged_row.get_top() + LEFT * 0.6, color=STROKE_SORTED, buff=0.15)
        merge_arr_r = Arrow(right_s_row.get_bottom(), merged_row.get_top() + RIGHT * 0.6, color=STROKE_SORTED, buff=0.15)
        self.play(GrowArrow(merge_arr_l), GrowArrow(merge_arr_r), run_time=0.6)

        for cell in merged_row:
            self.play(FadeIn(cell, shift=DOWN * 0.25), run_time=0.45)

        b_done = Text(f"Merge Complete — O(n log n): {{merged_vals}}", font_size=25, weight=BOLD, color=STROKE_SORTED)
        b_done.next_to(title, DOWN, buff=0.25)
        self.play(ReplacementTransform(banner, b_done), run_time=0.5)
        self.play(*[Indicate(c, color=STROKE_SORTED, scale_factor=1.06) for c in merged_row], run_time=0.6)
        self.wait(1.5)
'''


TEMPLATE_REGISTRY: dict[str, Any] = {
    "insertion_sort": render_insertion_sort_template,
    "straight_insertion_sort": render_insertion_sort_template,
    "merge_sort": render_merge_sort_template,
}


class ManimGenerator:
    """
    Service sinh hoạt họa Manim 2 tầng:
    - Tầng 1: Parameterized Templates (nếu template_name khớp TEMPLATE_REGISTRY).
    - Tầng 2: MANIM_AUTO Agent (LLM Coder <-> Error Fixer, tối đa max_attempts vòng).
    - Chế độ MANIM_CUSTOM: Render trực tiếp mã nguồn do giảng viên cung cấp (vẫn qua AST Guard).
    """

    def __init__(
        self,
        llm_client: SimpleLLMClient | None = None,
        python_bin: str | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.python_bin = python_bin or sys.executable

    async def generate_scene_clip(
        self,
        request: ManimGenerationRequest,
        output_mp4_path: str,
    ) -> ManimGenerationResult:
        mode = (request.visual_mode or "MANIM_AUTO").upper()

        # Nhánh 0: MANIM_CUSTOM (giảng viên tự cung cấp/chỉnh sửa code)
        if mode == "MANIM_CUSTOM" and request.custom_source_code:
            return await self._validate_and_render_single(
                code=request.custom_source_code,
                request=request,
                output_mp4_path=output_mp4_path,
                tier_used="CUSTOM",
            )

        # Tầng 1: Parameterized Template
        template_key = (request.template_name or "").strip().lower()
        if mode == "MANIM_TEMPLATE" or (template_key and template_key in TEMPLATE_REGISTRY):
            builder = TEMPLATE_REGISTRY.get(template_key)
            if builder is None:
                return ManimGenerationResult(
                    success=False,
                    tier_used="TEMPLATE",
                    attempts=0,
                    error=f"Không tìm thấy Manim template '{request.template_name}'",
                    error_category="TEMPLATE_NOT_FOUND",
                )
            code = builder(request.template_params or {"title": request.title})
            return await self._validate_and_render_single(
                code=code,
                request=request,
                output_mp4_path=output_mp4_path,
                tier_used="TEMPLATE",
            )

        # Tầng 2: MANIM_AUTO Agent (Coder <-> Error Fixer)
        if self.llm_client is None:
            return ManimGenerationResult(
                success=False,
                tier_used="AUTO_LLM",
                attempts=0,
                error="LLM client chưa được cấu hình cho chế độ MANIM_AUTO",
                error_category="LLM_UNAVAILABLE",
            )

        return await self._run_auto_agent_loop(request=request, output_mp4_path=output_mp4_path)

    async def _validate_and_render_single(
        self,
        *,
        code: str,
        request: ManimGenerationRequest,
        output_mp4_path: str,
        tier_used: str,
    ) -> ManimGenerationResult:
        validation = validate_manim_source_code(code)
        if not validation.valid or not validation.scene_class_name:
            return ManimGenerationResult(
                success=False,
                source_code=code,
                tier_used=tier_used,
                attempts=1,
                error=validation.error,
                error_category=validation.error_category,
            )

        ok, err_msg, err_cat = await self._render_manim_subprocess(
            code=code,
            scene_class_name=validation.scene_class_name,
            request=request,
            output_mp4_path=output_mp4_path,
        )
        if ok:
            return ManimGenerationResult(
                success=True,
                clip_path=output_mp4_path,
                source_code=code,
                tier_used=tier_used,
                attempts=1,
            )
        return ManimGenerationResult(
            success=False,
            source_code=code,
            tier_used=tier_used,
            attempts=1,
            error=err_msg,
            error_category=err_cat,
        )

    async def _run_auto_agent_loop(
        self,
        *,
        request: ManimGenerationRequest,
        output_mp4_path: str,
    ) -> ManimGenerationResult:
        max_attempts = max(1, min(request.max_attempts, 5))
        last_code: str | None = None
        last_error: str | None = None
        last_category: str | None = None

        for attempt in range(1, max_attempts + 1):
            if attempt == 1:
                prompt = self._build_initial_prompt(request)
            else:
                prompt = self._build_repair_prompt(
                    request=request,
                    broken_code=last_code or "",
                    error_feedback=last_error or "Unknown render error",
                    attempt=attempt,
                )

            assert self.llm_client is not None
            llm_res = await asyncio.to_thread(
                self.llm_client.generate,
                prompt,
                MANIM_SYSTEM_PROMPT,
            )
            if llm_res.is_err():
                last_error = f"Lỗi gọi LLM ở lần thử {attempt}: {llm_res.error}"
                last_category = "LLM_ERROR"
                logger.warning(
                    "manim_generator.llm_failed",
                    attempt=attempt,
                    error=str(llm_res.error),
                )
                continue

            raw_response = llm_res.unwrap()
            code = extract_python_code(raw_response)
            last_code = code

            validation = validate_manim_source_code(code)
            if not validation.valid or not validation.scene_class_name:
                last_error = validation.error
                last_category = validation.error_category
                logger.warning(
                    "manim_generator.validation_failed",
                    attempt=attempt,
                    category=last_category,
                    error=last_error,
                )
                continue

            ok, err_msg, err_cat = await self._render_manim_subprocess(
                code=code,
                scene_class_name=validation.scene_class_name,
                request=request,
                output_mp4_path=output_mp4_path,
            )
            if ok:
                logger.info(
                    "manim_generator.succeeded",
                    tier="AUTO_LLM",
                    attempt=attempt,
                    title=request.title,
                    output=output_mp4_path,
                )
                return ManimGenerationResult(
                    success=True,
                    clip_path=output_mp4_path,
                    source_code=code,
                    tier_used="AUTO_LLM",
                    attempts=attempt,
                )

            last_error = err_msg
            last_category = err_cat
            logger.warning(
                "manim_generator.render_attempt_failed",
                attempt=attempt,
                category=last_category,
                error=(err_msg or "")[:300],
            )
            # Nếu môi trường thiếu gói Manim thì dừng sớm, thử lại cũng vô ích
            if err_cat == "MANIM_UNAVAILABLE":
                break

        return ManimGenerationResult(
            success=False,
            source_code=last_code,
            tier_used="AUTO_LLM",
            attempts=max_attempts,
            error=last_error or "Không thể sinh clip Manim sau số lần thử tối đa",
            error_category=last_category or "UNKNOWN_ERROR",
        )

    def _build_initial_prompt(self, request: ManimGenerationRequest) -> str:
        bullets_str = "\n".join(f"- {b}" for b in request.bullets) if request.bullets else "(Không có)"
        return (
            f"Hãy viết script Manim minh họa trực quan cho cảnh bài giảng sau:\n"
            f"- Tiêu đề cảnh: {request.title}\n"
            f"- Thời lượng audio mục tiêu: {request.target_duration_seconds:.1f} giây\n"
            f"- Các ý chính (bullets):\n{bullets_str}\n"
            f"- Lời giảng (narration):\n{request.narration}\n\n"
            f"Yêu cầu: Trực quan hóa từng bước biến đổi dữ liệu/cấu trúc khớp với lời giảng, "
            f"tổng thời lượng animation khoảng {request.target_duration_seconds:.1f} giây, "
            f"đặt tên lớp là `GeneratedScene`."
        )

    def _build_repair_prompt(
        self,
        *,
        request: ManimGenerationRequest,
        broken_code: str,
        error_feedback: str,
        attempt: int,
    ) -> str:
        return (
            f"Lần thử {attempt - 1} sinh code Manim cho cảnh '{request.title}' đã bị lỗi.\n"
            f"THÔNG BÁO LỖI:\n```\n{error_feedback[:1500]}\n```\n\n"
            f"MÃ NGUỒN BỊ LỖI:\n```python\n{broken_code}\n```\n\n"
            f"Hãy sửa triệt để lỗi trên và trả về toàn bộ file Python hoàn chỉnh "
            f"(lớp `GeneratedScene`, không dùng LaTeX/MathTex, không vi phạm quy tắc VGroup)."
        )

    async def _render_manim_subprocess(
        self,
        *,
        code: str,
        scene_class_name: str,
        request: ManimGenerationRequest,
        output_mp4_path: str,
    ) -> tuple[bool, str | None, str | None]:
        """Chạy `python -m manim` trong thư mục tạm cô lập có timeout cứng."""
        width, height = request.resolution
        with tempfile.TemporaryDirectory(prefix="manim_gen_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            script_path = tmp_path / "generated_scene.py"
            media_dir = tmp_path / "media"
            script_path.write_text(code, encoding="utf-8")

            cmd = [
                self.python_bin,
                "-m",
                "manim",
                "-r",
                f"{width},{height}",
                "--fps",
                str(request.fps),
                "--media_dir",
                str(media_dir),
                "-v",
                "WARNING",
                str(script_path),
                scene_class_name,
            ]

            def _run_sync() -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=tmp_dir,
                    timeout=float(request.timeout_seconds),
                )

            try:
                completed = await asyncio.to_thread(_run_sync)
            except FileNotFoundError as exc:
                return False, f"Không tìm thấy trình thông dịch Python: {exc}", "MANIM_UNAVAILABLE"
            except subprocess.TimeoutExpired:
                return (
                    False,
                    f"Render Manim vượt quá giới hạn thời gian ({request.timeout_seconds}s)",
                    "TIMEOUT",
                )

            stderr_text = (completed.stderr or "").strip()
            stdout_text = (completed.stdout or "").strip()

            if completed.returncode != 0:
                combined = stderr_text or stdout_text or f"Manim exit code {completed.returncode}"
                return False, combined[-1500:], classify_runtime_error(combined)

            # Tìm file MP4 đầu ra (bỏ qua thư mục partial_movie_files)
            candidates = [
                p
                for p in media_dir.rglob("*.mp4")
                if "partial_movie_files" not in p.parts and p.stat().st_size > 0
            ]
            if not candidates:
                return (
                    False,
                    "Manim kết thúc với exit code 0 nhưng không tạo ra file MP4 hợp lệ.",
                    "EMPTY_OUTPUT",
                )

            best_mp4 = max(candidates, key=lambda p: p.stat().st_mtime)
            os.makedirs(os.path.dirname(os.path.abspath(output_mp4_path)), exist_ok=True)
            shutil.copyfile(best_mp4, output_mp4_path)
            return True, None, None
