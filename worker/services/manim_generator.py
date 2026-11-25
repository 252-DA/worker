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
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from document_chunk.shared.logger import get_logger
from document_chunk.shared.result import Err, Ok, Result

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


class EnvGeminiLLMClient:
    """
    Client gọi trực tiếp Gemini REST API từ biến môi trường GEMINI_API_KEY
    cho MicroContentVideoContainer (không cần kết nối Postgres theo đúng
    ranh giới dữ liệu của video worker).
    """

    def __init__(self, api_key: str, model: str | None = None) -> None:
        self._api_key = api_key.strip()
        self._model = (model or os.getenv("MANIM_LLM_MODEL") or "gemini-2.5-flash").strip()

    def generate(self, prompt: str, system: str | None = None) -> Result[str, Exception]:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent?key={self._api_key}"
        )
        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                candidates = data.get("candidates") or []
                if not candidates:
                    return Err(RuntimeError("Gemini trả về danh sách candidates rỗng"))
                parts = (candidates[0].get("content") or {}).get("parts") or []
                text = "".join(str(p.get("text") or "") for p in parts)
                return Ok(text)
        except Exception as exc:
            return Err(exc)


def build_default_manim_llm_client() -> SimpleLLMClient | None:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    return EnvGeminiLLMClient(api_key=api_key)


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
    timeout_seconds: int = 180
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


def _sanitize_pace(raw: Any, default: float = 1.0) -> float:
    """Hệ số nhịp độ hoạt họa (1.0 = chậm vừa dễ hiểu, >1.0 = rất chậm, <1.0 = nhanh hơn)."""
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return default
    return round(max(0.5, min(2.0, val)), 2)


def render_insertion_sort_template(params: dict[str, Any] | None = None) -> str:
    """Sinh code Manim tham số hóa ĐẦY ĐỦ cho Straight Insertion Sort (nhịp độ chậm, dễ theo dõi)."""
    params = params or {}
    arr = _sanitize_int_array(params.get("array"), default=[23, 78, 45, 8, 32, 56])
    title_text = str(params.get("title") or "Straight Insertion Sort — Array Trace")[:60]
    pace = _sanitize_pace(params.get("pace"), default=1.0)

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
        BG_COMPARE = "#1e3a5f"
        STROKE_COMPARE = "#38bdf8"
        TEXT_WHITE = "#f8fafc"
        TEXT_MUTED = "#94a3b8"

        P = {pace!r}
        values = {arr!r}
        n = len(values)
        box_w = min(1.2, 9.0 / max(n, 1))
        box_h = box_w
        gap = 0.28
        step_x = box_w + gap
        start_x = -((n - 1) * step_x) / 2.0
        base_y = -0.45
        temp_y = base_y + box_h + 0.72

        def slot_pos(idx: int) -> np.ndarray:
            return np.array([start_x + idx * step_x, base_y, 0.0])

        title = Text({title_text!r}, font_size=32, weight=BOLD, color=TEXT_WHITE)
        title.to_edge(UP, buff=0.38)
        if title.width > 12.0:
            title.scale_to_fit_width(12.0)

        def make_banner(txt: str, col: str = "#38bdf8"):
            b = Text(txt, font_size=23, color=col)
            b.next_to(title, DOWN, buff=0.28)
            if b.width > 12.0:
                b.scale_to_fit_width(12.0)
            return b

        banner = make_banner(f"Initial array: {{values}}  (index 0 starts as sorted)")

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

        self.play(FadeIn(title), FadeIn(banner), FadeIn(VGroup(*cells)), FadeIn(idx_labels), run_time=1.2 * P)
        cells[0][0].set_fill(BG_SORTED).set_stroke(STROKE_SORTED, width=3.0)
        self.wait(1.0 * P)

        arr_state = list(values)
        for current in range(1, n):
            key_val = arr_state[current]
            key_cell = cells[current]
            b_pick = make_banner(f"Pass {{current}}: Pick temp = {{key_val}} from index [{{current}}]", STROKE_ACTIVE)

            key_cell[0].set_fill(BG_ACTIVE).set_stroke(STROKE_ACTIVE, width=4.0)
            temp_tag = Text("temp", font_size=18, weight=BOLD, color=STROKE_ACTIVE)
            temp_tag.next_to(np.array([slot_pos(current)[0], temp_y, 0.0]), UP, buff=box_h / 2 + 0.1)
            self.play(
                FadeTransform(banner, b_pick),
                key_cell.animate.move_to(np.array([slot_pos(current)[0], temp_y, 0.0])),
                FadeIn(temp_tag),
                run_time=0.9 * P,
            )
            banner = b_pick
            self.wait(0.65 * P)

            # So sánh & dời từng phần tử lớn hơn sang phải (từng bước để dễ quan sát)
            walker = current - 1
            while walker >= 0 and arr_state[walker] > key_val:
                w_val = arr_state[walker]
                w_cell = cells[walker]
                w_cell[0].set_fill(BG_COMPARE).set_stroke(STROKE_COMPARE, width=3.5)
                b_shift = make_banner(
                    f"Compare {{w_val}} > temp ({{key_val}})  ->  shift {{w_val}} right to [{{walker + 1}}]",
                    STROKE_COMPARE,
                )
                self.play(FadeTransform(banner, b_shift), run_time=0.45 * P)
                banner = b_shift
                self.wait(0.35 * P)

                arr_state[walker + 1] = w_val
                cells[walker + 1] = w_cell
                w_cell[0].set_fill(BG_SORTED).set_stroke(STROKE_SORTED, width=3.0)
                self.play(
                    w_cell.animate.move_to(slot_pos(walker + 1)),
                    key_cell.animate.move_to(np.array([slot_pos(walker)[0], temp_y, 0.0])),
                    temp_tag.animate.next_to(np.array([slot_pos(walker)[0], temp_y, 0.0]), UP, buff=box_h / 2 + 0.1),
                    run_time=0.72 * P,
                )
                self.wait(0.25 * P)
                walker -= 1

            target_idx = walker + 1
            arr_state[target_idx] = key_val
            cells[target_idx] = key_cell

            b_ins = make_banner(f"Insert temp = {{key_val}} at index [{{target_idx}}]  ->  {{arr_state}}", STROKE_SORTED)
            key_cell[0].set_fill(BG_SORTED).set_stroke(STROKE_SORTED, width=3.0)
            self.play(
                FadeTransform(banner, b_ins),
                key_cell.animate.move_to(slot_pos(target_idx)),
                FadeOut(temp_tag),
                run_time=0.85 * P,
            )
            banner = b_ins
            self.wait(0.75 * P)

        b_done = make_banner(f"Sorted Complete! Straight Insertion Sort: {{arr_state}}", STROKE_SORTED)
        self.play(FadeTransform(banner, b_done), run_time=0.65 * P)
        self.play(*[Indicate(c, color=STROKE_SORTED, scale_factor=1.06) for c in cells], run_time=0.75 * P)
        self.wait(1.8 * P)
'''


def render_merge_sort_template(params: dict[str, Any] | None = None) -> str:
    """
    Sinh code Manim tham số hóa ĐẦY ĐỦ TOÀN BỘ THUẬT TOÁN Merge Sort (Divide & Conquer)
    cho mảng đầu vào bất kỳ (3..8 phần tử), nhịp độ sư phạm chậm rãi rõ ràng:
      - Giai đoạn 1 (Divide): Chia đệ quy đầy đủ từng tầng đến các mảng con 1 phần tử (Base Case).
      - Giai đoạn 2 (Bottom-Up Conquer): Trộn thực sự từng cặp mảng con từ dưới lên trên (so sánh & sắp xếp).
      - Giai đoạn 3 (Final Two-Pointer Merge): Minh họa chi tiết 2 con trỏ i, j so sánh từng cặp phần tử
        của nửa trái & nửa phải, di chuyển phần tử nhỏ hơn vào mảng kết quả cho tới khi hoàn tất 100%.
    """
    params = params or {}
    arr = _sanitize_int_array(params.get("array"), default=[38, 27, 43, 3, 9, 82])
    title_text = str(params.get("title") or "Merge Sort — Full Divide & Conquer Trace")[:60]
    pace = _sanitize_pace(params.get("pace"), default=1.0)

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
        BG_ACTIVE = "#78350f"
        STROKE_ACTIVE = "#f59e0b"
        BG_SORTED = "#064e3b"
        STROKE_SORTED = "#10b981"
        TEXT_WHITE = "#f8fafc"
        TEXT_MUTED = "#94a3b8"

        P = {pace!r}
        values = {arr!r}
        n = len(values)
        box_w = 0.78 if n <= 6 else 0.68
        box_h = 0.64 if n <= 6 else 0.58
        inner_gap = 0.06
        font_sz = 23 if n <= 6 else 20

        def make_cell(val, fill_c=BG_DEFAULT, stroke_c=STROKE_DEFAULT, w=box_w, h=box_h, fz=font_sz):
            rect = RoundedRectangle(
                width=w, height=h, corner_radius=0.11,
                fill_color=fill_c, fill_opacity=1.0,
                stroke_color=stroke_c, stroke_width=2.4,
            )
            txt = Text(str(val), font_size=fz, weight=BOLD, color=TEXT_WHITE)
            return VGroup(rect, txt)

        def make_row(vals, center_x, y_pos, fill_c=BG_DEFAULT, stroke_c=STROKE_DEFAULT, w=box_w, h=box_h, gap=inner_gap, fz=font_sz):
            grp = VGroup()
            step = w + gap
            sx = center_x - ((len(vals) - 1) * step) / 2.0
            for i, v in enumerate(vals):
                cell = make_cell(v, fill_c=fill_c, stroke_c=stroke_c, w=w, h=h, fz=fz)
                cell.move_to(np.array([sx + i * step, y_pos, 0.0]))
                grp.add(cell)
            return grp

        def make_banner(text_str, color="#38bdf8"):
            b = Text(text_str, font_size=22, color=color)
            b.next_to(title, DOWN, buff=0.22)
            if b.width > 12.2:
                b.scale_to_fit_width(12.2)
            return b

        def make_pill(text_str, stroke_c=STROKE_DEFAULT, text_c=TEXT_WHITE):
            t = Text(text_str, font_size=20, weight=BOLD, color=text_c)
            if t.width > 11.4:
                t.scale_to_fit_width(11.4)
            bg = RoundedRectangle(
                width=t.width + 0.6, height=0.52, corner_radius=0.12,
                fill_color="#1e293b", fill_opacity=0.95,
                stroke_color=stroke_c, stroke_width=2.0,
            )
            return VGroup(bg, t).to_edge(DOWN, buff=0.28)

        # Xây dựng cây đệ quy Merge Sort thực tế cho mảng bất kỳ (3..8 phần tử)
        def build_tree(sub_vals, depth=0, side="root"):
            node = {{
                "vals": list(sub_vals),
                "sorted_vals": sorted(sub_vals),
                "depth": depth,
                "side": side,
                "left": None,
                "right": None,
            }}
            if len(sub_vals) > 1:
                mid = len(sub_vals) // 2
                node["left"] = build_tree(sub_vals[:mid], depth + 1, "left" if depth == 0 else side)
                node["right"] = build_tree(sub_vals[mid:], depth + 1, "right" if depth == 0 else side)
            return node

        root = build_tree(values)

        def get_max_depth(node):
            if not node["left"] and not node["right"]:
                return node["depth"]
            return max(get_max_depth(node["left"]), get_max_depth(node["right"]))

        max_d = get_max_depth(root)
        # Tọa độ Y theo từng tầng (phủ gọn vùng y in [1.55, -1.85])
        y_top = 1.55
        y_bot = -1.85 if max_d >= 3 else -1.45
        level_y = [
            y_top - (y_top - y_bot) * (d / max(max_d, 1))
            for d in range(max_d + 1)
        ]

        # Gán tọa độ X không bao giờ chồng lấn bằng không gian lá đều nhau
        leaves = []

        def collect_leaves(node):
            if not node["left"] and not node["right"]:
                leaves.append(node)
                return
            collect_leaves(node["left"])
            collect_leaves(node["right"])

        collect_leaves(root)
        leaf_step = min(1.55, 11.2 / max(len(leaves), 1))
        leaf_start_x = -((len(leaves) - 1) * leaf_step) / 2.0
        for idx, lf in enumerate(leaves):
            lf["x"] = leaf_start_x + idx * leaf_step

        def assign_internal_x(node):
            if not node["left"] and not node["right"]:
                return node["x"]
            lx = assign_internal_x(node["left"])
            rx = assign_internal_x(node["right"])
            node["x"] = (lx + rx) / 2.0
            return node["x"]

        assign_internal_x(root)

        # Thu thập các nút theo từng tầng
        nodes_by_depth = {{d: [] for d in range(max_d + 1)}}

        def collect_by_depth(node):
            nodes_by_depth[node["depth"]].append(node)
            if node["left"]:
                collect_by_depth(node["left"])
            if node["right"]:
                collect_by_depth(node["right"])

        collect_by_depth(root)

        # Kiểm tra chống đè ngang giữa các nút anh em cùng tầng
        for d in range(1, max_d + 1):
            row_nodes = nodes_by_depth[d]
            for k in range(1, len(row_nodes)):
                prev_n = row_nodes[k - 1]
                curr_n = row_nodes[k]
                prev_w = len(prev_n["vals"]) * (box_w + inner_gap)
                curr_w = len(curr_n["vals"]) * (box_w + inner_gap)
                min_dist = (prev_w + curr_w) / 2.0 + 0.28
                if curr_n["x"] - prev_n["x"] < min_dist:
                    shift = (min_dist - (curr_n["x"] - prev_n["x"])) / 2.0
                    prev_n["x"] -= shift
                    curr_n["x"] += shift

        # Tiêu đề & Banner
        title = Text({title_text!r}, font_size=30, weight=BOLD, color=TEXT_WHITE)
        title.to_edge(UP, buff=0.32)
        if title.width > 12.2:
            title.scale_to_fit_width(12.2)

        banner = make_banner(f"Step 1 (Divide): Input array (n = {{n}}) -> split recursively to size 1", STROKE_LEFT)
        pill = make_pill(f"Input: {{values}}")

        root["mob"] = make_row(root["vals"], root["x"], level_y[0])
        self.play(FadeIn(title), FadeIn(banner), FadeIn(root["mob"]), FadeIn(pill), run_time=1.3 * P)
        self.wait(1.0 * P)

        # =====================================================================
        # GIAI ĐOẠN 1: CHIA ĐỆ QUY ĐẦY ĐỦ TỪNG TẦNG TỚI BASE CASE (LEN == 1)
        # =====================================================================
        tree_arrows = VGroup()
        for d in range(1, max_d + 1):
            anims = []
            level_groups_str = []
            for parent in nodes_by_depth[d - 1]:
                if not parent["left"] or not parent["right"]:
                    continue
                for child in (parent["left"], parent["right"]):
                    fc = BG_LEFT if child["side"] == "left" else BG_RIGHT
                    sc = STROKE_LEFT if child["side"] == "left" else STROKE_RIGHT
                    child["mob"] = make_row(child["vals"], child["x"], level_y[d], fill_c=fc, stroke_c=sc)
                    arr_edge = Arrow(
                        parent["mob"].get_bottom(),
                        child["mob"].get_top(),
                        color=sc, stroke_width=2.2, max_tip_length_to_length_ratio=0.18, buff=0.08,
                    )
                    tree_arrows.add(arr_edge)
                    anims.extend([GrowArrow(arr_edge), FadeIn(child["mob"], shift=DOWN * 0.15)])
                    level_groups_str.append(str(child["vals"]))

            b_div = make_banner(f"Divide Level {{d}}: " + "  |  ".join(level_groups_str), STROKE_ACTIVE)
            self.play(FadeTransform(banner, b_div), *anims, run_time=1.45 * P)
            banner = b_div
            self.wait(0.95 * P)

        # Đánh dấu toàn bộ các lá đơn (Base Case = đã sắp xếp)
        b_base = make_banner("Base Case reached: Every single-element subarray is trivially sorted!", STROKE_SORTED)
        for lf in leaves:
            lf["mob"][0][0].set_fill(BG_SORTED).set_stroke(STROKE_SORTED, width=2.8)
        self.play(
            FadeTransform(banner, b_base),
            *[Indicate(lf["mob"], color=STROKE_SORTED, scale_factor=1.08) for lf in leaves],
            run_time=1.15 * P,
        )
        banner = b_base
        self.wait(1.0 * P)

        # =====================================================================
        # GIAI ĐOẠN 2: TRỘN TỪNG TẦNG TỪ DƯỚI LÊN (BOTTOM-UP MERGE tới Tầng 1)
        # =====================================================================
        for d in range(max_d - 1, 0, -1):
            merge_targets = [nd for nd in nodes_by_depth[d] if nd["left"] and nd["right"]]
            if not merge_targets:
                continue
            parts_desc = []
            anims = []
            for nd in merge_targets:
                fc = BG_LEFT if nd["side"] == "left" else BG_RIGHT
                sc = STROKE_LEFT if nd["side"] == "left" else STROKE_RIGHT
                sorted_mob = make_row(nd["sorted_vals"], nd["x"], level_y[d], fill_c=fc, stroke_c=sc)
                for c in sorted_mob:
                    c[0].set_stroke(STROKE_SORTED, width=3.0)
                anims.append(ReplacementTransform(nd["mob"], sorted_mob))
                nd["mob"] = sorted_mob
                parts_desc.append(f"{{nd['left']['sorted_vals']}}+{{nd['right']['sorted_vals']}}->{{nd['sorted_vals']}}")

            b_up = make_banner("Bottom-Up Merge (Level " + str(d) + "): " + "  |  ".join(parts_desc), STROKE_SORTED)
            p_up = make_pill("Merged subarrays: " + "  |  ".join(str(nd["sorted_vals"]) for nd in merge_targets), stroke_c=STROKE_SORTED)
            self.play(FadeTransform(banner, b_up), FadeTransform(pill, p_up), *anims, run_time=1.55 * P)
            banner, pill = b_up, p_up
            self.wait(1.15 * P)

        # =====================================================================
        # GIAI ĐOẠN 3: TRỘN CUỐI CÙNG BẰNG 2 CON TRỎ i, j (STEP-BY-STEP MERGE)
        # =====================================================================
        left_half = root["left"]
        right_half = root["right"]
        left_vals = left_half["sorted_vals"]
        right_vals = right_half["sorted_vals"]

        fade_out_mobs = [root["mob"], tree_arrows]
        for d in range(2, max_d + 1):
            for nd in nodes_by_depth[d]:
                fade_out_mobs.append(nd["mob"])

        stage_w = min(1.05, 8.8 / max(n, 1))
        stage_h = 0.86
        stage_gap = 0.16
        stage_fz = 28 if n <= 6 else 24

        left_stage = make_row(left_vals, -3.15, 0.85, fill_c=BG_LEFT, stroke_c=STROKE_LEFT, w=stage_w, h=stage_h, gap=stage_gap, fz=stage_fz)
        right_stage = make_row(right_vals, 3.15, 0.85, fill_c=BG_RIGHT, stroke_c=STROKE_RIGHT, w=stage_w, h=stage_h, gap=stage_gap, fz=stage_fz)
        left_tag = Text("Sorted Left Half", font_size=19, weight=BOLD, color=STROKE_LEFT).next_to(left_stage, UP, buff=0.55)
        right_tag = Text("Sorted Right Half", font_size=19, weight=BOLD, color=STROKE_RIGHT).next_to(right_stage, UP, buff=0.55)

        res_step = stage_w + stage_gap
        res_sx = -((n - 1) * res_step) / 2.0
        res_y = -1.25

        def res_slot_pos(idx):
            return np.array([res_sx + idx * res_step, res_y, 0.0])

        empty_slots = VGroup()
        for k in range(n):
            slot = DashedVMobject(
                RoundedRectangle(width=stage_w, height=stage_h, corner_radius=0.12, stroke_color=TEXT_MUTED, stroke_width=2.0),
                num_dashes=16,
            ).move_to(res_slot_pos(k))
            empty_slots.add(slot)

        b_final_intro = make_banner("Final Merge: Two pointers (i, j) compare front elements & pick smaller", STROKE_SORTED)
        p_final_intro = make_pill(f"Left = {{left_vals}}   vs   Right = {{right_vals}}", stroke_c=STROKE_LEFT)

        self.play(
            *[FadeOut(m) for m in fade_out_mobs],
            ReplacementTransform(left_half["mob"], left_stage),
            ReplacementTransform(right_half["mob"], right_stage),
            FadeIn(left_tag), FadeIn(right_tag), FadeIn(empty_slots),
            FadeTransform(banner, b_final_intro),
            FadeTransform(pill, p_final_intro),
            run_time=1.45 * P,
        )
        banner, pill = b_final_intro, p_final_intro
        self.wait(0.8 * P)

        ptr_i = Text("i", font_size=24, weight=BOLD, color=STROKE_ACTIVE).next_to(left_stage[0], UP, buff=0.12)
        ptr_j = Text("j", font_size=24, weight=BOLD, color="#38bdf8").next_to(right_stage[0], UP, buff=0.12)
        self.play(FadeIn(ptr_i), FadeIn(ptr_j), run_time=0.55 * P)
        self.wait(0.5 * P)

        i_idx, j_idx, k_idx = 0, 0, 0
        merged_cells = []
        merged_nums = []

        while i_idx < len(left_vals) and j_idx < len(right_vals):
            lv, rv = left_vals[i_idx], right_vals[j_idx]
            lc, rc = left_stage[i_idx], right_stage[j_idx]

            lc[0].set_stroke(STROKE_ACTIVE, width=4.0)
            rc[0].set_stroke(STROKE_ACTIVE, width=4.0)

            if lv <= rv:
                b_cmp = make_banner(f"Compare L[{{i_idx}}] = {{lv}}  <=  R[{{j_idx}}] = {{rv}}   ->   pick {{lv}}", STROKE_ACTIVE)
                self.play(FadeTransform(banner, b_cmp), run_time=0.55 * P)
                banner = b_cmp
                self.wait(0.65 * P)

                rc[0].set_stroke(STROKE_RIGHT, width=2.4)
                lc[0].set_fill(BG_SORTED).set_stroke(STROKE_SORTED, width=3.0)
                move_anims = [lc.animate.move_to(res_slot_pos(k_idx))]
                i_idx += 1
                if i_idx < len(left_vals):
                    move_anims.append(ptr_i.animate.next_to(left_stage[i_idx], UP, buff=0.12))
                else:
                    move_anims.append(FadeOut(ptr_i))
                merged_cells.append(lc)
                merged_nums.append(lv)
            else:
                b_cmp = make_banner(f"Compare L[{{i_idx}}] = {{lv}}  >  R[{{j_idx}}] = {{rv}}   ->   pick {{rv}}", STROKE_ACTIVE)
                self.play(FadeTransform(banner, b_cmp), run_time=0.55 * P)
                banner = b_cmp
                self.wait(0.65 * P)

                lc[0].set_stroke(STROKE_LEFT, width=2.4)
                rc[0].set_fill(BG_SORTED).set_stroke(STROKE_SORTED, width=3.0)
                move_anims = [rc.animate.move_to(res_slot_pos(k_idx))]
                j_idx += 1
                if j_idx < len(right_vals):
                    move_anims.append(ptr_j.animate.next_to(right_stage[j_idx], UP, buff=0.12))
                else:
                    move_anims.append(FadeOut(ptr_j))
                merged_cells.append(rc)
                merged_nums.append(rv)

            p_step = make_pill(f"Merged [{{k_idx + 1}}/{{n}}]: {{merged_nums}}", stroke_c=STROKE_SORTED)
            self.play(*move_anims, FadeTransform(pill, p_step), run_time=0.95 * P)
            pill = p_step
            self.wait(0.45 * P)
            k_idx += 1

        # Sao chép các phần tử còn sót lại ở nửa trái hoặc nửa phải
        while i_idx < len(left_vals):
            lv = left_vals[i_idx]
            lc = left_stage[i_idx]
            b_rem = make_banner(f"Right exhausted -> copy remaining Left element {{lv}}", STROKE_SORTED)
            lc[0].set_fill(BG_SORTED).set_stroke(STROKE_SORTED, width=3.0)
            move_anims = [FadeTransform(banner, b_rem), lc.animate.move_to(res_slot_pos(k_idx))]
            i_idx += 1
            if i_idx < len(left_vals):
                move_anims.append(ptr_i.animate.next_to(left_stage[i_idx], UP, buff=0.12))
            else:
                move_anims.append(FadeOut(ptr_i))
            merged_cells.append(lc)
            merged_nums.append(lv)
            p_step = make_pill(f"Merged [{{k_idx + 1}}/{{n}}]: {{merged_nums}}", stroke_c=STROKE_SORTED)
            self.play(*move_anims, FadeTransform(pill, p_step), run_time=0.9 * P)
            banner, pill = b_rem, p_step
            self.wait(0.4 * P)
            k_idx += 1

        while j_idx < len(right_vals):
            rv = right_vals[j_idx]
            rc = right_stage[j_idx]
            b_rem = make_banner(f"Left exhausted -> copy remaining Right element {{rv}}", STROKE_SORTED)
            rc[0].set_fill(BG_SORTED).set_stroke(STROKE_SORTED, width=3.0)
            move_anims = [FadeTransform(banner, b_rem), rc.animate.move_to(res_slot_pos(k_idx))]
            j_idx += 1
            if j_idx < len(right_vals):
                move_anims.append(ptr_j.animate.next_to(right_stage[j_idx], UP, buff=0.12))
            else:
                move_anims.append(FadeOut(ptr_j))
            merged_cells.append(rc)
            merged_nums.append(rv)
            p_step = make_pill(f"Merged [{{k_idx + 1}}/{{n}}]: {{merged_nums}}", stroke_c=STROKE_SORTED)
            self.play(*move_anims, FadeTransform(pill, p_step), run_time=0.9 * P)
            banner, pill = b_rem, p_step
            self.wait(0.4 * P)
            k_idx += 1

        # Hoàn tất toàn bộ mảng đã sắp xếp
        b_done = make_banner("Sorted Complete! Merge Sort guarantees O(n log n) in all cases", STROKE_SORTED)
        p_done = make_pill(f"Sorted: {{merged_nums}}", stroke_c=STROKE_SORTED, text_c="#bbf7d0")
        self.play(
            FadeOut(empty_slots), FadeOut(left_tag), FadeOut(right_tag),
            FadeTransform(banner, b_done),
            FadeTransform(pill, p_done),
            run_time=0.8 * P,
        )
        self.play(*[Indicate(c, color=STROKE_SORTED, scale_factor=1.06) for c in merged_cells], run_time=0.8 * P)
        self.wait(2.0 * P)
'''


def render_selection_sort_template(params: dict[str, Any] | None = None) -> str:
    """Sinh code Manim tham số hóa ĐẦY ĐỦ cho Selection Sort (nhịp độ chậm, có nhãn min & hoán đổi vòng cung)."""
    params = params or {}
    arr = _sanitize_int_array(params.get("array"), default=[29, 10, 14, 37, 13, 8])
    title_text = str(params.get("title") or "Selection Sort — Full Algorithm Trace")[:60]
    pace = _sanitize_pace(params.get("pace"), default=1.0)

    return f'''from manim import *
import numpy as np


class GeneratedScene(Scene):
    def construct(self):
        self.camera.background_color = "#0f172a"
        BG_DEFAULT = "#1e293b"
        STROKE_DEFAULT = "#64748b"
        BG_SCAN = "#1e3a5f"
        STROKE_SCAN = "#38bdf8"
        BG_MIN = "#78350f"
        STROKE_MIN = "#f59e0b"
        BG_SORTED = "#064e3b"
        STROKE_SORTED = "#10b981"
        TEXT_WHITE = "#f8fafc"
        TEXT_MUTED = "#94a3b8"

        P = {pace!r}
        values = {arr!r}
        n = len(values)
        box_w = min(1.15, 8.8 / max(n, 1))
        box_h = box_w
        gap = 0.25
        step_x = box_w + gap
        start_x = -((n - 1) * step_x) / 2.0
        base_y = -0.2

        def slot_pos(idx):
            return np.array([start_x + idx * step_x, base_y, 0.0])

        title = Text({title_text!r}, font_size=30, weight=BOLD, color=TEXT_WHITE).to_edge(UP, buff=0.35)
        if title.width > 12.2:
            title.scale_to_fit_width(12.2)

        def make_banner(txt, col=STROKE_SCAN):
            b = Text(txt, font_size=22, color=col).next_to(title, DOWN, buff=0.25)
            if b.width > 12.2:
                b.scale_to_fit_width(12.2)
            return b

        cells = []
        idx_labels = VGroup()
        for i, v in enumerate(values):
            rect = RoundedRectangle(width=box_w, height=box_h, corner_radius=0.14, fill_color=BG_DEFAULT, fill_opacity=1.0, stroke_color=STROKE_DEFAULT, stroke_width=2.5)
            num = Text(str(v), font_size=28, weight=BOLD, color=TEXT_WHITE)
            cells.append(VGroup(rect, num).move_to(slot_pos(i)))
            idx_labels.add(Text(str(i), font_size=18, color=TEXT_MUTED).next_to(slot_pos(i), DOWN, buff=box_h / 2 + 0.2))

        banner = make_banner(f"Input array: {{values}} -> find min in unsorted region each pass")
        self.play(FadeIn(title), FadeIn(banner), FadeIn(VGroup(*cells)), FadeIn(idx_labels), run_time=1.2 * P)
        self.wait(1.0 * P)

        arr_state = list(values)
        for i in range(n - 1):
            min_idx = i
            cells[min_idx][0].set_fill(BG_MIN).set_stroke(STROKE_MIN, width=3.8)
            min_tag = Text("min", font_size=18, weight=BOLD, color=STROKE_MIN).next_to(slot_pos(min_idx), UP, buff=box_h / 2 + 0.16)
            b_pass = make_banner(f"Pass {{i + 1}}: Scan unsorted [{{i}}..{{n - 1}}] -> initial min = {{arr_state[min_idx]}}", STROKE_MIN)
            self.play(FadeTransform(banner, b_pass), FadeIn(min_tag), run_time=0.65 * P)
            banner = b_pass
            self.wait(0.55 * P)

            for j in range(i + 1, n):
                cells[j][0].set_fill(BG_SCAN).set_stroke(STROKE_SCAN, width=3.4)
                self.wait(0.32 * P)
                if arr_state[j] < arr_state[min_idx]:
                    cells[min_idx][0].set_fill(BG_DEFAULT).set_stroke(STROKE_DEFAULT, width=2.5)
                    min_idx = j
                    cells[min_idx][0].set_fill(BG_MIN).set_stroke(STROKE_MIN, width=3.8)
                    b_new = make_banner(f"Pass {{i + 1}}: Found smaller min = {{arr_state[min_idx]}} at index [{{min_idx}}]", STROKE_MIN)
                    self.play(
                        FadeTransform(banner, b_new),
                        min_tag.animate.next_to(slot_pos(min_idx), UP, buff=box_h / 2 + 0.16),
                        Indicate(cells[min_idx], color=STROKE_MIN),
                        run_time=0.65 * P,
                    )
                    banner = b_new
                    self.wait(0.4 * P)
                else:
                    cells[j][0].set_fill(BG_DEFAULT).set_stroke(STROKE_DEFAULT, width=2.5)

            if min_idx != i:
                b_sw = make_banner(f"Swap index [{{i}}] ({{arr_state[i]}}) <-> min [{{min_idx}}] ({{arr_state[min_idx]}})", STROKE_SORTED)
                arr_state[i], arr_state[min_idx] = arr_state[min_idx], arr_state[i]
                cells[i], cells[min_idx] = cells[min_idx], cells[i]
                cells[i][0].set_fill(BG_SORTED).set_stroke(STROKE_SORTED, width=3.0)
                self.play(
                    FadeTransform(banner, b_sw),
                    FadeOut(min_tag),
                    cells[i].animate(path_arc=-0.8).move_to(slot_pos(i)),
                    cells[min_idx].animate(path_arc=-0.8).move_to(slot_pos(min_idx)),
                    run_time=1.05 * P,
                )
                banner = b_sw
            else:
                cells[i][0].set_fill(BG_SORTED).set_stroke(STROKE_SORTED, width=3.0)
                b_keep = make_banner(f"Index [{{i}}] ({{arr_state[i]}}) is already minimum -> lock sorted", STROKE_SORTED)
                self.play(FadeTransform(banner, b_keep), FadeOut(min_tag), run_time=0.65 * P)
                banner = b_keep
            self.wait(0.75 * P)

        cells[n - 1][0].set_fill(BG_SORTED).set_stroke(STROKE_SORTED, width=3.0)
        b_done = make_banner(f"Sorted Complete — O(n^2) comparisons, O(n) swaps: {{arr_state}}", STROKE_SORTED)
        self.play(FadeTransform(banner, b_done), run_time=0.7 * P)
        self.play(*[Indicate(c, color=STROKE_SORTED, scale_factor=1.06) for c in cells], run_time=0.75 * P)
        self.wait(1.8 * P)
'''


def render_bubble_sort_template(params: dict[str, Any] | None = None) -> str:
    """Sinh code Manim tham số hóa ĐẦY ĐỦ cho Bubble Sort (nhịp độ chậm, rõ từng cặp so sánh & hoán đổi)."""
    params = params or {}
    arr = _sanitize_int_array(params.get("array"), default=[52, 18, 41, 9, 33, 24])
    title_text = str(params.get("title") or "Bubble Sort — Full Algorithm Trace")[:60]
    pace = _sanitize_pace(params.get("pace"), default=1.0)

    return f'''from manim import *
import numpy as np


class GeneratedScene(Scene):
    def construct(self):
        self.camera.background_color = "#0f172a"
        BG_DEFAULT = "#1e293b"
        STROKE_DEFAULT = "#64748b"
        BG_ACTIVE = "#78350f"
        STROKE_ACTIVE = "#f59e0b"
        BG_SORTED = "#064e3b"
        STROKE_SORTED = "#10b981"
        TEXT_WHITE = "#f8fafc"
        TEXT_MUTED = "#94a3b8"

        P = {pace!r}
        values = {arr!r}
        n = len(values)
        box_w = min(1.15, 8.8 / max(n, 1))
        box_h = box_w
        gap = 0.25
        step_x = box_w + gap
        start_x = -((n - 1) * step_x) / 2.0
        base_y = -0.2

        def slot_pos(idx):
            return np.array([start_x + idx * step_x, base_y, 0.0])

        title = Text({title_text!r}, font_size=30, weight=BOLD, color=TEXT_WHITE).to_edge(UP, buff=0.35)
        if title.width > 12.2:
            title.scale_to_fit_width(12.2)

        def make_banner(txt, col="#38bdf8"):
            b = Text(txt, font_size=22, color=col).next_to(title, DOWN, buff=0.25)
            if b.width > 12.2:
                b.scale_to_fit_width(12.2)
            return b

        cells = []
        idx_labels = VGroup()
        for i, v in enumerate(values):
            rect = RoundedRectangle(width=box_w, height=box_h, corner_radius=0.14, fill_color=BG_DEFAULT, fill_opacity=1.0, stroke_color=STROKE_DEFAULT, stroke_width=2.5)
            num = Text(str(v), font_size=28, weight=BOLD, color=TEXT_WHITE)
            cells.append(VGroup(rect, num).move_to(slot_pos(i)))
            idx_labels.add(Text(str(i), font_size=18, color=TEXT_MUTED).next_to(slot_pos(i), DOWN, buff=box_h / 2 + 0.2))

        banner = make_banner(f"Input array: {{values}} -> bubble largest element to the right")
        self.play(FadeIn(title), FadeIn(banner), FadeIn(VGroup(*cells)), FadeIn(idx_labels), run_time=1.2 * P)
        self.wait(1.0 * P)

        arr_state = list(values)
        for p in range(n - 1):
            swapped = False
            b_p = make_banner(f"Pass {{p + 1}}: Compare adjacent pairs in [0..{{n - 1 - p}}]", STROKE_ACTIVE)
            self.play(FadeTransform(banner, b_p), run_time=0.6 * P)
            banner = b_p
            self.wait(0.55 * P)

            for j in range(n - 1 - p):
                cells[j][0].set_fill(BG_ACTIVE).set_stroke(STROKE_ACTIVE, width=3.5)
                cells[j + 1][0].set_fill(BG_ACTIVE).set_stroke(STROKE_ACTIVE, width=3.5)
                self.wait(0.45 * P)
                if arr_state[j] > arr_state[j + 1]:
                    swapped = True
                    arr_state[j], arr_state[j + 1] = arr_state[j + 1], arr_state[j]
                    cells[j], cells[j + 1] = cells[j + 1], cells[j]
                    self.play(
                        cells[j].animate(path_arc=-0.65).move_to(slot_pos(j)),
                        cells[j + 1].animate(path_arc=-0.65).move_to(slot_pos(j + 1)),
                        run_time=0.72 * P,
                    )
                    self.wait(0.28 * P)
                else:
                    self.wait(0.32 * P)
                cells[j][0].set_fill(BG_DEFAULT).set_stroke(STROKE_DEFAULT, width=2.5)
                cells[j + 1][0].set_fill(BG_DEFAULT).set_stroke(STROKE_DEFAULT, width=2.5)

            locked = n - 1 - p
            cells[locked][0].set_fill(BG_SORTED).set_stroke(STROKE_SORTED, width=3.0)
            self.play(Indicate(cells[locked], color=STROKE_SORTED, scale_factor=1.08), run_time=0.55 * P)
            self.wait(0.45 * P)
            if not swapped:
                for k in range(locked):
                    cells[k][0].set_fill(BG_SORTED).set_stroke(STROKE_SORTED, width=3.0)
                break

        cells[0][0].set_fill(BG_SORTED).set_stroke(STROKE_SORTED, width=3.0)
        b_done = make_banner(f"Sorted Complete — Bubble Sort: {{arr_state}}", STROKE_SORTED)
        self.play(FadeTransform(banner, b_done), run_time=0.65 * P)
        self.play(*[Indicate(c, color=STROKE_SORTED, scale_factor=1.06) for c in cells], run_time=0.75 * P)
        self.wait(1.8 * P)
'''


def render_quicksort_template(params: dict[str, Any] | None = None) -> str:
    """Sinh code Manim tham số hóa ĐẦY ĐỦ cho QuickSort (Lomuto Partition, chậm rãi, nhãn pivot & swap vòng cung)."""
    params = params or {}
    arr = _sanitize_int_array(params.get("array"), default=[38, 12, 45, 7, 29, 21])
    title_text = str(params.get("title") or "QuickSort (Lomuto) — Full Algorithm Trace")[:60]
    pace = _sanitize_pace(params.get("pace"), default=1.0)

    return f'''from manim import *
import numpy as np


class GeneratedScene(Scene):
    def construct(self):
        self.camera.background_color = "#0f172a"
        BG_DEFAULT = "#1e293b"
        STROKE_DEFAULT = "#64748b"
        BG_PIVOT = "#78350f"
        STROKE_PIVOT = "#f59e0b"
        BG_LE = "#1e3a5f"
        STROKE_LE = "#38bdf8"
        BG_SORTED = "#064e3b"
        STROKE_SORTED = "#10b981"
        TEXT_WHITE = "#f8fafc"
        TEXT_MUTED = "#94a3b8"

        P = {pace!r}
        values = {arr!r}
        n = len(values)
        box_w = min(1.15, 8.8 / max(n, 1))
        box_h = box_w
        gap = 0.25
        step_x = box_w + gap
        start_x = -((n - 1) * step_x) / 2.0
        base_y = -0.2

        def slot_pos(idx):
            return np.array([start_x + idx * step_x, base_y, 0.0])

        title = Text({title_text!r}, font_size=30, weight=BOLD, color=TEXT_WHITE).to_edge(UP, buff=0.35)
        if title.width > 12.2:
            title.scale_to_fit_width(12.2)

        def make_banner(txt, col=STROKE_LE):
            b = Text(txt, font_size=22, color=col).next_to(title, DOWN, buff=0.25)
            if b.width > 12.2:
                b.scale_to_fit_width(12.2)
            return b

        cells = []
        idx_labels = VGroup()
        for i, v in enumerate(values):
            rect = RoundedRectangle(width=box_w, height=box_h, corner_radius=0.14, fill_color=BG_DEFAULT, fill_opacity=1.0, stroke_color=STROKE_DEFAULT, stroke_width=2.5)
            num = Text(str(v), font_size=28, weight=BOLD, color=TEXT_WHITE)
            cells.append(VGroup(rect, num).move_to(slot_pos(i)))
            idx_labels.add(Text(str(i), font_size=18, color=TEXT_MUTED).next_to(slot_pos(i), DOWN, buff=box_h / 2 + 0.2))

        banner = make_banner(f"Input array: {{values}} -> recursive Lomuto Partition")
        self.play(FadeIn(title), FadeIn(banner), FadeIn(VGroup(*cells)), FadeIn(idx_labels), run_time=1.2 * P)
        self.wait(1.0 * P)

        arr_state = list(values)
        locked = set()

        def qsort(lo, hi):
            nonlocal banner
            if lo > hi:
                return
            if lo == hi:
                locked.add(lo)
                cells[lo][0].set_fill(BG_SORTED).set_stroke(STROKE_SORTED, width=3.0)
                self.play(Indicate(cells[lo], color=STROKE_SORTED, scale_factor=1.08), run_time=0.5 * P)
                self.wait(0.35 * P)
                return

            pivot_val = arr_state[hi]
            cells[hi][0].set_fill(BG_PIVOT).set_stroke(STROKE_PIVOT, width=4.0)
            pivot_tag = Text("pivot", font_size=18, weight=BOLD, color=STROKE_PIVOT).next_to(slot_pos(hi), UP, buff=box_h / 2 + 0.16)
            b_pv = make_banner(f"Partition [{{lo}}..{{hi}}]: Choose pivot = {{pivot_val}} (index {{hi}})", STROKE_PIVOT)
            self.play(FadeTransform(banner, b_pv), FadeIn(pivot_tag), run_time=0.7 * P)
            banner = b_pv
            self.wait(0.75 * P)

            i = lo - 1
            for j in range(lo, hi):
                cells[j][0].set_stroke(STROKE_PIVOT, width=3.4)
                self.wait(0.4 * P)
                if arr_state[j] <= pivot_val:
                    i += 1
                    cells[j][0].set_fill(BG_LE).set_stroke(STROKE_LE, width=3.2)
                    if i != j:
                        arr_state[i], arr_state[j] = arr_state[j], arr_state[i]
                        cells[i], cells[j] = cells[j], cells[i]
                        self.play(
                            cells[i].animate(path_arc=-0.8).move_to(slot_pos(i)),
                            cells[j].animate(path_arc=-0.8).move_to(slot_pos(j)),
                            run_time=0.85 * P,
                        )
                        self.wait(0.35 * P)
                    else:
                        self.wait(0.4 * P)
                else:
                    cells[j][0].set_stroke(STROKE_DEFAULT, width=2.5)
                    self.wait(0.25 * P)

            p_idx = i + 1
            if p_idx != hi:
                arr_state[p_idx], arr_state[hi] = arr_state[hi], arr_state[p_idx]
                cells[p_idx], cells[hi] = cells[hi], cells[p_idx]
                self.play(
                    cells[p_idx].animate(path_arc=-0.8).move_to(slot_pos(p_idx)),
                    cells[hi].animate(path_arc=-0.8).move_to(slot_pos(hi)),
                    pivot_tag.animate.next_to(slot_pos(p_idx), UP, buff=box_h / 2 + 0.16),
                    run_time=0.95 * P,
                )
                self.wait(0.35 * P)

            locked.add(p_idx)
            for k in range(lo, hi + 1):
                if k in locked:
                    cells[k][0].set_fill(BG_SORTED).set_stroke(STROKE_SORTED, width=3.0)
                else:
                    cells[k][0].set_fill(BG_DEFAULT).set_stroke(STROKE_DEFAULT, width=2.5)

            b_pl = make_banner(f"Pivot {{pivot_val}} placed at final index [{{p_idx}}]  ->  {{arr_state}}", STROKE_SORTED)
            self.play(FadeTransform(banner, b_pl), FadeOut(pivot_tag), run_time=0.65 * P)
            banner = b_pl
            self.wait(0.9 * P)

            qsort(lo, p_idx - 1)
            qsort(p_idx + 1, hi)

        qsort(0, n - 1)
        for k in range(n):
            cells[k][0].set_fill(BG_SORTED).set_stroke(STROKE_SORTED, width=3.0)

        b_done = make_banner(f"Sorted Complete — QuickSort O(n log n) avg: {{arr_state}}", STROKE_SORTED)
        self.play(FadeTransform(banner, b_done), run_time=0.65 * P)
        self.play(*[Indicate(c, color=STROKE_SORTED, scale_factor=1.06) for c in cells], run_time=0.75 * P)
        self.wait(1.8 * P)
'''


TEMPLATE_REGISTRY: dict[str, Any] = {
    "insertion_sort": render_insertion_sort_template,
    "straight_insertion_sort": render_insertion_sort_template,
    "merge_sort": render_merge_sort_template,
    "selection_sort": render_selection_sort_template,
    "bubble_sort": render_bubble_sort_template,
    "quicksort": render_quicksort_template,
    "quick_sort": render_quicksort_template,
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
