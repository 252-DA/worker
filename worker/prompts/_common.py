"""Phần dùng chung giữa các họ prompt: kiểu ``Prompt`` và vài mảnh câu lặp lại."""
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    """
    Một lần gọi LLM.

    ``name`` đồng thời là ``use_case`` ghi vào ``llm_usage_logs``: mỗi loại lời
    gọi có đúng một prompt, nên chi phí đọc theo use_case cũng là theo prompt.
    """

    name: str
    system: str
    user: str


SOURCE_LANGUAGE = "Use the language of the source content (Vietnamese unless source is English)."


def json_system(role: str) -> str:
    """System prompt: một câu vai trò, rồi yêu cầu chỉ trả JSON."""
    return f"{role} Return strict JSON only."


def lo_line(code: str, statement_vi: str, statement_en: str | None) -> str:
    line = f"{code} — {statement_vi}"
    if statement_en:
        line += f" ({statement_en})"
    return line


def excerpts(context_parts: Sequence[str]) -> str:
    """Các đoạn nguồn (đã kèm chunk_id) nối thành một khối."""
    return "\n\n".join(context_parts) if context_parts else "(no source excerpts available)"
