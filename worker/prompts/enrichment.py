"""
Prompt sinh card và quiz theo từng section tài liệu (``RunEnrichmentUseCase``).

Đường cũ, chạy mỗi lần ingest: không nhắm LO và không có bước kiểm tra. Câu
chữ ở đây vì thế khác hai họ prompt theo LO.
"""
from typing import Protocol

from worker.prompts._common import Prompt, json_system


class Section(Protocol):
    heading_path: tuple[str, ...]
    title: str
    language: str | None
    content_text: str


def _section_block(section: Section) -> str:
    return (
        f"Heading path: {' > '.join(section.heading_path)}\n"
        f"Section title: {section.title}\n"
        f"Language: {section.language or 'same as source'}\n\n"
        "Section content:\n"
        f"{section.content_text}"
    )


def cards(section: Section) -> Prompt:
    return Prompt(
        name="enrichment_cards",
        system=json_system(
            "You are a teaching assistant that creates concise micro-learning lesson cards."
        ),
        user=(
            "Create 2-3 micro-learning lesson cards from the section below.\n"
            "Use the same language as the source content.\n"
            "Return strict JSON only with this shape:\n"
            '{"cards":[{"title":"...","bullets":["..."],"key_insight":"..."}]}\n\n'
            f"{_section_block(section)}"
        ),
    )


def quiz(section: Section) -> Prompt:
    return Prompt(
        name="enrichment_quiz",
        system=json_system(
            "You are an assessment writer that creates multiple-choice questions from academic content."
        ),
        user=(
            "Create 3-5 multiple-choice questions from the section below.\n"
            "Each question must have exactly 4 choices, one correct answer index, and an explanation.\n"
            "Use the same language as the source content.\n"
            "Return strict JSON only with this shape:\n"
            '{"questions":[{"question":"...","choices":["...","...","...","..."],'
            '"correct_index":0,"explanation":"...","difficulty":"easy|medium|hard"}]}\n\n'
            f"{_section_block(section)}"
        ),
    )
