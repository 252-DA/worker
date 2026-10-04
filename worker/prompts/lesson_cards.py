"""
Prompt sinh micro-content card theo LO (``GenerateLessonCardsUseCase``).

Cùng ba bước như quiz: sinh, kiểm tra với đoạn nguồn, viết lại card bị loại.
"""
import json
from collections.abc import Sequence
from typing import Any, Protocol

from worker.prompts._common import SOURCE_LANGUAGE, Prompt, excerpts, json_system, lo_line

# Dạng JSON của một card: bước sinh trả một danh sách, bước sửa trả một card.
_CARD_JSON = (
    '{"title":"...","key_insight":"...","bullets":["...","..."],'
    '"example":"... or null","misconception":"... or null",'
    '"check_question":"...","check_answer":"...","source_chunk_ids":["..."]}'
)


class Card(Protocol):
    title: str
    key_insight: str
    bullets: list[str]
    example: str | None
    misconception: str | None
    check_question: str
    check_answer: str

    def model_dump(self) -> dict[str, Any]: ...


def generation(
    *,
    course_title: str,
    lo_code: str,
    lo_vi: str,
    lo_en: str | None,
    count: int,
    max_bullets: int,
    context_parts: Sequence[str],
) -> Prompt:
    return Prompt(
        name="card_generation",
        system=json_system(
            "You are an instructional designer writing micro-learning cards for a university course."
        ),
        user=(
            f"You are writing micro-learning cards for course: {course_title}\n"
            f"Target Learning Outcome: {lo_line(lo_code, lo_vi, lo_en)}\n"
            f"{SOURCE_LANGUAGE}\n\n"
            f"Source excerpts (each with chunk_id):\n{excerpts(context_parts)}\n\n"
            f"Write {count} lesson cards that together help a learner reach this LO. "
            "Order them so that earlier cards introduce what later cards build on.\n"
            "Rules:\n"
            "- One idea per card. If a title needs \"and\", split it into two cards.\n"
            "- Cards must not repeat each other.\n"
            "- Every statement must be supported by the source excerpts. Do not add outside facts.\n"
            "- title: a short name for the idea, at most 10 words.\n"
            "- key_insight: the 1-2 sentences a learner must remember if they read nothing else.\n"
            f"- bullets: 2-{max_bullets} short points explaining the idea, at most 25 words each.\n"
            "- example: a concrete example taken from the source, or null if the source has none.\n"
            "- misconception: a common misunderstanding, only if the source states or clearly "
            "implies it; otherwise null.\n"
            "- check_question: one short question a learner answers from memory to test this "
            "card. Not a yes/no question.\n"
            "- check_answer: the correct answer in 1-2 sentences.\n"
            "- source_chunk_ids: the chunk_ids this card is based on.\n"
            "Return strict JSON:\n"
            f'{{"cards":[{_CARD_JSON}]}}'
        ),
    )


def verification(context_parts: Sequence[str], cards: Sequence[Card]) -> Prompt:
    cards_text = "\n\n".join(
        f"[{idx}] Title: {card.title}\n"
        f"Key insight: {card.key_insight}\n"
        f"Bullets: {card.bullets}\n"
        f"Example: {card.example or '(none)'}\n"
        f"Misconception: {card.misconception or '(none)'}\n"
        f"Check question: {card.check_question}\n"
        f"Check answer: {card.check_answer}"
        for idx, card in enumerate(cards)
    )
    return Prompt(
        name="card_verification",
        system=json_system("You are a strict fact-checker for university learning material."),
        user=(
            "You are checking micro-learning cards for factual correctness. "
            "Using ONLY the source excerpts below (not outside knowledge), check "
            "each card.\n\n"
            f"Source excerpts:\n{excerpts(context_parts)}\n\n"
            f"Cards to check:\n{cards_text}\n\n"
            "For each card index, decide:\n"
            '- "supported": every statement in the key insight, bullets, example and '
            "misconception is backed by the source, AND the check answer correctly "
            "answers the check question according to the source.\n"
            '- "contradicted": any statement misstates the source, or the check answer '
            "is wrong.\n"
            '- "not_enough_evidence": the card makes claims the source excerpts do not '
            "cover.\n"
            "Return strict JSON:\n"
            '{"verdicts":[{"card_index":0,'
            '"verdict":"supported|contradicted|not_enough_evidence","reason":"..."}]}'
        ),
    )


def regeneration(context_parts: Sequence[str], original: Card, reason: str) -> Prompt:
    rejected = json.dumps(original.model_dump(), ensure_ascii=False)
    return Prompt(
        name="card_regeneration",
        system=json_system(
            "You are an instructional designer fixing one rejected micro-learning card."
        ),
        user=(
            "A fact-checker rejected the following micro-learning card because it is "
            "not properly grounded in the source excerpts below. Rewrite it so the "
            "problem is fixed. Keep the same idea; remove anything the source does "
            "not support.\n\n"
            f"Source excerpts:\n{excerpts(context_parts)}\n\n"
            f"Rejected card: {rejected}\n"
            f"Fact-checker feedback: {reason}\n\n"
            "Return strict JSON for ONE corrected card:\n"
            f"{_CARD_JSON}"
        ),
    )
