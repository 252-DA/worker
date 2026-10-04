"""
Prompt sinh câu trắc nghiệm theo LO (``GenerateCurriculumQuizUseCase``).

Ba bước: sinh một mẻ câu, kiểm tra từng câu với đoạn nguồn, viết lại câu bị loại.
"""
from collections.abc import Sequence
from typing import Protocol

from worker.prompts._common import SOURCE_LANGUAGE, Prompt, excerpts, json_system, lo_line

_STYLES = {
    "quiz": "quick recall / comprehension",
    "midterm": "mixed application and analysis",
    "final": "integrative synthesis across topics",
}

# Dạng JSON của một câu: bước sinh trả một danh sách, bước sửa trả một câu.
_QUESTION_JSON = (
    '{"question":"...","choices":["...","...","...","..."],'
    '"correct_index":0,"explanation":"...","difficulty":"easy|medium|hard",'
    '"lo_alignment_rationale":"...","source_chunk_ids":["..."]}'
)


class Question(Protocol):
    question: str
    choices: list[str]
    correct_index: int
    explanation: str


def generation(
    *,
    course_title: str,
    lo_code: str,
    lo_vi: str,
    lo_en: str | None,
    bloom: str,
    style: str,
    count: int,
    context_parts: Sequence[str],
) -> Prompt:
    style_desc = _STYLES.get(style, style)
    return Prompt(
        name="quiz_generation",
        system=json_system("You are an assessment writer for a university course."),
        user=(
            f"You are creating an assessment item for course: {course_title}\n"
            f"Target Learning Outcome: {lo_line(lo_code, lo_vi, lo_en)}\n"
            f"Bloom level required: {bloom}\n"
            f"Assessment style: {style} ({style_desc})\n"
            f"{SOURCE_LANGUAGE}\n\n"
            f"Source excerpts (each with chunk_id):\n{excerpts(context_parts)}\n\n"
            f"Generate {count} multiple-choice questions strictly aligned to the LO.\n"
            "Each question MUST have exactly 4 choices, one correct_index (0-3), "
            "an explanation, and a lo_alignment_rationale.\n"
            "Return strict JSON:\n"
            f'{{"questions":[{_QUESTION_JSON}]}}'
        ),
    )


def verification(context_parts: Sequence[str], questions: Sequence[Question]) -> Prompt:
    questions_text = "\n\n".join(
        f"[{idx}] Question: {q.question}\n"
        f"Choices: {list(q.choices)}\n"
        "Marked correct: "
        f"{q.choices[q.correct_index] if 0 <= q.correct_index < len(q.choices) else '(invalid index)'}\n"
        f"Explanation: {q.explanation}"
        for idx, q in enumerate(questions)
    )
    return Prompt(
        name="quiz_verification",
        system=json_system("You are a strict fact-checker for a university assessment."),
        user=(
            "You are checking a multiple-choice quiz for factual correctness. "
            "Using ONLY the source excerpts below (not outside knowledge), check "
            "each question.\n\n"
            f"Source excerpts:\n{excerpts(context_parts)}\n\n"
            f"Questions to check:\n{questions_text}\n\n"
            "For each question index, decide:\n"
            '- "supported": the marked-correct choice and explanation are accurate '
            "per the source, AND the other three choices are clearly wrong (not "
            "arguably correct too).\n"
            '- "contradicted": the marked-correct choice is wrong, the explanation '
            "misstates the source, or another choice is also defensible as correct.\n"
            '- "not_enough_evidence": the source excerpts do not cover this enough '
            "to check.\n"
            "Return strict JSON:\n"
            '{"verdicts":[{"question_index":0,'
            '"verdict":"supported|contradicted|not_enough_evidence","reason":"..."}]}'
        ),
    )


def regeneration(context_parts: Sequence[str], original: Question, reason: str) -> Prompt:
    return Prompt(
        name="quiz_regeneration",
        system=json_system("You are an assessment writer fixing one rejected question."),
        user=(
            "A fact-checker rejected the following multiple-choice question because "
            "it is not properly grounded in the source excerpts below. Rewrite it so "
            "the problem is fixed. Keep the same topic and difficulty; you may change "
            "the wording, choices, or correct_index as needed.\n\n"
            f"Source excerpts:\n{excerpts(context_parts)}\n\n"
            f"Rejected question: {original.question}\n"
            f"Rejected choices: {list(original.choices)}\n"
            f"Fact-checker feedback: {reason}\n\n"
            "Return strict JSON for ONE corrected question:\n"
            f"{_QUESTION_JSON}"
        ),
    )
