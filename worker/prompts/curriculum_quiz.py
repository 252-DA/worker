"""
Prompt sinh câu hỏi theo LO (``GenerateCurriculumQuizUseCase``).

Ba bước: sinh một mẻ câu, kiểm tra từng câu với đoạn nguồn, viết lại câu bị loại.
Mỗi mẻ một kiểu câu; phần khác nhau giữa các kiểu nằm trong ``_KINDS``. Dạng
đáp án lưu xuống DB của từng kiểu: core-api ``quiz/question-types.ts``.
"""
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from worker.prompts._common import SOURCE_LANGUAGE, Prompt, excerpts, json_system, lo_line

_STYLES = {
    "quiz": "quick recall / comprehension",
    "midterm": "mixed application and analysis",
    "final": "integrative synthesis across topics",
}

_TAIL_JSON = (
    '"explanation":"...","difficulty":"easy|medium|hard",'
    '"lo_alignment_rationale":"...","source_chunk_ids":["..."]}'
)


@dataclass(frozen=True)
class _Kind:
    """Những gì một kiểu câu thay đổi trong ba prompt."""

    noun: str
    rules: str
    item_json: str
    check_subject: str
    verdicts: str
    render: Callable[[int, Any], str]
    regen_subject: str
    regen_freedom: str
    render_rejected: Callable[[Any], str]


def _rejected_json(q: Any) -> str:
    return f"Rejected question: {json.dumps(q.model_dump(), ensure_ascii=False)}\n"


def _mcq_single(idx: int, q: Any) -> str:
    marked = q.choices[q.correct_index] if 0 <= q.correct_index < len(q.choices) else "(invalid index)"
    return (
        f"[{idx}] Question: {q.question}\n"
        f"Choices: {list(q.choices)}\n"
        f"Marked correct: {marked}\n"
        f"Explanation: {q.explanation}"
    )


def _mcq_multi(idx: int, q: Any) -> str:
    marked = [q.choices[i] for i in q.correct_indices if 0 <= i < len(q.choices)]
    return (
        f"[{idx}] Question: {q.question}\n"
        f"Choices: {list(q.choices)}\n"
        f"Marked correct: {marked}\n"
        f"Explanation: {q.explanation}"
    )


def _true_false(idx: int, q: Any) -> str:
    return (
        f"[{idx}] Statement: {q.question}\n"
        f"Marked: {'true' if q.is_true else 'false'}\n"
        f"Explanation: {q.explanation}"
    )


def _fill_blank(idx: int, q: Any) -> str:
    return (
        f"[{idx}] Sentence: {q.question}\n"
        f"Accepted answers: {list(q.accepted_answers)}\n"
        f"Explanation: {q.explanation}"
    )


def _short_answer(idx: int, q: Any) -> str:
    return (
        f"[{idx}] Question: {q.question}\n"
        f"Model answer: {q.model_answer}\n"
        f"Rubric: {list(q.rubric)}\n"
        f"Explanation: {q.explanation}"
    )


_KINDS: dict[str, _Kind] = {
    "MCQ_SINGLE": _Kind(
        noun="multiple-choice questions",
        rules=(
            "Each question MUST have exactly 4 choices, one correct_index (0-3), "
            "an explanation, and a lo_alignment_rationale.\n"
        ),
        item_json='{"question":"...","choices":["...","...","...","..."],"correct_index":0,' + _TAIL_JSON,
        check_subject="a multiple-choice quiz",
        verdicts=(
            '- "supported": the marked-correct choice and explanation are accurate '
            "per the source, AND the other three choices are clearly wrong (not "
            "arguably correct too).\n"
            '- "contradicted": the marked-correct choice is wrong, the explanation '
            "misstates the source, or another choice is also defensible as correct.\n"
        ),
        render=_mcq_single,
        regen_subject="multiple-choice question",
        regen_freedom="you may change the wording, choices, or correct_index as needed",
        render_rejected=lambda q: (
            f"Rejected question: {q.question}\n"
            f"Rejected choices: {list(q.choices)}\n"
        ),
    ),
    "MCQ_MULTI": _Kind(
        noun="multiple-answer questions (select all that apply)",
        rules=(
            "Each question MUST have 4 or 5 choices and 2 or 3 correct_indices (0-based); "
            "at least one choice must be wrong. Every correct choice must be fully correct "
            "and every other choice clearly wrong according to the source. Phrase the "
            "question so the learner knows to select all correct choices. Include an "
            "explanation that justifies each choice, and a lo_alignment_rationale.\n"
        ),
        item_json='{"question":"...","choices":["...","...","...","..."],"correct_indices":[0,2],' + _TAIL_JSON,
        check_subject="a multiple-answer quiz (select all that apply)",
        verdicts=(
            '- "supported": every marked-correct choice is accurate per the source, '
            "every other choice is clearly wrong, and the explanation is accurate.\n"
            '- "contradicted": a marked-correct choice is wrong, an unmarked choice is '
            "also defensible as correct, or the explanation misstates the source.\n"
        ),
        render=_mcq_multi,
        regen_subject="multiple-answer question",
        regen_freedom="you may change the wording, choices, or correct_indices as needed",
        render_rejected=_rejected_json,
    ),
    "TRUE_FALSE": _Kind(
        noun="true/false statements",
        rules=(
            "Each item MUST be one declarative statement, written in the question field, "
            "that the source makes unambiguously true or false; set is_true accordingly. "
            "Make roughly half of the statements false. A false statement must change one "
            "specific fact the source states, not just add a negation, and must not hinge "
            'on vague words such as "always" or "never". Include an explanation that '
            "cites what the source says, and a lo_alignment_rationale.\n"
        ),
        item_json='{"question":"<one declarative statement>","is_true":true,' + _TAIL_JSON,
        check_subject="a true/false quiz",
        verdicts=(
            '- "supported": the source clearly makes the statement true or false as '
            "marked, the statement is not ambiguous or only partly true, and the "
            "explanation is accurate.\n"
            '- "contradicted": the source gives the statement the opposite truth value, '
            "the statement is ambiguous or only partly true, or the explanation misstates "
            "the source.\n"
        ),
        render=_true_false,
        regen_subject="true/false statement",
        regen_freedom="you may change the wording or is_true as needed",
        render_rejected=_rejected_json,
    ),
    "FILL_BLANK": _Kind(
        noun="fill-in-the-blank questions",
        rules=(
            "Each question MUST be one sentence with exactly one blank written as ____ in "
            "place of a key term. The missing term must be short (1-4 words) and the only "
            "reasonable completion of the sentence. accepted_answers lists every acceptable "
            "form of that term, for example the Vietnamese term and the English term the "
            "source uses, or an abbreviation and its full form. Include an explanation and "
            "a lo_alignment_rationale.\n"
        ),
        item_json='{"question":"... ____ ...","accepted_answers":["...","..."],' + _TAIL_JSON,
        check_subject="a fill-in-the-blank quiz",
        verdicts=(
            '- "supported": the sentence completed with any accepted answer is backed by '
            "the source, AND no reasonable answer outside the accepted list would also fit "
            "the blank.\n"
            '- "contradicted": an accepted answer makes the sentence wrong, or the blank has '
            "another reasonable completion that is not accepted.\n"
        ),
        render=_fill_blank,
        regen_subject="fill-in-the-blank question",
        regen_freedom="you may change the sentence, the blank, or accepted_answers as needed",
        render_rejected=_rejected_json,
    ),
    "SHORT_ANSWER": _Kind(
        noun="short-answer questions",
        rules=(
            "Each question MUST ask the learner to explain, compare or apply an idea in 1-3 "
            "sentences. model_answer is a full-credit answer in 1-3 sentences. rubric lists "
            "2-4 key points a full-credit answer must contain; each point can be checked on "
            "its own and is supported by the source. Include an explanation and a "
            "lo_alignment_rationale.\n"
        ),
        item_json='{"question":"...","model_answer":"...","rubric":["...","..."],' + _TAIL_JSON,
        check_subject="a short-answer quiz",
        verdicts=(
            '- "supported": the model answer is accurate per the source, and every rubric '
            "point is backed by the source and needed for a full-credit answer.\n"
            '- "contradicted": the model answer or a rubric point misstates the source, or '
            "the question cannot be answered from the source.\n"
        ),
        render=_short_answer,
        regen_subject="short-answer question",
        regen_freedom="you may change the wording, model_answer, or rubric as needed",
        render_rejected=_rejected_json,
    ),
}

QUESTION_TYPES = tuple(_KINDS)


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
    question_type: str = "MCQ_SINGLE",
) -> Prompt:
    kind = _KINDS[question_type]
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
            f"Generate {count} {kind.noun} strictly aligned to the LO.\n"
            f"{kind.rules}"
            "Return strict JSON:\n"
            f'{{"questions":[{kind.item_json}]}}'
        ),
    )


def verification(
    context_parts: Sequence[str],
    questions: Sequence[Any],
    question_type: str = "MCQ_SINGLE",
) -> Prompt:
    kind = _KINDS[question_type]
    questions_text = "\n\n".join(kind.render(idx, q) for idx, q in enumerate(questions))
    return Prompt(
        name="quiz_verification",
        system=json_system("You are a strict fact-checker for a university assessment."),
        user=(
            f"You are checking {kind.check_subject} for factual correctness. "
            "Using ONLY the source excerpts below (not outside knowledge), check "
            "each question.\n\n"
            f"Source excerpts:\n{excerpts(context_parts)}\n\n"
            f"Questions to check:\n{questions_text}\n\n"
            "For each question index, decide:\n"
            f"{kind.verdicts}"
            '- "not_enough_evidence": the source excerpts do not cover this enough '
            "to check.\n"
            "Return strict JSON:\n"
            '{"verdicts":[{"question_index":0,'
            '"verdict":"supported|contradicted|not_enough_evidence","reason":"..."}]}'
        ),
    )


def regeneration(
    context_parts: Sequence[str],
    original: Any,
    reason: str,
    question_type: str = "MCQ_SINGLE",
) -> Prompt:
    kind = _KINDS[question_type]
    return Prompt(
        name="quiz_regeneration",
        system=json_system("You are an assessment writer fixing one rejected question."),
        user=(
            f"A fact-checker rejected the following {kind.regen_subject} because "
            "it is not properly grounded in the source excerpts below. Rewrite it so "
            f"the problem is fixed. Keep the same topic and difficulty; {kind.regen_freedom}.\n\n"
            f"Source excerpts:\n{excerpts(context_parts)}\n\n"
            f"{kind.render_rejected(original)}"
            f"Fact-checker feedback: {reason}\n\n"
            "Return strict JSON for ONE corrected question:\n"
            f"{kind.item_json}"
        ),
    )
