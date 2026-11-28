"""
Schema câu hỏi quiz mà model phải trả, theo từng kiểu câu, cùng cách dựng đáp án
lưu và kiểm đáp án trước khi gửi về core-api.

Dùng chung bởi ``generate_curriculum_quiz`` (quiz theo LO) và
``generate_micro_quiz`` (quiz của Micro Content).
"""
import re
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, Field, field_validator, model_validator

from document_chunk.application.services.answer_layout import shuffle_choices, shuffle_order
from document_chunk.domain.exceptions import ProcessingError

# Phương án cố định của câu Đúng/Sai, khớp TRUE_FALSE_OPTIONS ở core-api.
TRUE_FALSE_OPTIONS = ("Đúng", "Sai")
_BLANK = re.compile(r"_{3,}")


def _texts(values: list[str], *, label: str) -> list[str]:
    normalized = [v.strip() for v in values if v and v.strip()]
    if len(normalized) != len(values):
        raise ValueError(f"{label} must not contain empty items")
    if len({v.casefold() for v in normalized}) != len(normalized):
        raise ValueError(f"{label} must be distinct")
    return normalized


class QuestionBase(BaseModel):
    question: str
    explanation: str
    difficulty: str = "medium"
    lo_alignment_rationale: str = ""
    source_chunk_ids: list[str] = Field(default_factory=list)

    @field_validator("question", "explanation")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        v = value.strip()
        if not v:
            raise ValueError("value must not be empty")
        return v

    @field_validator("difficulty")
    @classmethod
    def _normalize_difficulty(cls, value: str) -> str:
        v = value.strip().lower()
        if v not in {"easy", "medium", "hard"}:
            return "medium"
        return v


class CurriculumQuestion(QuestionBase):
    """MCQ_SINGLE: 4 phương án, một đáp án đúng."""

    choices: list[str] = Field(min_length=4, max_length=4)
    correct_index: int = Field(ge=0, le=3)

    @field_validator("choices")
    @classmethod
    def _normalize_choices(cls, choices: list[str]) -> list[str]:
        normalized = [c.strip() for c in choices if c and c.strip()]
        if len(normalized) != 4:
            raise ValueError("choices must contain exactly 4 non-empty items")
        return normalized


class MultiAnswerQuestion(QuestionBase):
    """MCQ_MULTI: 4–6 phương án, chọn tất cả đáp án đúng."""

    choices: list[str] = Field(min_length=4, max_length=6)
    correct_indices: list[int] = Field(min_length=1)

    @field_validator("choices")
    @classmethod
    def _normalize_choices(cls, choices: list[str]) -> list[str]:
        return _texts(choices, label="choices")

    @model_validator(mode="after")
    def _check_indices(self) -> "MultiAnswerQuestion":
        indices = set(self.correct_indices)
        if len(indices) != len(self.correct_indices):
            raise ValueError("correct_indices must be distinct")
        if not all(0 <= i < len(self.choices) for i in indices):
            raise ValueError("correct_indices must point at choices")
        if len(indices) == len(self.choices):
            raise ValueError("at least one choice must be wrong")
        return self


class TrueFalseQuestion(QuestionBase):
    """TRUE_FALSE: ``question`` là một mệnh đề."""

    is_true: bool


class FillBlankQuestion(QuestionBase):
    """FILL_BLANK: một chỗ trống ``____``, các cách viết đáp án chấp nhận."""

    accepted_answers: list[str] = Field(min_length=1, max_length=10)

    @field_validator("question")
    @classmethod
    def _one_blank(cls, value: str) -> str:
        if len(_BLANK.findall(value)) != 1:
            raise ValueError("question must contain exactly one blank written as ____")
        return value

    @field_validator("accepted_answers")
    @classmethod
    def _normalize_answers(cls, answers: list[str]) -> list[str]:
        return _texts(answers, label="accepted_answers")


class ShortAnswerQuestion(QuestionBase):
    """SHORT_ANSWER: bài mẫu và các ý chấm; worker chấm bài làm bằng LLM."""

    model_answer: str
    rubric: list[str] = Field(min_length=1, max_length=6)

    @field_validator("model_answer")
    @classmethod
    def _model_answer(cls, value: str) -> str:
        v = value.strip()
        if not v:
            raise ValueError("model_answer must not be empty")
        return v

    @field_validator("rubric")
    @classmethod
    def _normalize_rubric(cls, rubric: list[str]) -> list[str]:
        return _texts(rubric, label="rubric")


_QuestionT = TypeVar("_QuestionT", bound=QuestionBase)


class CurriculumQuizPayload(BaseModel, Generic[_QuestionT]):
    questions: list[_QuestionT] = Field(min_length=1)


# Kiểu câu → (schema một câu, tên schema cho bước sửa JSON).
QUESTION_MODELS: dict[str, tuple[type[QuestionBase], str]] = {
    "MCQ_SINGLE": (CurriculumQuestion, "CurriculumQuestion"),
    "MCQ_MULTI": (MultiAnswerQuestion, "MultiAnswerQuestion"),
    "TRUE_FALSE": (TrueFalseQuestion, "TrueFalseQuestion"),
    "FILL_BLANK": (FillBlankQuestion, "FillBlankQuestion"),
    "SHORT_ANSWER": (ShortAnswerQuestion, "ShortAnswerQuestion"),
}


class QuestionVerdict(BaseModel):
    question_index: int
    verdict: Literal["supported", "contradicted", "not_enough_evidence"]
    reason: str = ""


class QuizVerificationPayload(BaseModel):
    verdicts: list[QuestionVerdict] = Field(min_length=1)


def answer_layout(
    q: QuestionBase,
    question_type: str,
    *,
    seed: str,
    target_index: int,
) -> tuple[list[str], int, object]:
    """
    (phương án lưu, correct_index, correct_answer) theo dạng của từng kiểu câu.

    ``correct_answer`` None với MCQ_SINGLE: repository lấy ``choices[correct_index]``.
    """
    if isinstance(q, CurriculumQuestion):
        # G3: xáo phương án, vị trí đáp án đúng phân bố đều trong mẻ.
        choices, correct_index = shuffle_choices(
            list(q.choices), q.correct_index, seed=seed, target_index=target_index
        )
        return choices, correct_index, None
    if isinstance(q, MultiAnswerQuestion):
        correct = {q.choices[i] for i in q.correct_indices}
        choices = shuffle_order(list(q.choices), seed=seed)
        return choices, 0, [c for c in choices if c in correct]
    if isinstance(q, TrueFalseQuestion):
        answer = TRUE_FALSE_OPTIONS[0 if q.is_true else 1]
        return list(TRUE_FALSE_OPTIONS), TRUE_FALSE_OPTIONS.index(answer), answer
    if isinstance(q, FillBlankQuestion):
        return [], 0, list(q.accepted_answers)
    if isinstance(q, ShortAnswerQuestion):
        return [], 0, {"model_answer": q.model_answer, "rubric": list(q.rubric)}
    raise ProcessingError(f"Unsupported question type: {question_type}")


def _distinct(values: list[str]) -> bool:
    return len({v.strip().casefold() for v in values}) == len(values)


def answer_key_problem(
    question_type: str, question: str, choices: list[str], correct_answer: object
) -> str | None:
    """
    Lý do câu sẽ bị core-api từ chối (``normalizeAnswerKey`` trong
    core-api/src/modules/question/question-types.ts), hoặc None nếu hợp lệ.

    core-api kiểm cả mẻ chặt và từ chối cả kết quả nếu một câu sai dạng; lọc
    trước ở đây để một câu hỏng không làm mất cả mẻ. Schema Pydantic đã bắt gần
    hết; chỗ còn hở là MCQ_SINGLE có phương án trùng nhau.
    """
    if question_type == "MCQ_SINGLE":
        if len(choices) != 4 or not _distinct(choices):
            return "invalid answer key: MCQ_SINGLE needs 4 distinct choices"
        if correct_answer not in choices:
            return "invalid answer key: correct answer is not one of the choices"
    elif question_type == "MCQ_MULTI":
        if not 4 <= len(choices) <= 6 or not _distinct(choices):
            return "invalid answer key: MCQ_MULTI needs 4-6 distinct choices"
        if (
            not isinstance(correct_answer, list)
            or not correct_answer
            or len(correct_answer) >= len(choices)
            or any(c not in choices for c in correct_answer)
        ):
            return "invalid answer key: MCQ_MULTI correct answers must be a strict subset of choices"
    elif question_type == "TRUE_FALSE":
        if correct_answer not in TRUE_FALSE_OPTIONS:
            return "invalid answer key: TRUE_FALSE answer must be Đúng or Sai"
    elif question_type == "FILL_BLANK":
        if not _BLANK.search(question):
            return "invalid answer key: FILL_BLANK question has no ____ blank"
        if (
            not isinstance(correct_answer, list)
            or not 1 <= len(correct_answer) <= 10
            or not _distinct(correct_answer)
        ):
            return "invalid answer key: FILL_BLANK needs 1-10 distinct accepted answers"
    elif question_type == "SHORT_ANSWER":
        if (
            not isinstance(correct_answer, dict)
            or not str(correct_answer.get("model_answer") or "").strip()
            or not isinstance(correct_answer.get("rubric"), list)
            or not 1 <= len(correct_answer["rubric"]) <= 6
            or not _distinct(correct_answer["rubric"])
        ):
            return "invalid answer key: SHORT_ANSWER needs a model answer and 1-6 distinct rubric points"
    return None


__all__ = [
    "CurriculumQuestion",
    "CurriculumQuizPayload",
    "FillBlankQuestion",
    "MultiAnswerQuestion",
    "QUESTION_MODELS",
    "QuestionBase",
    "QuestionVerdict",
    "QuizVerificationPayload",
    "ShortAnswerQuestion",
    "TRUE_FALSE_OPTIONS",
    "TrueFalseQuestion",
    "answer_key_problem",
    "answer_layout",
]
