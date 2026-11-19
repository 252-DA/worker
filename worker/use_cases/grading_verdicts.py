"""
Kiểm chỉ số model trả khi chấm theo rubric (tự luận quiz, Micro Content, bài luận).

Model chấm theo ``index`` của từng ý/tiêu chí trong prompt (đếm từ 0). Nó hay đếm
từ 1, trả trùng hoặc bịa chỉ số. Bỏ qua chỉ số lạ rồi cộng điểm sẽ cho ra điểm
lệch hoặc 0 mà không ai biết: cam kết của hệ thống là lỗi định dạng của model
không bao giờ thành điểm. Nên chỉ số sai là ``MalformedGrade``: người gọi coi như
model lỗi (thử lại, lượt cuối thì giao giảng viên), không chấm bằng nửa kết quả.
"""
from collections.abc import Iterable

from document_chunk.domain.exceptions import ProcessingError


class MalformedGrade(ProcessingError):
    """Model trả JSON đúng schema nhưng chỉ số ý/tiêu chí không khớp rubric."""


def check_verdict_indices(
    indices: Iterable[int], rubric_size: int, *, require_all: bool, what: str
) -> None:
    """
    Ném ``MalformedGrade`` nếu chỉ số ngoài [0, rubric_size) hoặc trùng nhau;
    ``require_all`` thì thiếu chỉ số nào cũng lỗi. ``what`` ("tiêu chí", "ý") chỉ
    để câu báo lỗi dễ đọc.
    """
    seen: set[int] = set()
    out_of_range: list[int] = []
    duplicated: list[int] = []
    for index in indices:
        if not 0 <= index < rubric_size:
            out_of_range.append(index)
        elif index in seen:
            duplicated.append(index)
        seen.add(index)

    problems: list[str] = []
    if out_of_range:
        problems.append(f"chỉ số {what} ngoài khoảng 0-{rubric_size - 1}: {sorted(set(out_of_range))}")
    if duplicated:
        problems.append(f"trùng chỉ số {what}: {sorted(set(duplicated))}")
    if require_all:
        missing = [i for i in range(rubric_size) if i not in seen]
        if missing:
            problems.append(f"thiếu {what}: {missing}")
    if problems:
        raise MalformedGrade("AI trả kết quả sai rubric (" + "; ".join(problems) + ")")


__all__ = ["MalformedGrade", "check_verdict_indices"]
