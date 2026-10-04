"""
Vòng kiểm chứng nguồn dùng chung cho quiz và card (doc §4.4/§4.5: CoVe + Reflexion).

Chấm cả mẻ, cho từng phần tử bị loại một số lượt sửa có giới hạn, phần còn
sai sau lượt cuối thì trả về riêng để người gọi **không lưu**.
"""
from typing import Callable, TypeVar

T = TypeVar("T")


def verify_and_refine(
    items: list[T],
    *,
    find_failing: Callable[[list[T]], dict[int, str] | None],
    regenerate: Callable[[T, str], T | None],
    max_retries: int,
    on_exhausted: Callable[[list[int]], None] | None = None,
) -> tuple[list[T], list[tuple[T, str]]]:
    """
    Trả về (giữ lại, bị loại kèm lý do).

    ``find_failing`` trả về {chỉ số: lý do} của các phần tử không bám nguồn,
    hoặc None khi chính bước chấm hỏng. Chấm hỏng thì coi là không kết luận
    được và giữ nguyên mẻ: người duyệt vẫn là cổng cuối.
    """
    current = list(items)
    if not current:
        return current, []

    for attempt in range(max_retries + 1):
        failing = find_failing(current)
        if failing is None:
            return current, []

        failing = {i: reason for i, reason in failing.items() if 0 <= i < len(current)}
        if not failing:
            return current, []

        if attempt == max_retries:
            if on_exhausted is not None:
                on_exhausted(sorted(failing.keys()))
            rejected = [(current[i], reason) for i, reason in sorted(failing.items())]
            kept = [item for i, item in enumerate(current) if i not in failing]
            return kept, rejected

        for index, reason in failing.items():
            fixed = regenerate(current[index], reason)
            if fixed is not None:
                current[index] = fixed

    return current, []


__all__ = ["verify_and_refine"]
