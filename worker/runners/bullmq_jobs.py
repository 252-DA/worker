"""Đọc số lần chạy của một job BullMQ, dùng chung cho mọi runner."""
from typing import Any


def job_attempt(job: Any) -> int:
    """
    Lần chạy thứ mấy (từ 1) của job BullMQ đang xử lý. ``attemptsStarted`` tăng
    cả khi job bị chạy lại sau stall (worker chết giữa chừng), ``attemptsMade``
    chỉ tăng khi job lỗi; lấy số lớn hơn để lần chạy sau luôn lớn hơn lần trước.
    """
    started = _non_negative_int(getattr(job, "attemptsStarted", 0))
    made = _non_negative_int(getattr(job, "attemptsMade", 0))
    return max(started, made + 1, 1)


def is_final_attempt(bull_job: Any) -> bool:
    """``attemptsMade`` đếm số lần đã thất bại, nên lần đang chạy là lần thứ +1."""
    opts = getattr(bull_job, "opts", None) or {}
    attempts = int(opts.get("attempts") or 1)
    return int(getattr(bull_job, "attemptsMade", 0) or 0) + 1 >= attempts


def _non_negative_int(value: Any) -> int:
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(number, 0)


__all__ = ["is_final_attempt", "job_attempt"]
