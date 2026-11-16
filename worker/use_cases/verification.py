"""
Vòng kiểm chứng nguồn dùng chung cho quiz và card (doc §4.4/§4.5: CoVe + Reflexion).

Chấm cả mẻ, cho từng phần tử bị loại một số lượt sửa có giới hạn, phần còn
sai sau lượt cuối thì trả về riêng để người gọi **không lưu**.

Mỗi phần tử giữ lại mang theo kết luận của judge (``Verification``) để core-api
lưu vào ``verification_status`` / ``verification_evidence``. Judge chỉ kiểm
câu/card có bám đoạn nguồn hay không (``criterion = source_grounding``); nó
KHÔNG đánh giá mức Bloom — mức Bloom trong hồ sơ là mức được yêu cầu.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterable, TypeVar

T = TypeVar("T")

SUPPORTED = "supported"

# Khớp CHECK quiz_items/lesson_cards.verification_status (migration 20261107120000).
VERIFIED = "VERIFIED"
INCONCLUSIVE = "INCONCLUSIVE"
FLAGGED = "FLAGGED"

# Lý do INCONCLUSIVE, ghi vào hồ sơ để phân biệt "judge hỏng" với "judge im lặng".
JUDGE_FAILED = "judge_failed"
NO_VERDICT = "no_verdict"

CRITERION = "source_grounding"

# Lý do của judge là văn bản tự do của model: chặn độ dài để hồ sơ (jsonb) và
# payload kết quả không phình theo một câu trả lời lan man.
_MAX_REASON_CHARS = 1000


def _clip(text: str | None) -> str | None:
    if text is None:
        return None
    text = text.strip()
    return text if len(text) <= _MAX_REASON_CHARS else text[: _MAX_REASON_CHARS - 1] + "…"


@dataclass(frozen=True)
class JudgeVerdict:
    verdict: str
    reason: str = ""

    @property
    def supported(self) -> bool:
        return self.verdict == SUPPORTED


@dataclass(frozen=True)
class VerdictRecord:
    """Một lần judge nói về phần tử ở vị trí này (có thể là bản trước khi sinh lại)."""

    round: int
    verdict: str
    reason: str


@dataclass(frozen=True)
class Verification:
    """
    Kết luận cho bản cuối của một phần tử được giữ lại.

    - VERIFIED: judge nói ``supported`` cho đúng bản này.
    - INCONCLUSIVE: bản này chưa từng được judge cho kết luận — bước chấm hỏng
      (``judge_failed``) hoặc judge không trả verdict cho nó (``no_verdict``).
    - FLAGGED: kết luận gần nhất của judge cho bản này là không đạt, nhưng vòng
      chấm sau hỏng/không nhắc tới nó nên vẫn được giữ như hành vi cũ.
    """

    status: str
    judge_verdict: str | None
    judge_reason: str | None
    inconclusive_reason: str | None
    regeneration_attempts: int
    regenerations_applied: int
    history: tuple[VerdictRecord, ...] = ()

    def evidence(
        self,
        *,
        writer_model: str | None,
        judge_model: str | None,
        source_chunk_ids: Iterable[str],
        **extra: Any,
    ) -> dict[str, Any]:
        """Hồ sơ lưu vào ``verification_evidence`` (jsonb)."""
        return {
            "criterion": CRITERION,
            "judge_verdict": self.judge_verdict,
            "judge_reason": _clip(self.judge_reason),
            "inconclusive_reason": self.inconclusive_reason,
            "regeneration_attempts": self.regeneration_attempts,
            "regenerations_applied": self.regenerations_applied,
            "history": [
                {"round": r.round, "verdict": r.verdict, "reason": _clip(r.reason)}
                for r in self.history
            ],
            "writer_model": writer_model,
            "judge_model": judge_model,
            # Cùng một model vừa viết vừa chấm có xu hướng tự nhận là đúng:
            # ghi lại để người đọc hồ sơ biết độ tin của VERIFIED.
            "judge_is_writer": bool(writer_model) and writer_model == judge_model,
            "source_chunk_ids": list(source_chunk_ids),
            **extra,
        }


@dataclass(frozen=True)
class Kept(Generic[T]):
    item: T
    verification: Verification


@dataclass(frozen=True)
class Rejected(Generic[T]):
    item: T
    reason: str
    regeneration_attempts: int
    history: tuple[VerdictRecord, ...] = ()


@dataclass(frozen=True)
class RefineResult(Generic[T]):
    kept: list[Kept[T]]
    rejected: list[Rejected[T]]
    judge_rounds: int = 0
    judge_failed: bool = False

    @property
    def items(self) -> list[T]:
        return [k.item for k in self.kept]


def collect_verdicts(entries: Iterable[tuple[int, str, str]]) -> dict[int, JudgeVerdict]:
    """
    (chỉ số, verdict, lý do) của judge → {chỉ số: JudgeVerdict}.

    Judge trả hai verdict cho cùng một chỉ số thì verdict không đạt thắng: code
    cũ coi phần tử là hỏng nếu có bất kỳ verdict hỏng nào cho nó.
    """
    out: dict[int, JudgeVerdict] = {}
    for index, verdict, reason in entries:
        new = JudgeVerdict(verdict=verdict, reason=reason or "")
        old = out.get(index)
        if old is None or (old.supported and not new.supported):
            out[index] = new
    return out


def verify_and_refine(
    items: list[T],
    *,
    judge: Callable[[list[T]], dict[int, JudgeVerdict] | None],
    regenerate: Callable[[T, str], T | None],
    max_retries: int,
    on_exhausted: Callable[[list[int]], None] | None = None,
) -> RefineResult[T]:
    """
    Chấm, sinh lại phần tử hỏng, lặp tối đa ``max_retries`` lượt sửa.

    ``judge`` trả {chỉ số: verdict} cho mẻ hiện tại, hoặc None khi chính bước
    chấm hỏng. Chấm hỏng thì coi là không kết luận được và giữ nguyên mẻ: người
    duyệt vẫn là cổng cuối. Quyết định sinh lại/loại chỉ dựa trên verdict của
    vòng hiện tại (như trước); trạng thái hồ sơ dựa trên verdict gần nhất cho
    đúng bản đang giữ.
    """
    current = list(items)
    n = len(current)
    if not current:
        return RefineResult(kept=[], rejected=[])

    latest: list[JudgeVerdict | None] = [None] * n  # verdict cho bản hiện tại
    history: list[list[VerdictRecord]] = [[] for _ in range(n)]
    attempts = [0] * n
    applied = [0] * n

    def finish(
        rejected_reasons: dict[int, str], rounds: int, judge_failed: bool
    ) -> RefineResult[T]:
        kept: list[Kept[T]] = []
        rejected: list[Rejected[T]] = []
        for i, item in enumerate(current):
            if i in rejected_reasons:
                rejected.append(Rejected(
                    item=item,
                    reason=rejected_reasons[i],
                    regeneration_attempts=attempts[i],
                    history=tuple(history[i]),
                ))
                continue
            verdict = latest[i]
            if verdict is None:
                status = INCONCLUSIVE
                why = JUDGE_FAILED if judge_failed else NO_VERDICT
            else:
                status = VERIFIED if verdict.supported else FLAGGED
                why = None
            kept.append(Kept(item=item, verification=Verification(
                status=status,
                judge_verdict=verdict.verdict if verdict else None,
                judge_reason=verdict.reason if verdict else None,
                inconclusive_reason=why,
                regeneration_attempts=attempts[i],
                regenerations_applied=applied[i],
                history=tuple(history[i]),
            )))
        return RefineResult(
            kept=kept, rejected=rejected, judge_rounds=rounds, judge_failed=judge_failed
        )

    for round_no in range(max_retries + 1):
        verdicts = judge(current)
        if verdicts is None:
            return finish({}, round_no + 1, judge_failed=True)

        this_round = {i: v for i, v in verdicts.items() if 0 <= i < n}
        for i, verdict in sorted(this_round.items()):
            latest[i] = verdict
            history[i].append(VerdictRecord(round=round_no, verdict=verdict.verdict, reason=verdict.reason))

        failing = {i: v.reason for i, v in this_round.items() if not v.supported}
        if not failing:
            return finish({}, round_no + 1, judge_failed=False)

        if round_no == max_retries:
            if on_exhausted is not None:
                on_exhausted(sorted(failing.keys()))
            return finish(failing, round_no + 1, judge_failed=False)

        for index, reason in failing.items():
            attempts[index] += 1
            fixed = regenerate(current[index], reason)
            if fixed is not None:
                current[index] = fixed
                applied[index] += 1
                # Bản mới chưa được chấm: kết luận cũ thuộc về bản đã bị thay.
                latest[index] = None

    return finish({}, max_retries + 1, judge_failed=False)  # pragma: no cover


__all__ = [
    "CRITERION",
    "FLAGGED",
    "INCONCLUSIVE",
    "JUDGE_FAILED",
    "NO_VERDICT",
    "SUPPORTED",
    "VERIFIED",
    "JudgeVerdict",
    "Kept",
    "RefineResult",
    "Rejected",
    "VerdictRecord",
    "Verification",
    "collect_verdicts",
    "verify_and_refine",
]
