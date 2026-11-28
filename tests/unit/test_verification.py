"""verify_and_refine: quyết định giữ/loại và kết luận lưu vào hồ sơ kiểm chứng."""
from worker.use_cases.shared.verification import (
    FLAGGED,
    INCONCLUSIVE,
    JUDGE_FAILED,
    NO_VERDICT,
    VERIFIED,
    JudgeVerdict,
    collect_verdicts,
    verify_and_refine,
)

OK = JudgeVerdict("supported", "khớp nguồn")


def bad(reason="sai"):
    return JudgeVerdict("contradicted", reason)


def scripted(*rounds):
    """Judge trả lần lượt các vòng đã soạn; None = bước chấm hỏng."""
    calls = []
    queue = list(rounds)

    def judge(current):
        calls.append(list(current))
        return queue.pop(0)

    judge.calls = calls
    return judge


def rewrite(item, reason):
    return f"{item}*"


def test_supported_items_are_verified_without_rewrites():
    result = verify_and_refine(
        ["a", "b"], judge=scripted({0: OK, 1: OK}), regenerate=rewrite, max_retries=2
    )

    assert [k.item for k in result.kept] == ["a", "b"]
    assert {k.verification.status for k in result.kept} == {VERIFIED}
    assert all(k.verification.regeneration_attempts == 0 for k in result.kept)
    assert result.rejected == []
    assert result.judge_rounds == 1


def test_rewritten_item_carries_the_verdict_of_its_new_version():
    result = verify_and_refine(
        ["a", "b"],
        judge=scripted({0: OK, 1: bad("thiếu nguồn")}, {0: OK, 1: OK}),
        regenerate=rewrite,
        max_retries=2,
    )

    a, b = result.kept
    assert b.item == "b*"
    assert b.verification.status == VERIFIED
    assert b.verification.regeneration_attempts == 1
    assert b.verification.regenerations_applied == 1
    assert [(h.round, h.verdict) for h in b.verification.history] == [
        (0, "contradicted"),
        (1, "supported"),
    ]
    assert a.verification.history[-1].round == 1


def test_judge_failure_on_first_round_keeps_everything_inconclusive():
    result = verify_and_refine(["a", "b"], judge=scripted(None), regenerate=rewrite, max_retries=2)

    assert [k.item for k in result.kept] == ["a", "b"]
    assert {k.verification.status for k in result.kept} == {INCONCLUSIVE}
    assert {k.verification.inconclusive_reason for k in result.kept} == {JUDGE_FAILED}
    assert result.judge_failed is True


def test_judge_failure_after_a_rewrite():
    """
    Bản chưa đổi giữ kết luận đã có; bản sinh lại chưa được chấm là
    INCONCLUSIVE; bản bị chê mà sinh lại hỏng (vẫn là bản cũ) là FLAGGED.
    """
    def regenerate(item, reason):
        return None if item == "c" else f"{item}*"

    result = verify_and_refine(
        ["a", "b", "c"],
        judge=scripted({0: OK, 1: bad(), 2: bad("mâu thuẫn slide 3")}, None),
        regenerate=regenerate,
        max_retries=2,
    )

    by_item = {k.item: k.verification for k in result.kept}
    assert by_item["a"].status == VERIFIED
    assert by_item["b*"].status == INCONCLUSIVE
    assert by_item["b*"].inconclusive_reason == JUDGE_FAILED
    assert by_item["c"].status == FLAGGED
    assert by_item["c"].judge_reason == "mâu thuẫn slide 3"
    assert by_item["c"].regeneration_attempts == 1
    assert by_item["c"].regenerations_applied == 0
    assert result.rejected == []


def test_item_without_a_verdict_is_kept_as_inconclusive():
    result = verify_and_refine(["a", "b"], judge=scripted({0: OK}), regenerate=rewrite, max_retries=2)

    a, b = result.kept
    assert a.verification.status == VERIFIED
    assert b.verification.status == INCONCLUSIVE
    assert b.verification.inconclusive_reason == NO_VERDICT


def test_item_still_failing_after_the_last_round_is_rejected_with_its_attempts():
    exhausted = []
    result = verify_and_refine(
        ["a", "b"],
        judge=scripted({0: OK, 1: bad("1")}, {0: OK, 1: bad("2")}, {0: OK, 1: bad("3")}),
        regenerate=rewrite,
        max_retries=2,
        on_exhausted=exhausted.append,
    )

    assert [k.item for k in result.kept] == ["a"]
    [rejected] = result.rejected
    assert rejected.item == "b**"
    assert rejected.reason == "3"
    assert rejected.regeneration_attempts == 2
    assert exhausted == [[1]]
    assert result.judge_rounds == 3


def test_out_of_range_verdicts_are_ignored():
    result = verify_and_refine(["a"], judge=scripted({0: OK, 7: bad()}), regenerate=rewrite, max_retries=0)

    assert [k.verification.status for k in result.kept] == [VERIFIED]


def test_conflicting_verdicts_for_one_item_count_as_failing():
    verdicts = collect_verdicts([(0, "supported", "ok"), (0, "contradicted", "sai"), (1, "supported", "")])

    assert verdicts[0] == JudgeVerdict("contradicted", "sai")
    assert verdicts[1].supported


def test_evidence_is_json_ready_and_names_what_was_checked():
    [kept] = verify_and_refine(
        ["a"], judge=scripted({0: JudgeVerdict("supported", "x" * 5000)}),
        regenerate=rewrite, max_retries=0,
    ).kept

    evidence = kept.verification.evidence(
        writer_model="w", judge_model="w", source_chunk_ids=["c1"], requested_bloom_level=2
    )

    assert evidence["criterion"] == "source_grounding"
    assert evidence["judge_is_writer"] is True
    assert len(evidence["judge_reason"]) == 1000
    assert evidence["requested_bloom_level"] == 2
    assert evidence["source_chunk_ids"] == ["c1"]
