"""Unit tests for v5 discrete-native family in `forecast_eval.analysis.accuracy`.

Pins:
* `tversky_score` formula (plan §1.1) — single-choice strict / multi partial credit.
* `tversky_baseline` precise enumeration (plan §1.6) — closed form vs brute-force.
* `fss` three-step aggregate (plan §1.7) — per-question Tversky → chance correction → mean.
* `cohen_kappa` / `cohen_kappa_for_aggregate` (plan §2.3) — yes_no / 10-choice examples.
* `hamming_score_per_question` / `hamming_score` (plan §2.4) — multi-only partial credit.
"""
from __future__ import annotations

import itertools
import math

import pytest

from forecast_eval.analysis.accuracy import (
    cohen_kappa,
    cohen_kappa_for_aggregate,
    fss,
    hamming_score,
    hamming_score_per_question,
    tversky_baseline,
    tversky_score,
)
from forecast_eval.analysis.flatten import SampleRow


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_sample(
    *,
    model: str = "model_x",
    question_id: str,
    question_type: str = "single",
    choice_type: str = "single",
    options: list[str] | None = None,
    sample_idx: int = 0,
    correct: int | None = 1,
    parse_ok: int | None = 1,
    parsed: frozenset[str] | None = None,
) -> SampleRow:
    """Construct a SampleRow. `parsed` overrides `final_answer_letters`."""
    if options is None:
        options = ["A", "B", "C", "D"]
    final_letters_json = None
    if parsed is not None:
        import json
        final_letters_json = json.dumps(sorted(parsed))
    return SampleRow(
        model=model,
        question_id=question_id,
        question_type=question_type,
        choice_type=choice_type,
        options=options,
        sample_idx=sample_idx,
        correct=correct,
        parse_ok=parse_ok,
        tool_calls_count=0,
        react_steps=0,
        prompt_tokens=0,
        completion_tokens=0,
        reasoning_tokens=0,
        latency_ms=0,
        final_answer_letters=final_letters_json,
        error=None,
        created_at="2026-04-26T00:00:00Z",
        finish_reason="stop",
        nudges_used=0,
        belief_final=None,
        belief_trace=None,
        belief_parse_ok=0,
        probabilities=None,
        is_fallback=False,
    )


# --------------------------------------------------------------------------- #
# tversky_score
# --------------------------------------------------------------------------- #


def test_tversky_score_single_choice_strict_01() -> None:
    """Task 1.3: single-choice answer hit → 1.0, miss → 0.0."""
    assert tversky_score(frozenset({"A"}), frozenset({"A"})) == pytest.approx(1.0)
    assert tversky_score(frozenset({"B"}), frozenset({"A"})) == pytest.approx(0.0)


def test_tversky_score_multi_partial_credit() -> None:
    """Task 1.4: plan §1.4 8-choose-3 GT={A,C,F} table, all 8 cases manually verified.

    Default α=2/β=0.5: missing 1 costs 0.5, extra 1 costs 2 (4× asymmetry).
    """
    gt = frozenset({"A", "C", "F"})
    cases = [
        (frozenset({"A", "C", "F"}), 1.0),                # perfect: TP=3, denom=3
        (frozenset({"A", "C"}), 2.0 / 2.5),               # miss 1: TP=2, FN=1, denom=2.5 → 0.80
        (frozenset({"A"}), 1.0 / 2.0),                    # miss 2: TP=1, FN=2, denom=2.0 → 0.50
        (frozenset({"A", "C", "F", "X"}), 3.0 / 5.0),     # extra 1: TP=3, FP=1, denom=5 → 0.60
        (frozenset({"A", "C", "F", "X", "Y"}), 3.0 / 7.0),  # extra 2: TP=3, FP=2, denom=7 → 0.4286
        (frozenset({"A", "C", "X"}), 2.0 / 4.5),          # mixed: TP=2, FP=1, FN=1, denom=4.5 → 0.4444
        (frozenset({"X", "Y"}), 0.0),                     # all wrong: TP=0 → 0
        (frozenset(), 0.0),                               # empty pred (anti-conservative): TP=0 → 0
    ]
    for pred, expected in cases:
        actual = tversky_score(pred, gt)
        assert actual == pytest.approx(expected, abs=1e-2), (
            f"pred={pred} expected={expected} actual={actual}"
        )


def test_tversky_score_multi_7choose4() -> None:
    """Task 1.5: plan §1.4 second table, 7-choose-4 GT={A,B,C,D}.

    Constructed cases probing perfect / miss-1 / extra-1 / mixed
    asymmetric penalties.
    """
    gt = frozenset({"A", "B", "C", "D"})
    cases = [
        (frozenset({"A", "B", "C", "D"}), 1.0),               # perfect
        (frozenset({"A", "B", "C"}), 3.0 / 3.5),              # miss 1: 3/(3+0+0.5)
        (frozenset({"A", "B", "C", "D", "E"}), 4.0 / 6.0),    # extra 1: 4/(4+2+0)
        (frozenset({"A", "B", "C", "X"}), 3.0 / 5.5),         # mixed: 3/(3+2+0.5)
    ]
    for pred, expected in cases:
        actual = tversky_score(pred, gt)
        assert actual == pytest.approx(expected, abs=1e-2), (
            f"pred={pred} expected={expected} actual={actual}"
        )


def test_tversky_score_empty_pred_returns_zero() -> None:
    """Task 1.6: model outputs empty set → 0.0 (anti-conservative).

    Without this rule, "give up and predict nothing" would be a degenerate
    Pareto strategy with FN=|GT| only and tversky = 0 / (0 + 0 + 0.5*|GT|) = 0,
    which is what we want.
    """
    assert tversky_score(frozenset(), frozenset({"A", "B"})) == 0.0
    assert tversky_score(frozenset(), frozenset({"A"})) == 0.0


def test_tversky_score_alpha_beta_asymmetry() -> None:
    """Default α=2 β=0.5: a single FP costs 0.4 from a 3-TP base, while a
    single FN only costs 0.20 — exactly the 4× false-positive to false-negative penalty ratio."""
    gt = frozenset({"A", "C", "F"})
    score_extra = tversky_score(frozenset({"A", "C", "F", "X"}), gt)  # 0.60
    score_miss = tversky_score(frozenset({"A", "C"}), gt)              # 0.80
    # Loss from FP (0.40) is 2× loss from FN (0.20). The asymmetry says
    # "extra hurts more than missed" — the 4× phrase in design.md refers to
    # α/β = 4, not the per-case marginal loss ratio.
    assert (1.0 - score_extra) > (1.0 - score_miss)


def test_tversky_score_custom_alpha_beta() -> None:
    """Verify (α, β) propagate through the sensitivity CLI."""
    gt = frozenset({"A", "B"})
    pred = frozenset({"A", "C"})  # TP=1, FP=1, FN=1
    # (α, β) = (1, 1) → Jaccard: 1 / (1 + 1 + 1) = 1/3
    assert tversky_score(pred, gt, alpha=1.0, beta=1.0) == pytest.approx(1.0 / 3.0)
    # (α, β) = (1, 0.5): 1 / (1 + 1 + 0.5) = 1/2.5 = 0.4
    assert tversky_score(pred, gt, alpha=1.0, beta=0.5) == pytest.approx(0.4)
    # (α, β) = (3, 0.5): 1 / (1 + 3 + 0.5) = 1/4.5 ≈ 0.222
    assert tversky_score(pred, gt, alpha=3.0, beta=0.5) == pytest.approx(1.0 / 4.5)


def test_tversky_score_empty_gt_defensive() -> None:
    """GT empty (degenerate; should not occur in this dataset) returns 1.0
    iff pred is also empty, else 0.0 — keeps the function total."""
    assert tversky_score(frozenset(), frozenset()) == 1.0
    assert tversky_score(frozenset({"A"}), frozenset()) == 0.0


# --------------------------------------------------------------------------- #
# tversky_baseline
# --------------------------------------------------------------------------- #


def _brute_force_tversky_baseline(
    k: int, m: int, *, alpha: float = 2.0, beta: float = 0.5
) -> float:
    """Ground-truth via $2^k$ enumeration: iterate every subset of {0..k-1}
    that the random predictor could pick, compute Tversky against a fixed
    "first m positives" GT, and average. $O(2^k)$ — only used for $k \\le 12$
    test fixtures."""
    if k <= 0 or m <= 0 or m > k:
        return 0.0
    gt_indices = set(range(m))
    total = 0.0
    n_subsets = 0
    for r in range(k + 1):
        for combo in itertools.combinations(range(k), r):
            pred = set(combo)
            tp = len(pred & gt_indices)
            if tp == 0:
                tversky = 0.0
            else:
                fp = len(pred - gt_indices)
                fn = len(gt_indices - pred)
                tversky = tp / (tp + alpha * fp + beta * fn)
            total += tversky
            n_subsets += 1
    return total / n_subsets


def test_tversky_baseline_known_values() -> None:
    """Task 1.7: closed-form vs brute force on small k/m grid (precision 1e-4)."""
    for k, m in [(4, 1), (4, 2), (8, 3), (10, 4), (6, 6)]:
        closed = tversky_baseline(k, m)
        brute = _brute_force_tversky_baseline(k, m)
        assert closed == pytest.approx(brute, abs=1e-4), (
            f"k={k} m={m}: closed={closed} brute={brute}"
        )


def test_tversky_baseline_degenerate_returns_zero() -> None:
    """k <= 0 / m <= 0 / m > k should return 0.0 (defensive)."""
    assert tversky_baseline(0, 0) == 0.0
    assert tversky_baseline(4, 0) == 0.0
    assert tversky_baseline(0, 1) == 0.0
    assert tversky_baseline(3, 5) == 0.0  # m > k


def test_tversky_baseline_k35_fast() -> None:
    """k=35 / m=10 must complete in well under 1ms (plan §1.6 perf target).

    We don't time it here (timing is environment-dependent), but if the
    enumeration ever regresses to $O(2^k)$ this test would hang.
    """
    val = tversky_baseline(35, 10)
    # Sanity: baseline must be in (0, 1) — random predictor is between
    # always-empty (0) and perfect (1).
    assert 0.0 < val < 1.0


def test_tversky_baseline_low_for_difficult_questions() -> None:
    """Sanity: large k, small m → baseline very low (random predictor rarely
    gets even 1 of m correct without spurious FPs)."""
    assert tversky_baseline(20, 2) < 0.10
    # Whereas k=2 m=1 (yes/no) has high baseline
    assert tversky_baseline(2, 1) > 0.20


# --------------------------------------------------------------------------- #
# fss
# --------------------------------------------------------------------------- #


def test_fss_perfect_predictions_returns_one() -> None:
    """Task 2.3: every sample perfectly matches GT → FSS = 1.0."""
    options = ["A", "B", "C", "D"]
    samples = []
    gt_map = {}
    for q_idx in range(5):
        qid = f"q{q_idx}"
        gt = frozenset({"A"})  # single-choice
        gt_map[qid] = gt
        for trial in range(5):
            samples.append(_make_sample(
                question_id=qid,
                options=options,
                sample_idx=trial,
                parsed=gt,
            ))
    result = fss(samples, gt_map)
    assert result["fss"] == pytest.approx(1.0, abs=1e-9)
    assert result["n_valid"] == 5


def test_fss_random_predictions_returns_near_zero() -> None:
    """Task 2.4: seed-fixed uniform random predictor on 100 single-choice
    questions → FSS within [-0.1, 0.1] of 0 (skill score = 0 means "random")."""
    import random
    rng = random.Random(42)
    options = ["A", "B", "C", "D"]
    letters = ["A", "B", "C", "D"]
    samples = []
    gt_map = {}
    for q_idx in range(100):
        qid = f"q{q_idx}"
        gt_letter = rng.choice(letters)
        gt = frozenset({gt_letter})
        gt_map[qid] = gt
        for trial in range(5):
            pred_letter = rng.choice(letters)
            samples.append(_make_sample(
                question_id=qid,
                options=options,
                sample_idx=trial,
                parsed=frozenset({pred_letter}),
            ))
    result = fss(samples, gt_map)
    # Not asserting exact 0 — finite N has noise. ±0.1 generous bound.
    assert -0.10 < result["fss"] < 0.10


def test_fss_single_choice_4option_perfect_chance_corrected() -> None:
    """Single-choice 4-option, perfect prediction → c_q=1.0, p_e=0.25 → s_q=1.0."""
    options = ["A", "B", "C", "D"]
    qid = "q0"
    gt = frozenset({"B"})
    gt_map = {qid: gt}
    samples = [_make_sample(
        question_id=qid, options=options, parsed=gt, sample_idx=0,
    )]
    result = fss(samples, gt_map)
    pq = result["per_question"][qid]
    assert pq["c_q"] == pytest.approx(1.0)
    assert pq["p_e"] == pytest.approx(0.25)
    assert pq["s_q"] == pytest.approx(1.0)


def test_fss_skips_questions_with_zero_keff() -> None:
    """Task 2.6: question where every trial has parse_ok=0 → not in dict;
    n_valid is below the total question count."""
    options = ["A", "B", "C", "D"]
    samples = []
    gt_map = {}
    # q0: all 5 trials parse ok
    gt_map["q0"] = frozenset({"A"})
    for trial in range(5):
        samples.append(_make_sample(
            question_id="q0", options=options, sample_idx=trial,
            parsed=frozenset({"A"}), parse_ok=1,
        ))
    # q1: all 5 trials fail to parse (parse_ok=0, parsed=None)
    gt_map["q1"] = frozenset({"A"})
    for trial in range(5):
        samples.append(_make_sample(
            question_id="q1", options=options, sample_idx=trial,
            parsed=None, parse_ok=0,
        ))
    result = fss(samples, gt_map)
    assert result["n_valid"] == 1
    assert "q0" in result["per_question"]
    assert "q1" not in result["per_question"]


def test_fss_multi_choice_perfect_chance_corrected() -> None:
    """Multi-label perfect prediction → c_q=1.0 → s_q=1.0 regardless of p_e."""
    options = ["A", "B", "C", "D", "E", "F", "G", "H"]
    qid = "q0"
    gt = frozenset({"A", "C", "F"})  # 8 choose 3
    gt_map = {qid: gt}
    samples = [_make_sample(
        question_id=qid, options=options, choice_type="multi", parsed=gt,
    )]
    result = fss(samples, gt_map)
    pq = result["per_question"][qid]
    assert pq["c_q"] == pytest.approx(1.0)
    # multi p_e = tversky_baseline(8, 3) ≈ 0.18-0.25
    assert pq["s_q"] == pytest.approx(1.0)


def test_fss_empty_samples_returns_none() -> None:
    """Empty input → fss=None, n_valid=0 (sentinel for writer to emit blank cell)."""
    result = fss([], {})
    assert result["fss"] is None
    assert result["n_valid"] == 0
    assert result["mean_pe"] is None


def test_fss_per_question_reports_K_eff() -> None:
    """K_eff in per_question dict reflects only parse_ok=1 samples."""
    options = ["A", "B", "C", "D"]
    qid = "q0"
    gt = frozenset({"A"})
    gt_map = {qid: gt}
    # 3 ok + 2 fail = K_eff=3
    samples = []
    for trial in range(3):
        samples.append(_make_sample(
            question_id=qid, options=options, sample_idx=trial,
            parsed=frozenset({"A"}), parse_ok=1,
        ))
    for trial in range(3, 5):
        samples.append(_make_sample(
            question_id=qid, options=options, sample_idx=trial,
            parsed=None, parse_ok=0,
        ))
    result = fss(samples, gt_map)
    assert result["per_question"][qid]["K_eff"] == 3


# --------------------------------------------------------------------------- #
# Cohen's κ
# --------------------------------------------------------------------------- #


def test_cohen_kappa_for_aggregate_yes_no_60pct_acc() -> None:
    """Task 3.5: yes_no acc=0.6 / p_e=0.5 → κ=0.20 (plan §2.3 example)."""
    assert cohen_kappa_for_aggregate(0.6, 0.5) == pytest.approx(0.20, abs=1e-9)


def test_cohen_kappa_for_aggregate_10choice_60pct_acc() -> None:
    """Task 3.6: 10-choice acc=0.6 / p_e=0.1 → κ ≈ 0.5556 (plan §2.3)."""
    expected = (0.6 - 0.1) / (1.0 - 0.1)
    assert cohen_kappa_for_aggregate(0.6, 0.1) == pytest.approx(expected, abs=1e-6)
    assert cohen_kappa_for_aggregate(0.6, 0.1) == pytest.approx(0.5556, abs=1e-3)


def test_cohen_kappa_for_aggregate_negative_when_below_chance() -> None:
    """acc < p_e → κ < 0 (worse-than-random is reportable, not clipped)."""
    assert cohen_kappa_for_aggregate(0.05, 0.1) < 0


def test_cohen_kappa_for_aggregate_pe_one_returns_none() -> None:
    """p_e = 1.0 → division by zero; return None."""
    assert cohen_kappa_for_aggregate(1.0, 1.0) is None


def test_cohen_kappa_aggregate_single_choice() -> None:
    """Aggregate cohen_kappa over a single-choice fixture: 4-choice, 60% correct."""
    options = ["A", "B", "C", "D"]
    gt_map = {f"q{i}": frozenset({"A"}) for i in range(10)}
    samples = []
    # 6 correct / 4 wrong out of 10
    for i in range(6):
        samples.append(_make_sample(
            question_id=f"q{i}", options=options, parsed=frozenset({"A"}),
            correct=1,
        ))
    for i in range(6, 10):
        samples.append(_make_sample(
            question_id=f"q{i}", options=options, parsed=frozenset({"B"}),
            correct=0,
        ))
    samples_by_q: dict[str, list[SampleRow]] = {}
    for s in samples:
        samples_by_q.setdefault(s.question_id, []).append(s)
    kappa = cohen_kappa(samples_by_q, gt_map)
    # acc=0.6, p_e=0.25 → κ = (0.6 - 0.25) / 0.75 = 0.4667
    assert kappa == pytest.approx((0.6 - 0.25) / 0.75, abs=1e-6)


def test_cohen_kappa_empty_returns_none() -> None:
    assert cohen_kappa({}, {}) is None


# --------------------------------------------------------------------------- #
# Hamming
# --------------------------------------------------------------------------- #


def test_hamming_score_4_label_3_correct() -> None:
    """Task 3.7: GT={A,B,C,D} all selected, pred={A,B,C} miss D → Hamming=0.75.

    1 mismatch out of 4 labels → 1 - 1/4 = 0.75.
    """
    options = ["A", "B", "C", "D"]
    gt = frozenset({"A", "B", "C", "D"})
    pred = frozenset({"A", "B", "C"})
    assert hamming_score_per_question(pred, gt, options) == pytest.approx(0.75)


def test_hamming_score_perfect_returns_one() -> None:
    options = ["A", "B", "C", "D"]
    assert hamming_score_per_question(
        frozenset({"A", "B"}), frozenset({"A", "B"}), options,
    ) == pytest.approx(1.0)


def test_hamming_score_all_wrong() -> None:
    """GT={A,B}, pred={C,D} → 4 label mismatches → 1 - 1 = 0."""
    options = ["A", "B", "C", "D"]
    assert hamming_score_per_question(
        frozenset({"C", "D"}), frozenset({"A", "B"}), options,
    ) == pytest.approx(0.0)


def test_hamming_score_partial() -> None:
    """GT={A,C}, pred={A,B,C} → mismatch on B only → 1 - 1/4 = 0.75."""
    options = ["A", "B", "C", "D"]
    assert hamming_score_per_question(
        frozenset({"A", "B", "C"}), frozenset({"A", "C"}), options,
    ) == pytest.approx(0.75)


def test_hamming_score_skips_single_choice() -> None:
    """Task 3.8: pure single-choice fixture → hamming_score returns None."""
    options = ["A", "B", "C", "D"]
    samples = [_make_sample(
        question_id="q0", options=options, choice_type="single",
        parsed=frozenset({"A"}),
    )]
    gt_map = {"q0": frozenset({"A"})}
    assert hamming_score(samples, gt_map) is None


def test_hamming_score_aggregate_multi_only() -> None:
    """Multi-only fixture: 2 samples averaged."""
    options = ["A", "B", "C", "D"]
    gt_map = {
        "q0": frozenset({"A", "B"}),
        "q1": frozenset({"C", "D"}),
    }
    samples = [
        _make_sample(  # perfect match
            question_id="q0", options=options, choice_type="multi",
            parsed=frozenset({"A", "B"}),
        ),
        _make_sample(  # 2 mismatches out of 4 → 0.5
            question_id="q1", options=options, choice_type="multi",
            parsed=frozenset({"A", "C"}),
        ),
    ]
    score = hamming_score(samples, gt_map)
    assert score == pytest.approx(0.75, abs=1e-6)  # mean(1.0, 0.5)


def test_hamming_score_skips_parse_failures() -> None:
    """parse_ok=0 samples are excluded from the mean."""
    options = ["A", "B", "C", "D"]
    gt_map = {"q0": frozenset({"A", "B"}), "q1": frozenset({"C", "D"})}
    samples = [
        _make_sample(
            question_id="q0", options=options, choice_type="multi",
            parsed=frozenset({"A", "B"}),  # perfect
        ),
        _make_sample(
            question_id="q1", options=options, choice_type="multi",
            parsed=None, parse_ok=0,  # excluded
        ),
    ]
    assert hamming_score(samples, gt_map) == pytest.approx(1.0)
