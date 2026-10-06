"""Unit tests for probability aggregation.

Covers `arithmetic_mean`, `logit_space_mean`, `loo_shrinkage`,
`majority_vote_v4_letter`, `majority_vote_accuracy_v4`.

The geometric-mean calculation moves the
output AWAY from 0.5 when trials disagree asymmetrically (e.g. 2× (0.9,
0.1) + 1× (0.1, 0.9)). We assert the unambiguous invariants — K=1 reduce,
simplex preservation, trial-consistent equals arithmetic — and assert
non-equality with arithmetic mean for the divergent case.
"""
from __future__ import annotations

import math

import pytest

from forecast_eval.analysis.aggregation import (
    ShrinkageResult,
    _default_alpha_grid,
    arithmetic_mean,
    logit_space_mean,
    loo_shrinkage,
    majority_vote_accuracy_v4,
    majority_vote_v4_letter,
    shrinkage_predict,
)


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


# --------------------------------------------------------------------------- #
# arithmetic_mean
# --------------------------------------------------------------------------- #


def test_arithmetic_mean_k1_identity():
    out = arithmetic_mean([[0.7, 0.3]])
    assert out == [0.7, 0.3]


def test_arithmetic_mean_uniform():
    out = arithmetic_mean([[0.5, 0.5]] * 5)
    assert _close(out[0], 0.5) and _close(out[1], 0.5)


def test_arithmetic_mean_trial_divergent():
    out = arithmetic_mean([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9]])
    assert _close(out[0], (0.9 + 0.9 + 0.1) / 3)
    assert _close(out[1], (0.1 + 0.1 + 0.9) / 3)


def test_arithmetic_mean_empty_raises():
    with pytest.raises(ValueError):
        arithmetic_mean([])


def test_arithmetic_mean_length_mismatch_raises():
    with pytest.raises(ValueError):
        arithmetic_mean([[0.5, 0.5], [0.3, 0.3, 0.4]])


# --------------------------------------------------------------------------- #
# logit_space_mean
# --------------------------------------------------------------------------- #


def test_logit_mean_k1_identity_single():
    """Spec: K=1 → output equals input within clip precision."""
    out = logit_space_mean([[0.7, 0.3]], "single")
    assert _close(out[0], 0.7, tol=1e-6)
    assert _close(out[1], 0.3, tol=1e-6)


def test_logit_mean_k1_identity_multi():
    out = logit_space_mean([[0.8, 0.2, 0.6, 0.1]], "multi")
    expected = [0.8, 0.2, 0.6, 0.1]
    for o, e in zip(out, expected):
        assert _close(o, e, tol=1e-6)


def test_logit_mean_trial_consistent_single():
    """Spec scenario: 3 trials all (0.7, 0.3) → logit mean = arith mean = (0.7, 0.3)."""
    pred = [[0.7, 0.3]] * 3
    arith = arithmetic_mean(pred)
    logit = logit_space_mean(pred, "single")
    assert _close(arith[0], logit[0], tol=1e-9)
    assert _close(arith[1], logit[1], tol=1e-9)


def test_logit_mean_trial_consistent_multi():
    pred = [[0.7, 0.3, 0.1]] * 3
    logit = logit_space_mean(pred, "multi")
    for o, e in zip(logit, [0.7, 0.3, 0.1]):
        assert _close(o, e, tol=1e-6)


def test_logit_mean_trial_divergent_single_differs_from_arith():
    """Spec scenario: 2× (0.9, 0.1) + 1× (0.1, 0.9) — logit mean differs from arithmetic."""
    pred = [[0.9, 0.1], [0.9, 0.1], [0.1, 0.9]]
    arith = arithmetic_mean(pred)
    logit = logit_space_mean(pred, "single")
    # Arithmetic mean = (0.633, 0.367) from spec
    assert _close(arith[0], (0.9 + 0.9 + 0.1) / 3, tol=1e-9)
    # Both must be valid simplexes
    assert _close(sum(logit), 1.0, tol=1e-9)
    # logit ≠ arith
    assert abs(arith[0] - logit[0]) > 0.01
    # logit must respect majority direction (label 0 > label 1)
    assert logit[0] > logit[1]


def test_logit_mean_simplex_preservation_single():
    pred = [[0.4, 0.5, 0.1], [0.6, 0.2, 0.2], [0.3, 0.3, 0.4]]
    logit = logit_space_mean(pred, "single")
    assert _close(sum(logit), 1.0, tol=1e-9)
    for v in logit:
        assert 0.0 <= v <= 1.0


def test_logit_mean_multi_independent_per_label():
    """Multi: each label is an independent Bernoulli; sigmoid(mean(logit p))."""
    pred = [[0.9, 0.1, 0.5], [0.9, 0.1, 0.5]]
    logit = logit_space_mean(pred, "multi")
    # Each label is the same value across trials → sigmoid(logit) = original.
    assert _close(logit[0], 0.9, tol=1e-6)
    assert _close(logit[1], 0.1, tol=1e-6)
    assert _close(logit[2], 0.5, tol=1e-6)


def test_logit_mean_multi_does_not_normalize():
    """Multi sum should NOT be forced to 1 (paper §A.2 convention)."""
    pred = [[0.8, 0.8, 0.8]] * 3
    logit = logit_space_mean(pred, "multi")
    # Sum is 2.4, definitely not 1.0.
    assert sum(logit) > 2.0


def test_logit_mean_handles_extreme_clipping():
    """log(0) and logit(1) must not blow up; clip floor handles it."""
    pred = [[1.0, 0.0]] * 2
    logit_single = logit_space_mean(pred, "single")
    # Even after clipping, label 0 should dominate.
    assert logit_single[0] > 0.99
    logit_multi = logit_space_mean(pred, "multi")
    assert logit_multi[0] > 0.99


# --------------------------------------------------------------------------- #
# shrinkage_predict and loo_shrinkage
# --------------------------------------------------------------------------- #


def test_shrinkage_alpha_zero_returns_uniform_single():
    """alpha=0 in single → uniform (1/k, ..., 1/k) for any input."""
    agg_logits = [math.log(0.9), math.log(0.05), math.log(0.05)]
    out = shrinkage_predict(agg_logits, 0.0, "single")
    for v in out:
        assert _close(v, 1.0 / 3, tol=1e-9)


def test_shrinkage_alpha_zero_returns_half_multi():
    """alpha=0 in multi → 0.5 for every label (sigmoid(0))."""
    agg_logits = [3.0, -2.0, 0.5]
    out = shrinkage_predict(agg_logits, 0.0, "multi")
    for v in out:
        assert _close(v, 0.5, tol=1e-9)


def test_shrinkage_alpha_one_equals_logit_mean_single():
    """alpha=1 in single → softmax of mean log p, exactly logit_space_mean."""
    pred = [[0.7, 0.3], [0.5, 0.5]]
    direct = logit_space_mean(pred, "single")
    n = len(pred)
    log_sums = [sum(math.log(min(max(v, 1e-3), 1 - 1e-3)) for v in (p[i] for p in pred)) / n
                for i in range(2)]
    via_shrinkage = shrinkage_predict(log_sums, 1.0, "single")
    assert _close(direct[0], via_shrinkage[0], tol=1e-9)
    assert _close(direct[1], via_shrinkage[1], tol=1e-9)


def test_loo_shrinkage_default_grid_size():
    """Default alpha grid is 11 points from 0.0 to 1.0 step 0.1."""
    grid = _default_alpha_grid()
    assert grid == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def test_loo_shrinkage_returns_alpha_star_and_curve():
    """K=2 trials, perfect prediction on a yes/no question — alpha=1 minimizes BS."""
    # All trials predict (0.99, 0.01) on a question where answer is A.
    pred_q1 = [[0.99, 0.01], [0.99, 0.01]]
    pred_q2 = [[0.99, 0.01], [0.99, 0.01]]
    obs_q1 = [1, 0]
    obs_q2 = [1, 0]
    res = loo_shrinkage(
        [pred_q1, pred_q2], [obs_q1, obs_q2], "single",
    )
    assert isinstance(res, ShrinkageResult)
    assert res.n_questions == 2
    assert len(res.curve) == 11
    # Confident-and-correct → BS minimized at alpha=1 (full signal).
    assert _close(res.alpha_star, 1.0, tol=1e-9)
    # Curve should include both endpoints with sensible BI.
    bi_alpha_zero = res.curve[0][2]   # alpha=0 → uniform → BI = 100*(1-sqrt(0.25)) = 50
    bi_alpha_one = res.curve[-1][2]
    assert bi_alpha_zero < bi_alpha_one  # signal helps


def test_loo_shrinkage_finds_intermediate_alpha_for_overconfident():
    """If the model is overconfident on wrong predictions, smaller alpha wins.

    Setup: 2 trials per question, all predict (0.99, 0.01) — but the answer
    is B (label 1). Shrinking toward uniform reduces NLL/BS.
    """
    pred = [[0.99, 0.01], [0.99, 0.01]]
    obs = [0, 1]  # answer is label 1
    res = loo_shrinkage([pred] * 5, [obs] * 5, "single")
    # alpha=0 (uniform 0.5/0.5) is better than alpha=1 (very wrong prediction)
    bs_alpha_zero = res.curve[0][1]
    bs_alpha_one = res.curve[-1][1]
    assert bs_alpha_zero < bs_alpha_one
    # alpha* should be at one of the smaller alphas
    assert res.alpha_star <= 0.5


def test_loo_shrinkage_validates_lengths():
    with pytest.raises(ValueError):
        loo_shrinkage([], [], "single")
    with pytest.raises(ValueError):
        loo_shrinkage([[[0.5, 0.5]]], [[1, 0], [0, 1]], "single")


def test_loo_shrinkage_recovers_synthetic_alpha_star():
    """Inject an α=0.5 prior and verify the grid recovers α* near 0.5.

    Construct: model is overconfident at strength α=1 (raw signal). Truth
    matches with probability 0.7. Optimal shrinkage on Brier is α<1.

    We don't claim recovery to 4 decimal places — gridded LOO selects from
    {0.0, 0.1, ..., 1.0}, and the true minimizer depends on the noise model.
    The assertion is that α* lies in (0.0, 1.0) — the model neither rejects
    the K-trial signal entirely nor uses it raw. In practice α* lands at 0.5
    or 0.6 depending on the random sample.
    """
    import random

    rng = random.Random(20260425)
    pred_per_q = []
    obs_per_q = []
    for _ in range(80):
        # Truth: label A wins with probability 0.7.
        # Trial 1 + 2: model says (0.95, 0.05) — strongly biased toward A.
        # This is overconfident: α<1 helps.
        if rng.random() < 0.7:
            obs = [1, 0]
        else:
            obs = [0, 1]
        pred_per_q.append([[0.95, 0.05], [0.95, 0.05]])
        obs_per_q.append(obs)
    res = loo_shrinkage(pred_per_q, obs_per_q, "single")
    # α* should NOT be at the extreme ends.
    assert 0.0 < res.alpha_star < 1.0
    # The BI at α* should be better (≥ in BI ↔ ≤ in BS) than at α=1.0.
    bs_at_one = res.curve[-1][1]
    assert res.mean_bs_star <= bs_at_one + 1e-9


def test_loo_shrinkage_custom_alpha_grid():
    pred = [[0.7, 0.3]]
    obs = [1, 0]
    res = loo_shrinkage([pred], [obs], "single", alpha_grid=[0.0, 0.5, 1.0])
    assert len(res.curve) == 3
    assert [a for a, _, _ in res.curve] == [0.0, 0.5, 1.0]


# --------------------------------------------------------------------------- #
# majority_vote_v4_letter
# --------------------------------------------------------------------------- #


def test_majority_vote_v4_letter_single_argmax():
    out = majority_vote_v4_letter([0.7, 0.2, 0.1], "single")
    assert out == frozenset({"A"})
    out = majority_vote_v4_letter([0.1, 0.7, 0.2], "single")
    assert out == frozenset({"B"})


def test_majority_vote_v4_letter_multi_threshold():
    """Multi: per-label threshold at 0.5 → letter set."""
    out = majority_vote_v4_letter([0.7, 0.4, 0.6, 0.1], "multi")
    assert out == frozenset({"A", "C"})


def test_majority_vote_v4_letter_multi_empty_set():
    out = majority_vote_v4_letter([0.4, 0.4, 0.4], "multi")
    assert out == frozenset()


def test_majority_vote_v4_letter_no_tie_under_floats():
    """Spec scenario: K=4 trials with different letters → unique aggregated letter."""
    # 4 trials each strongly favouring a different letter.
    pred = [
        [0.9, 0.04, 0.03, 0.03],
        [0.04, 0.9, 0.03, 0.03],
        [0.03, 0.04, 0.9, 0.03],
        [0.04, 0.03, 0.03, 0.9],
    ]
    agg = logit_space_mean(pred, "single")
    out = majority_vote_v4_letter(agg, "single")
    assert len(out) == 1
    # No assertion on which letter wins — point is uniqueness.


# --------------------------------------------------------------------------- #
# majority_vote_accuracy_v4
# --------------------------------------------------------------------------- #


def test_majority_vote_accuracy_v4_single():
    """Two questions, model right on first, wrong on second."""
    preds_per_q = [
        [[0.8, 0.2], [0.7, 0.3]],   # Q1: argmax = A
        [[0.3, 0.7], [0.4, 0.6]],   # Q2: argmax = B
    ]
    gt_per_q = [frozenset({"A"}), frozenset({"A"})]
    ctype = ["single", "single"]
    correct, resolvable = majority_vote_accuracy_v4(preds_per_q, gt_per_q, ctype)
    assert correct == 1
    assert resolvable == 2


def test_majority_vote_accuracy_v4_multi():
    """Multi with set equality."""
    preds_per_q = [
        [[0.8, 0.1, 0.7, 0.2], [0.7, 0.2, 0.6, 0.1]],   # → {A, C}
    ]
    gt_per_q = [frozenset({"A", "C"})]
    ctype = ["multi"]
    correct, resolvable = majority_vote_accuracy_v4(preds_per_q, gt_per_q, ctype)
    assert correct == 1
    assert resolvable == 1


def test_majority_vote_accuracy_v4_skips_empty_predictions():
    preds_per_q = [
        [[0.8, 0.2]],
        [],   # No trials for this question
    ]
    gt_per_q = [frozenset({"A"}), frozenset({"A"})]
    ctype = ["single", "single"]
    correct, resolvable = majority_vote_accuracy_v4(preds_per_q, gt_per_q, ctype)
    assert correct == 1
    assert resolvable == 1   # second question skipped


def test_majority_vote_accuracy_v4_validates_lengths():
    with pytest.raises(ValueError):
        majority_vote_accuracy_v4(
            [[[0.5, 0.5]]],
            [frozenset({"A"})],
            ["single", "single"],
        )
