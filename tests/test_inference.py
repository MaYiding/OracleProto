"""Unit tests for paired resampling and inference.

Covers `paired_bootstrap`, `holm_bonferroni`, `difficulty_tertile`,
`paired_bootstrap_by_difficulty`, `posterior_a_better_than_b`, and the
`pairwise_paired_bootstrap` orchestrator.

Contracts covered:
* Paired bootstrap is paired, not independent (same indices for A and B).
* Holm correction direction (multiply by remaining ranks descending).
* B = 5000 default; same seed → reproducible.
* Posterior probability is a valid [0, 1] number, same direction as p-value.
"""
from __future__ import annotations

import math
import random

import pytest

from forecast_eval.analysis.inference import (
    DifficultyTertile,
    ModelPairResult,
    PairedBootstrapResult,
    _normal_cdf,
    difficulty_tertile,
    holm_bonferroni,
    paired_bootstrap,
    paired_bootstrap_by_difficulty,
    pairwise_paired_bootstrap,
    posterior_a_better_than_b,
    posterior_normal_fit,
)


# --------------------------------------------------------------------------- #
# Paired bootstrap
# --------------------------------------------------------------------------- #


def test_paired_bootstrap_default_n_bootstrap_is_5000():
    """Spec scenario: B = 5000 by default."""
    bs_a = [0.1] * 50
    bs_b = [0.2] * 50
    res = paired_bootstrap(bs_a, bs_b)
    assert res.n_bootstrap == 5000


def test_paired_bootstrap_seed_is_reproducible():
    """Same seed → same bootstrap distribution → CIs differ by < 0.01."""
    bs_a = [0.1 + i * 0.01 for i in range(100)]
    bs_b = [0.2 - i * 0.01 for i in range(100)]
    res1 = paired_bootstrap(bs_a, bs_b, seed=999)
    res2 = paired_bootstrap(bs_a, bs_b, seed=999)
    assert abs(res1.ci_low - res2.ci_low) < 1e-9
    assert abs(res1.ci_high - res2.ci_high) < 1e-9
    assert abs(res1.p_two_sided - res2.p_two_sided) < 1e-9


def test_paired_bootstrap_zero_delta_p_value_high():
    """Identical models → delta = 0, p-value should be high (no signal)."""
    random.seed(7)
    bs_a = [random.random() for _ in range(100)]
    bs_b = list(bs_a)  # exactly the same
    res = paired_bootstrap(bs_a, bs_b)
    assert abs(res.delta_mean) < 1e-12
    # All bootstrap deltas = 0; p_two = 2 * min(1.0, 1.0) = 2.0 → clipped to 1
    assert res.p_two_sided >= 0.99


def test_paired_bootstrap_strong_difference_p_value_low():
    """Model A always 0.05 worse → strong signal, p < 0.01."""
    bs_a = [0.10] * 100
    bs_b = [0.05] * 100
    res = paired_bootstrap(bs_a, bs_b)
    assert res.delta_mean == pytest.approx(0.05, abs=1e-9)
    assert res.ci_low > 0  # CI excludes 0
    assert res.ci_high > 0
    assert res.p_two_sided < 0.01


def test_paired_bootstrap_validates_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap([0.1, 0.2], [0.1, 0.2, 0.3])


def test_paired_bootstrap_empty_input():
    res = paired_bootstrap([], [])
    assert res.n_questions == 0
    assert res.delta_mean == 0.0


def test_paired_bootstrap_pairing_property():
    """Spec scenario: the same bootstrap index hits BOTH arrays.

    With paired bootstrap, the variance of the delta is the variance of the
    per-question deltas. With independent bootstrap, it would be the sum
    of A's and B's variances. We verify by constructing data where paired
    deltas are constant (zero variance) — paired CI must be tight, while
    independent would be wide.
    """
    n = 50
    bs_a = [0.1 + i * 0.01 for i in range(n)]
    bs_b = [v - 0.05 for v in bs_a]   # delta is always exactly 0.05
    res = paired_bootstrap(bs_a, bs_b)
    # Paired delta has zero variance → CI collapses to ~0.05.
    assert abs(res.delta_mean - 0.05) < 1e-9
    assert abs(res.ci_low - 0.05) < 1e-9
    assert abs(res.ci_high - 0.05) < 1e-9


# --------------------------------------------------------------------------- #
# Holm-Bonferroni
# --------------------------------------------------------------------------- #


def test_holm_empty_input():
    assert holm_bonferroni([]) == []


def test_holm_single_pvalue_unchanged():
    """n=1: no correction needed."""
    assert holm_bonferroni([0.03]) == [0.03]


def test_holm_correction_direction_simple():
    """5 raw p-values → adjusted increase monotonically along sorted rank."""
    p_raw = [0.01, 0.02, 0.03, 0.04, 0.05]
    adj = holm_bonferroni(p_raw)
    # Smallest p multiplied by 5 = 0.05
    assert abs(adj[0] - 0.05) < 1e-9
    # 2nd smallest: max(0.05, 0.02 * 4) = max(0.05, 0.08) = 0.08
    assert abs(adj[1] - 0.08) < 1e-9
    # 3rd: max(0.08, 0.03 * 3) = max(0.08, 0.09) = 0.09
    assert abs(adj[2] - 0.09) < 1e-9
    # 4th: max(0.09, 0.04 * 2) = max(0.09, 0.08) = 0.09
    assert abs(adj[3] - 0.09) < 1e-9
    # 5th: max(0.09, 0.05 * 1) = max(0.09, 0.05) = 0.09
    assert abs(adj[4] - 0.09) < 1e-9


def test_holm_correction_with_unsorted_input():
    """Adjusted p-values returned in INPUT order, not sorted order."""
    p_raw = [0.04, 0.01, 0.03, 0.02]
    adj = holm_bonferroni(p_raw)
    assert len(adj) == 4
    # smallest is index 1 (p=0.01) → adjusted to 0.04
    assert abs(adj[1] - 0.04) < 1e-9


def test_holm_clips_to_one():
    """Adjusted p > 1 → clipped to 1.0."""
    adj = holm_bonferroni([0.5, 0.6])
    # 0.5 * 2 = 1.0; 0.6 * 1 = 0.6 < 1.0 → after cum max, 0.6 → 1.0
    assert all(v <= 1.0 for v in adj)


# --------------------------------------------------------------------------- #
# Difficulty tertile
# --------------------------------------------------------------------------- #


def test_difficulty_tertile_balanced_split():
    """N divisible by 3 → tertiles split as N/3 each."""
    gammas = {f"q{i}": i * 0.01 for i in range(30)}
    tert = difficulty_tertile(gammas)
    assert tert.n_low == 10
    assert tert.n_mid == 10
    assert tert.n_high == 10
    # Lowest gamma → "low"
    assert tert.by_question["q0"] == "low"
    assert tert.by_question["q15"] == "mid"
    assert tert.by_question["q29"] == "high"


def test_difficulty_tertile_uneven_split():
    """N=10 → tertiles 3/3/4 (or similar)."""
    gammas = {f"q{i}": i * 0.01 for i in range(10)}
    tert = difficulty_tertile(gammas)
    assert tert.n_low + tert.n_mid + tert.n_high == 10
    assert tert.n_high >= tert.n_mid >= tert.n_low - 1


def test_difficulty_tertile_empty_input():
    tert = difficulty_tertile({})
    assert tert.n_low == 0 and tert.n_mid == 0 and tert.n_high == 0
    assert tert.by_question == {}


def test_paired_bootstrap_by_difficulty_three_tiers():
    """Spec scenario: synthetic data where only 'high' tier has a difference.

    Construct: low tier all delta=0, mid tier all delta=0, high tier all
    delta=0.05. Per-tier paired bootstrap should give CI excluding 0 only
    on high.
    """
    bs_a_by_q: dict[str, float] = {}
    bs_b_by_q: dict[str, float] = {}
    gammas: dict[str, float] = {}
    for i in range(30):
        # Low tertile: gamma 0.0-0.1, no model difference.
        bs_a_by_q[f"l{i}"] = 0.10
        bs_b_by_q[f"l{i}"] = 0.10
        gammas[f"l{i}"] = 0.0 + i * 0.001
    for i in range(30):
        bs_a_by_q[f"m{i}"] = 0.20
        bs_b_by_q[f"m{i}"] = 0.20
        gammas[f"m{i}"] = 0.20 + i * 0.001
    for i in range(30):
        # High tertile: model A is 0.05 better.
        bs_a_by_q[f"h{i}"] = 0.30
        bs_b_by_q[f"h{i}"] = 0.35
        gammas[f"h{i}"] = 0.40 + i * 0.001
    tert = difficulty_tertile(gammas)
    out = paired_bootstrap_by_difficulty(bs_a_by_q, bs_b_by_q, tert, n_bootstrap=2000)
    # All three tiers populated
    assert "low" in out and "mid" in out and "high" in out
    # Low and mid: CI brackets 0.
    assert out["low"].ci_low <= 0 <= out["low"].ci_high
    assert out["mid"].ci_low <= 0 <= out["mid"].ci_high
    # High: CI strictly negative (A is better than B).
    assert out["high"].ci_high < 0
    assert abs(out["high"].delta_mean - (-0.05)) < 1e-9


# --------------------------------------------------------------------------- #
# Posterior over BI
# --------------------------------------------------------------------------- #


def test_normal_cdf_basic():
    """erf-based CDF: Φ(0) = 0.5, Φ(2) ≈ 0.977, Φ(-2) ≈ 0.023."""
    assert abs(_normal_cdf(0.0) - 0.5) < 1e-9
    assert abs(_normal_cdf(2.0) - 0.9772) < 0.01
    assert abs(_normal_cdf(-2.0) - 0.0228) < 0.01


def test_posterior_a_better_when_a_consistently_smaller():
    """A's BS uniformly smaller → posterior should be ~1.0."""
    bs_a = [0.05] * 50
    bs_b = [0.10] * 50
    p = posterior_a_better_than_b(bs_a, bs_b, n_bootstrap=1000)
    assert p > 0.99


def test_posterior_a_better_when_a_consistently_larger():
    """A's BS uniformly larger → posterior should be ~0.0."""
    bs_a = [0.30] * 50
    bs_b = [0.05] * 50
    p = posterior_a_better_than_b(bs_a, bs_b, n_bootstrap=1000)
    assert p < 0.01


def test_posterior_a_better_tied():
    """Identical BS arrays → posterior at 0.5 (each bootstrap is a tie, half-credit)."""
    bs_a = [0.1] * 30
    bs_b = [0.1] * 30
    p = posterior_a_better_than_b(bs_a, bs_b, n_bootstrap=2000)
    assert 0.49 <= p <= 0.51


def test_posterior_normal_fit_matches_bootstrap_on_large_sample():
    """For N=200 with Gaussian-ish deltas, both methods agree to ±0.05."""
    random.seed(101)
    bs_a = [0.10 + random.gauss(0, 0.05) for _ in range(200)]
    bs_b = [0.15 + random.gauss(0, 0.05) for _ in range(200)]
    p_normal = posterior_normal_fit(bs_a, bs_b)
    p_boot = posterior_a_better_than_b(bs_a, bs_b, n_bootstrap=2000)
    assert abs(p_normal - p_boot) < 0.05


def test_posterior_validates_lengths():
    with pytest.raises(ValueError):
        posterior_a_better_than_b([0.1], [0.1, 0.2])
    with pytest.raises(ValueError):
        posterior_normal_fit([0.1], [0.1, 0.2])


# --------------------------------------------------------------------------- #
# Pairwise orchestrator
# --------------------------------------------------------------------------- #


def test_pairwise_paired_bootstrap_basic():
    """3 models → C(3, 2) = 3 ordered pairs."""
    bs = {
        "model_a": {"q1": 0.10, "q2": 0.20, "q3": 0.30},
        "model_b": {"q1": 0.05, "q2": 0.15, "q3": 0.25},
        "model_c": {"q1": 0.30, "q2": 0.30, "q3": 0.30},
    }
    pairs = pairwise_paired_bootstrap(bs, n_bootstrap=500)
    assert len(pairs) == 3
    # All ordered (a<b<c lexicographically)
    assert pairs[0].model_a == "model_a" and pairs[0].model_b == "model_b"
    assert pairs[1].model_a == "model_a" and pairs[1].model_b == "model_c"
    assert pairs[2].model_a == "model_b" and pairs[2].model_b == "model_c"
    # Holm-adjusted p-values populated
    for p in pairs:
        assert p.p_holm is not None
        assert 0 <= p.p_holm <= 1


def test_pairwise_paired_bootstrap_intersection_only():
    """Only questions that both models answered enter the comparison."""
    bs = {
        "model_a": {"q1": 0.10, "q2": 0.20},
        "model_b": {"q2": 0.15, "q3": 0.25},
    }
    pairs = pairwise_paired_bootstrap(bs, n_bootstrap=500)
    assert len(pairs) == 1
    # Common questions: only {q2}
    assert pairs[0].n_questions == 1


def test_pairwise_paired_bootstrap_empty_input():
    assert pairwise_paired_bootstrap({}) == []


# --------------------------------------------------------------------------- #
# v5 multi-metric paired bootstrap
# --------------------------------------------------------------------------- #


def _make_sample(
    *,
    model: str,
    question_id: str,
    sample_idx: int = 0,
    parsed: frozenset[str] | None = None,
    options: list[str] | None = None,
    correct: int | None = 1,
    parse_ok: int = 1,
    choice_type: str = "single",
    probabilities: list[float] | None = None,
):
    """Construct a SampleRow with v5-friendly defaults."""
    import json as _json

    from forecast_eval.analysis.flatten import SampleRow as _SR

    if options is None:
        options = ["A", "B", "C", "D"]
    final = _json.dumps(sorted(parsed)) if parsed is not None else None
    return _SR(
        model=model,
        question_id=question_id,
        question_type="single",
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
        final_answer_letters=final,
        error=None,
        created_at="2026-04-26T00:00:00Z",
        finish_reason="stop",
        nudges_used=0,
        belief_final=None,
        belief_trace=None,
        belief_parse_ok=0,
        probabilities=probabilities,
        is_fallback=False,
    )


def _build_two_model_fixture(
    *,
    n_questions: int,
    K: int,
    a_correct_per_q: list[int],
    b_correct_per_q: list[int],
):
    """Build (samples_a_by_q, samples_b_by_q, gt_map): each model gets
    `correct_per_q` correct trials out of K on each question."""
    samples_a_by_q: dict = {}
    samples_b_by_q: dict = {}
    gt_map: dict = {}
    for qi in range(n_questions):
        qid = f"q{qi}"
        gt = frozenset({"A"})
        gt_map[qid] = gt
        a_correct = a_correct_per_q[qi]
        b_correct = b_correct_per_q[qi]
        a_list = []
        b_list = []
        for k in range(K):
            a_letter = "A" if k < a_correct else "B"
            b_letter = "A" if k < b_correct else "B"
            a_list.append(_make_sample(
                model="model_a", question_id=qid, sample_idx=k,
                parsed=frozenset({a_letter}),
                correct=1 if a_letter == "A" else 0,
            ))
            b_list.append(_make_sample(
                model="model_b", question_id=qid, sample_idx=k,
                parsed=frozenset({b_letter}),
                correct=1 if b_letter == "A" else 0,
            ))
        samples_a_by_q[qid] = a_list
        samples_b_by_q[qid] = b_list
    return samples_a_by_q, samples_b_by_q, gt_map


def test_metric_paired_bootstrap_fss_strong_difference():
    """Spec scenario: fixture A FSS=high / B FSS=low → p < 0.05, ΔFSS > 0."""
    from forecast_eval.analysis.inference import (
        DEFAULT_METRIC_FNS,
        metric_paired_bootstrap,
    )

    n_q = 50
    # A: 5/5 correct on every question (perfect FSS=1.0).
    # B: 0/5 correct on every question (FSS = (0 - 0.25) / 0.75 = -0.333).
    a_corr = [5] * n_q
    b_corr = [0] * n_q
    a, b, gt = _build_two_model_fixture(
        n_questions=n_q, K=5, a_correct_per_q=a_corr, b_correct_per_q=b_corr,
    )
    res = metric_paired_bootstrap(
        DEFAULT_METRIC_FNS["fss"], a, b, gt,
        metric_name="fss", model_a="A", model_b="B",
        n_bootstrap=1000,
    )
    assert res is not None
    assert res.delta_mean > 1.0  # 1.0 - (-0.333) ≈ 1.333
    assert res.p_two_sided < 0.01
    assert res.ci_low > 0  # CI excludes 0 → significant
    assert res.cohens_d > 0


def test_metric_paired_bootstrap_fss_no_difference():
    """A and B identical → ΔFSS ≈ 0, p high, |d| small."""
    from forecast_eval.analysis.inference import (
        DEFAULT_METRIC_FNS,
        metric_paired_bootstrap,
    )

    n_q = 50
    same_corr = [3] * n_q
    a, b, gt = _build_two_model_fixture(
        n_questions=n_q, K=5,
        a_correct_per_q=same_corr, b_correct_per_q=list(same_corr),
    )
    res = metric_paired_bootstrap(
        DEFAULT_METRIC_FNS["fss"], a, b, gt,
        metric_name="fss", model_a="A", model_b="B",
        n_bootstrap=1000,
    )
    assert res is not None
    assert abs(res.delta_mean) < 1e-6
    assert res.p_two_sided > 0.5
    assert abs(res.cohens_d) < 1e-3


def test_metric_paired_bootstrap_pairing_property():
    """Paired property: when A and B are POSITIVELY correlated per-question
    (both tend to get the same questions right or wrong), the paired
    bootstrap CI on ΔFSS is NARROWER than the de-paired version.

    This is the key invariant of paired bootstrap: variance of (A-B) under
    paired sampling equals Var(A) + Var(B) - 2 Cov(A, B). For positively
    correlated A and B, the -2 Cov term tightens the CI vs an independent
    bootstrap (which sees Var(A) + Var(B)).

    We simulate "independent" via a deterministic shuffle of A's question
    labels — that breaks the correlation while keeping the marginal
    distributions identical.
    """
    from forecast_eval.analysis.inference import (
        DEFAULT_METRIC_FNS,
        metric_paired_bootstrap,
    )

    n_q = 30
    # Mostly correlated: half the questions both nail (5/5), half both miss
    # (1/5 vs 0/5 — small constant offset, so positive Cov dominates).
    a_corr = [5 if i % 2 == 0 else 1 for i in range(n_q)]
    b_corr = [5 if i % 2 == 0 else 0 for i in range(n_q)]
    a, b, gt = _build_two_model_fixture(
        n_questions=n_q, K=5, a_correct_per_q=a_corr, b_correct_per_q=b_corr,
    )
    paired = metric_paired_bootstrap(
        DEFAULT_METRIC_FNS["fss"], a, b, gt,
        metric_name="fss", model_a="A", model_b="B",
        n_bootstrap=1000, seed=1,
    )
    # Shuffle A's question_id labels via dict reshuffle.
    import random as _r
    qids = list(a.keys())
    rng = _r.Random(99)
    shuffled_qids = qids.copy()
    rng.shuffle(shuffled_qids)
    a_shuffled = {orig: a[shuf] for orig, shuf in zip(qids, shuffled_qids)}
    shuffled = metric_paired_bootstrap(
        DEFAULT_METRIC_FNS["fss"], a_shuffled, b, gt,
        metric_name="fss", model_a="A", model_b="B",
        n_bootstrap=1000, seed=1,
    )
    paired_width = paired.ci_high - paired.ci_low
    shuffled_width = shuffled.ci_high - shuffled.ci_low
    # Positive correlation → paired CI narrower (Var(A-B) gets the -2Cov term).
    assert paired_width <= shuffled_width


def test_pairwise_metric_bootstrap_default_metrics():
    """5 metrics × C(3,2) = 3 pairs; ebi will return None (no probabilities),
    so expect at least 12 (4 metrics × 3 pairs) results."""
    from forecast_eval.analysis.inference import (
        DEFAULT_METRIC_FNS,
        pairwise_metric_bootstrap,
    )

    n_q = 20
    samples_by_model_by_q: dict = {}
    gt_map: dict = {}
    for qi in range(n_q):
        qid = f"q{qi}"
        gt_map[qid] = frozenset({"A"})
    for model_idx, model in enumerate(["model_x", "model_y", "model_z"]):
        per_q: dict = {}
        for qi in range(n_q):
            qid = f"q{qi}"
            samples = []
            for k in range(5):
                # Each model has different correctness pattern.
                letter = "A" if (k + model_idx) % 3 != 0 else "B"
                samples.append(_make_sample(
                    model=model, question_id=qid, sample_idx=k,
                    parsed=frozenset({letter}),
                    correct=1 if letter == "A" else 0,
                ))
            per_q[qid] = samples
        samples_by_model_by_q[model] = per_q

    results = pairwise_metric_bootstrap(
        samples_by_model_by_q, gt_map, DEFAULT_METRIC_FNS,
        n_bootstrap=200,
    )
    assert len(results) >= 12
    metric_names = {r.metric_name for r in results}
    assert {"fss", "acc", "mv_acc", "fleiss_kappa"}.issubset(metric_names)
    # Sorted by (metric_name, model_a, model_b).
    keys = [(r.metric_name, r.model_a, r.model_b) for r in results]
    assert keys == sorted(keys)


def test_metric_paired_bootstrap_ebi_skips_when_no_probabilities():
    """v3-style fixture (probabilities=None) → ebi metric returns None."""
    from forecast_eval.analysis.inference import (
        DEFAULT_METRIC_FNS,
        metric_paired_bootstrap,
    )

    n_q = 10
    a, b, gt = _build_two_model_fixture(
        n_questions=n_q, K=5,
        a_correct_per_q=[3] * n_q, b_correct_per_q=[2] * n_q,
    )
    res = metric_paired_bootstrap(
        DEFAULT_METRIC_FNS["ebi"], a, b, gt,
        metric_name="ebi", model_a="A", model_b="B",
        n_bootstrap=200,
    )
    assert res is None


def test_metric_paired_bootstrap_fleiss_skips_when_k1():
    """K=1 fixture → fleiss_kappa returns None → bootstrap returns None."""
    from forecast_eval.analysis.inference import (
        DEFAULT_METRIC_FNS,
        metric_paired_bootstrap,
    )

    n_q = 10
    a, b, gt = _build_two_model_fixture(
        n_questions=n_q, K=1,
        a_correct_per_q=[1] * n_q, b_correct_per_q=[0] * n_q,
    )
    res = metric_paired_bootstrap(
        DEFAULT_METRIC_FNS["fleiss_kappa"], a, b, gt,
        metric_name="fleiss_kappa", model_a="A", model_b="B",
        n_bootstrap=200,
    )
    assert res is None


def test_cohens_d_known_values():
    """Spec scenario: Δ = [0.10, 0.12, 0.08, 0.11, 0.09] →
    mean=0.10, std≈0.0158, d ≈ 6.32."""
    from forecast_eval.analysis.inference import cohens_d_from_bootstrap

    deltas = [0.10, 0.12, 0.08, 0.11, 0.09]
    d = cohens_d_from_bootstrap(deltas)
    assert d == pytest.approx(6.32, abs=1e-1)


def test_cohens_d_zero_delta_returns_zero():
    """All deltas zero → d = 0 (no effect, no spread)."""
    from forecast_eval.analysis.inference import cohens_d_from_bootstrap

    assert cohens_d_from_bootstrap([0.0, 0.0, 0.0, 0.0]) == 0.0


def test_cohens_d_empty_returns_zero():
    """Empty input → d = 0 (no data)."""
    from forecast_eval.analysis.inference import cohens_d_from_bootstrap

    assert cohens_d_from_bootstrap([]) == 0.0


def test_metric_paired_bootstrap_empty_common_returns_none():
    """A and B share zero questions → None."""
    from forecast_eval.analysis.inference import (
        DEFAULT_METRIC_FNS,
        metric_paired_bootstrap,
    )

    a = {"q0": [_make_sample(model="A", question_id="q0", parsed=frozenset({"A"}))]}
    b = {"q1": [_make_sample(model="B", question_id="q1", parsed=frozenset({"A"}))]}
    gt = {"q0": frozenset({"A"}), "q1": frozenset({"A"})}
    res = metric_paired_bootstrap(
        DEFAULT_METRIC_FNS["fss"], a, b, gt,
        metric_name="fss", n_bootstrap=200,
    )
    assert res is None
