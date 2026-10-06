"""Unit tests for `forecast_eval.analysis.proper_score`.

These pin the mathematical contracts:
formulas (BS lab vs dec, BI mean-then-sqrt, NLL clip, MBS single-only) plus
the ABI sign convention (negative ABS bumps ABI above 100).
"""
from __future__ import annotations

import math

import pytest

from forecast_eval.analysis.proper_score import (
    NLL_EPS,
    aggregate_probabilistic,
    brier_index,
    brier_score_dec,
    brier_score_lab,
    compute_abi,
    crowd_gamma_for,
    mbs,
    nll,
    per_question_scores_for,
    uniform_gamma_for,
)


# ---------- Brier scores -----------------------------------------------------


def test_brier_score_lab_perfect_yes_no() -> None:
    """yes_no perfect prediction → BS_lab = 0, BI = 100."""
    p = [1.0, 0.0]
    o = [1, 0]
    assert brier_score_lab(p, o) == pytest.approx(0.0, abs=1e-12)
    assert brier_index([brier_score_lab(p, o)]) == pytest.approx(100.0, abs=1e-12)


def test_brier_score_lab_50_50_baseline() -> None:
    """Spec scenario: 50/50 baseline → BS_lab = 0.25 per question, BI = 50."""
    p = [0.5, 0.5]
    o = [1, 0]
    bs = brier_score_lab(p, o)
    assert bs == pytest.approx(0.25, rel=1e-9)
    # Aggregate over 100 questions all the same → still 0.25 → BI = 50
    bi = brier_index([bs] * 100)
    assert bi == pytest.approx(50.0, rel=1e-9)


def test_brier_score_dec_equals_k_times_lab_single() -> None:
    """For any single question: BS_dec = k * BS_lab (numerical parity 1e-9)."""
    cases = [
        ([0.7, 0.3], [1, 0]),  # yes_no
        ([0.6, 0.3, 0.1], [1, 0, 0]),  # 3-choice single
        ([0.4, 0.3, 0.2, 0.1], [0, 1, 0, 0]),  # 4-choice single
    ]
    for p, o in cases:
        k = len(p)
        assert brier_score_dec(p, o) == pytest.approx(k * brier_score_lab(p, o), abs=1e-9)


def test_brier_score_lab_multi_choice() -> None:
    """multi: BS_lab averages 4 squared deviations.

    answer={A,C}, k=4, p=(0.8, 0.2, 0.6, 0.1)
      o = (1, 0, 1, 0)
      sq = (0.04, 0.04, 0.16, 0.01) -> sum=0.25 -> /4 = 0.0625
    """
    p = [0.8, 0.2, 0.6, 0.1]
    o = [1, 0, 1, 0]
    assert brier_score_lab(p, o) == pytest.approx(0.0625, rel=1e-9)


def test_brier_score_input_validation() -> None:
    with pytest.raises(ValueError):
        brier_score_lab([0.5, 0.5], [1, 0, 0])  # length mismatch
    with pytest.raises(ValueError):
        brier_score_lab([0.5, 1.5], [1, 0])  # prob > 1
    with pytest.raises(ValueError):
        brier_score_lab([0.5, 0.5], [1, 2])  # obs not 0/1


# ---------- NLL --------------------------------------------------------------


def test_nll_clip_prevents_log_zero_single() -> None:
    """Spec scenario: p_lstar = 0 → clip to ε, NLL = -log(ε) ≈ 6.91."""
    p = [0.0, 1.0]
    o = [1, 0]
    n = nll(p, o, "single")
    assert n == pytest.approx(-math.log(NLL_EPS), rel=1e-9)
    assert math.isfinite(n)


def test_nll_clip_prevents_log_one_multi() -> None:
    """multi: BCE — p=1 on a positive label clips to 1-ε so log(1-p) finite."""
    p = [1.0, 0.0]
    o = [0, 1]  # label 1 is positive, p[1]=0 is wrong → after clip log(ε)
    n = nll(p, o, "multi")
    assert math.isfinite(n)


def test_nll_perfect_single_near_zero() -> None:
    """Perfect prediction p_lstar = 1 → after clip to 1-ε, NLL ≈ -log(0.999)."""
    p = [1.0, 0.0]
    o = [1, 0]
    expected = -math.log(1.0 - NLL_EPS)
    assert nll(p, o, "single") == pytest.approx(expected, rel=1e-9)


def test_nll_multi_label_wise_formula() -> None:
    """Multi BCE: -(1/k) * Σ [o*log(p) + (1-o)*log(1-p)]."""
    p = [0.8, 0.2, 0.6, 0.1]
    o = [1, 0, 1, 0]
    expected = -(
        (1 * math.log(0.8) + 0 * math.log(0.2))
        + (0 * math.log(0.2) + 1 * math.log(0.8))  # 1-p=0.8 here
        + (1 * math.log(0.6) + 0 * math.log(0.4))
        + (0 * math.log(0.1) + 1 * math.log(0.9))  # 1-p=0.9 here
    ) / 4
    assert nll(p, o, "multi") == pytest.approx(expected, rel=1e-9)


# ---------- MBS --------------------------------------------------------------


def test_mbs_single_perfect() -> None:
    """Perfect single prediction MBS ≈ 100."""
    p = [1.0, 0.0]
    o = [1, 0]
    expected = 100.0 * (math.log2(1.0 - NLL_EPS) + 1.0)
    assert mbs(p, o, "single") == pytest.approx(expected, rel=1e-9)
    assert mbs(p, o, "single") == pytest.approx(100.0, abs=0.5)  # within 0.5 of 100


def test_mbs_multi_returns_none() -> None:
    """Spec: multi question type MUST NOT emit MBS; CSV column written as NULL."""
    assert mbs([0.5, 0.5, 0.5], [1, 0, 1], "multi") is None


# ---------- Brier Index ------------------------------------------------------


def test_brier_index_mean_then_sqrt() -> None:
    """Spec: square root MUST be taken AFTER averaging, MUST NOT take per-question root first then average."""
    bs_values = [0.04, 0.16]  # √mean = √0.10 ≈ 0.316; mean √ = (0.2+0.4)/2 = 0.3
    bi_correct = 100.0 * (1.0 - math.sqrt((0.04 + 0.16) / 2))
    bi_wrong = 100.0 * (1.0 - (math.sqrt(0.04) + math.sqrt(0.16)) / 2)
    assert brier_index(bs_values) == pytest.approx(bi_correct, rel=1e-9)
    assert brier_index(bs_values) != pytest.approx(bi_wrong, rel=1e-9)


def test_brier_index_empty_returns_none() -> None:
    assert brier_index([]) is None


# ---------- ABI sign convention ----------------------------------------------


def test_compute_abi_zero_avg_means_100() -> None:
    """Model exactly matches baseline → ABS = 0 → ABI = 100."""
    assert compute_abi([0.0, 0.0, 0.0]) == pytest.approx(100.0, abs=1e-12)


def test_compute_abi_negative_avg_above_100() -> None:
    """Spec scenario: ABS=-0.05 → ABI = 100(1 + √0.05) ≈ 122.36."""
    abi = compute_abi([-0.05])
    assert abi is not None
    assert abi == pytest.approx(100.0 * (1.0 + math.sqrt(0.05)), rel=1e-9)
    assert abi > 100.0


def test_compute_abi_positive_avg_below_100() -> None:
    """Worse than baseline → ABS positive → ABI < 100."""
    abi = compute_abi([0.04])
    assert abi is not None
    assert abi == pytest.approx(100.0 * (1.0 - math.sqrt(0.04)), rel=1e-9)
    assert abi < 100.0


# ---------- Baselines (γ_q) --------------------------------------------------


def test_uniform_gamma_yes_no() -> None:
    """k=2, o=(1,0), uniform p=(0.5,0.5) → γ = (0.25+0.25)/2 = 0.25."""
    assert uniform_gamma_for([1, 0]) == pytest.approx(0.25, rel=1e-9)


def test_uniform_gamma_4_choice_single() -> None:
    """k=4, o=(1,0,0,0), uniform p=(0.25,0.25,0.25,0.25)
       → squared = (0.5625, 0.0625, 0.0625, 0.0625) → /4 = 0.1875."""
    assert uniform_gamma_for([1, 0, 0, 0]) == pytest.approx(0.1875, rel=1e-9)


def test_crowd_gamma_excludes_self() -> None:
    """3-model fixture: A=(0.9,0.1) B=(0.7,0.3) C=(0.6,0.4), o=(1,0).
    Computing γ for A → average of B and C = (0.65, 0.35)
    → squared = (0.1225, 0.1225) → /2 = 0.1225.
    """
    obs = [1, 0]
    crowd_for_a = crowd_gamma_for(obs, [[0.7, 0.3], [0.6, 0.4]])
    assert crowd_for_a == pytest.approx(0.1225, rel=1e-9)


def test_crowd_gamma_empty_other_models_returns_none() -> None:
    """Single-model run → no other_models_probs → caller falls back to uniform."""
    assert crowd_gamma_for([1, 0], []) is None


# ---------- Aggregator -------------------------------------------------------


def test_aggregate_single_model_abi_crowd_equals_uniform() -> None:
    """Spec scenario: single-model run → ABI_crowd MUST equal ABI_uniform."""
    pq1 = per_question_scores_for(
        question_id="q1", choice_type="single",
        probs=[0.7, 0.3], obs=[1, 0],
    )
    pq2 = per_question_scores_for(
        question_id="q2", choice_type="single",
        probs=[0.5, 0.5], obs=[0, 1],
    )
    uniform = {"q1": uniform_gamma_for([1, 0]), "q2": uniform_gamma_for([0, 1])}
    # No crowd_gammas → expect abi_crowd to fall back to abi_uniform.
    agg = aggregate_probabilistic([pq1, pq2], uniform_gammas=uniform, crowd_gammas=None)
    assert agg.abi_crowd == agg.abi_uniform
    assert agg.abi_crowd is not None
    # Sanity: BI not None and within plausible range.
    assert agg.bi is not None
    assert 0.0 <= agg.bi <= 100.0
    assert agg.fallback_share == pytest.approx(0.0, abs=1e-12)


def test_aggregate_three_model_crowd_no_self_reference() -> None:
    """Hand-computed 3-model fixture. Model A's ABI MUST exclude A from baseline."""
    # Two questions, both yes_no, GT=A.
    obs = [1, 0]
    pq_a_q1 = per_question_scores_for(question_id="q1", choice_type="single", probs=[0.9, 0.1], obs=obs)
    pq_a_q2 = per_question_scores_for(question_id="q2", choice_type="single", probs=[0.8, 0.2], obs=obs)
    # Other models on each question.
    other_q1 = [[0.7, 0.3], [0.6, 0.4]]  # B, C
    other_q2 = [[0.5, 0.5], [0.4, 0.6]]  # B, C
    crowd = {
        "q1": crowd_gamma_for(obs, other_q1),
        "q2": crowd_gamma_for(obs, other_q2),
    }
    uniform = {"q1": uniform_gamma_for(obs), "q2": uniform_gamma_for(obs)}
    agg = aggregate_probabilistic([pq_a_q1, pq_a_q2], crowd_gammas=crowd, uniform_gammas=uniform)
    # Crowd γ_q1 = average of (0.7,0.3) and (0.6,0.4) = (0.65,0.35) → ((1-.65)^2+(0-.35)^2)/2
    # = (0.1225+0.1225)/2 = 0.1225.
    assert crowd["q1"] == pytest.approx(0.1225, rel=1e-9)
    # Model A BS_q1 = ((1-.9)^2 + (0-.1)^2)/2 = (0.01+0.01)/2 = 0.01
    # ABS_q1 = 0.01 - 0.1225 = -0.1125 (model A beats crowd)
    assert pq_a_q1.bs_lab == pytest.approx(0.01, rel=1e-9)
    # ABI sign convention: negative ABS → ABI > 100.
    assert agg.abi_crowd is not None
    assert agg.abi_crowd > 100.0


def test_aggregate_fallback_share_tracking() -> None:
    """fallback_share = is_fallback count / total questions."""
    pq1 = per_question_scores_for(
        question_id="q1", choice_type="single", probs=[0.7, 0.3], obs=[1, 0], is_fallback=False,
    )
    pq2 = per_question_scores_for(
        question_id="q2", choice_type="single", probs=[0.95, 0.05], obs=[1, 0], is_fallback=True,
    )
    pq3 = per_question_scores_for(
        question_id="q3", choice_type="single", probs=[0.95, 0.05], obs=[1, 0], is_fallback=True,
    )
    agg = aggregate_probabilistic([pq1, pq2, pq3])
    assert agg.fallback_share == pytest.approx(2 / 3, rel=1e-9)
    assert agg.n_fallback == 2
    assert agg.n_questions == 3


def test_aggregate_empty_returns_all_none() -> None:
    agg = aggregate_probabilistic([])
    assert agg.bi is None
    assert agg.nll is None
    assert agg.mbs is None
    assert agg.abi_crowd is None
    assert agg.abi_uniform is None
    assert agg.n_questions == 0


def test_per_question_scores_multi_no_dec_no_mbs() -> None:
    """multi question type: bs_dec MUST be None; mbs MUST be None."""
    pq = per_question_scores_for(
        question_id="q_multi", choice_type="multi",
        probs=[0.8, 0.2, 0.6, 0.1], obs=[1, 0, 1, 0],
    )
    assert pq.bs_dec is None
    assert pq.mbs is None
    assert pq.bs_lab > 0.0  # but bs_lab still defined


def test_per_question_scores_single_has_dec_and_mbs() -> None:
    """single question type: bs_dec = k * bs_lab; mbs is a real number."""
    pq = per_question_scores_for(
        question_id="q_single", choice_type="single",
        probs=[0.7, 0.3], obs=[1, 0],
    )
    assert pq.bs_dec is not None
    assert pq.bs_dec == pytest.approx(2 * pq.bs_lab, abs=1e-9)
    assert pq.mbs is not None
    assert isinstance(pq.mbs, float)
