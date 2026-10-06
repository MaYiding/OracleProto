"""Proper scoring rules for the probabilistic family.

The probability scoring contracts are:

* `brier_score_lab` / `brier_score_dec` — label-wise vs decision-wise Brier.
* `nll` — log-loss with $\\epsilon$ clip; per-question dispatch on choice_type.
* `mbs` — Metaculus Baseline Score (single-only; multi returns None).
* `brier_index` — aggregate $100(1 - \\sqrt{\\overline{BS}})$, **mean THEN sqrt**.
* `compute_abi` — adjusted Brier Index with the following sign convention
  ($\\overline{ABS} \\ge 0$ → $100(1-\\sqrt{\\cdot})$;
   $\\overline{ABS} < 0$ → $100(1+\\sqrt{|\\cdot|})$, model beats baseline).
* `crowd_baseline_gamma_per_q` / `uniform_baseline_gamma_per_q` — per-question
  baselines used by `compute_abi` to subtract off question difficulty.

All functions take plain Python lists; no numpy / scipy dependency. Inputs
are validated on the way in — passing a probability outside $[0, 1]$ or a
non-binary observation raises `ValueError` so the caller can't silently
poison aggregates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

# Clip floor for log-domain operations. The spec pins $\epsilon = 10^{-3}$
# so $\log p \in [-6.91, 0]$ and $\mathrm{NLL}$ stays finite.
NLL_EPS: float = 1e-3


def _is_single(choice_type: str) -> bool:
    """`choice_type` is `"single"` for yes_no / binary_named / single mc;
    everything else (notably `"multi"`) is treated as multi-label."""
    return choice_type == "single"


def _check_vectors(p: list[float], o: list[int]) -> None:
    if len(p) != len(o):
        raise ValueError(
            f"probability and observation vectors must have equal length; "
            f"got len(p)={len(p)} len(o)={len(o)}"
        )
    if not p:
        raise ValueError("probability vector must be non-empty")
    for v in p:
        if not (0.0 - 1e-9 <= v <= 1.0 + 1e-9):
            raise ValueError(f"probability {v!r} outside [0, 1]")
    for v in o:
        if v not in (0, 1):
            raise ValueError(f"observation {v!r} not in {{0, 1}}")


def brier_score_lab(p: list[float], o: list[int]) -> float:
    """Label-wise Brier score: $\\frac{1}{k}\\sum_l (p_l - o_l)^2$.

    Defined for every choice_type; cross-task comparable.
    """
    _check_vectors(p, o)
    k = len(p)
    return sum((float(pi) - oi) ** 2 for pi, oi in zip(p, o)) / k


def brier_score_dec(p: list[float], o: list[int]) -> float:
    """Decision-wise Brier score: $\\sum_l (p_l - o_l)^2 = k \\cdot$ label-wise.

    Caller should restrict to single-choice questions (the value still has
    arithmetic meaning for multi but the comparability with paper §A.2 is
    only well-defined for single).
    """
    _check_vectors(p, o)
    return sum((float(pi) - oi) ** 2 for pi, oi in zip(p, o))


def nll(p: list[float], o: list[int], choice_type: str, eps: float = NLL_EPS) -> float:
    """Negative log-likelihood, clipped to $[\\epsilon, 1 - \\epsilon]$ before $\\log$.

    * `single` (yes_no, binary_named, multiple_choice/single):
      $-\\log p_{l^*}$ where $l^* = \\arg\\max_l o_l$.
    * `multi`: label-wise binary cross-entropy
      $-\\frac{1}{k}\\sum_l [o_l \\log p_l + (1-o_l)\\log(1-p_l)]$.
    """
    _check_vectors(p, o)
    p_clipped = [min(max(float(v), eps), 1.0 - eps) for v in p]
    if _is_single(choice_type):
        # Find the single positive index. If none (degenerate question), fall
        # back to label-wise — same convention multi uses, keeps NLL finite.
        try:
            l_star = next(i for i, v in enumerate(o) if v == 1)
        except StopIteration:
            return _label_wise_nll(p_clipped, o)
        return -math.log(p_clipped[l_star])
    return _label_wise_nll(p_clipped, o)


def _label_wise_nll(p_clipped: list[float], o: list[int]) -> float:
    k = len(p_clipped)
    total = 0.0
    for pi, oi in zip(p_clipped, o):
        total += oi * math.log(pi) + (1 - oi) * math.log(1.0 - pi)
    return -total / k


def mbs(p: list[float], o: list[int], choice_type: str, eps: float = NLL_EPS) -> float | None:
    """Metaculus Baseline Score: $100(\\log_2 p_{l^*} + 1)$. Single-only; multi returns None."""
    _check_vectors(p, o)
    if not _is_single(choice_type):
        return None
    try:
        l_star = next(i for i, v in enumerate(o) if v == 1)
    except StopIteration:
        return None
    p_lstar = min(max(float(p[l_star]), eps), 1.0 - eps)
    return 100.0 * (math.log2(p_lstar) + 1.0)


def brier_index(per_question_bs: Iterable[float]) -> float | None:
    """$100\\bigl(1 - \\sqrt{\\overline{BS}}\\bigr)$. Mean **then** sqrt (paper §A.2)."""
    values = [float(v) for v in per_question_bs]
    if not values:
        return None
    avg = sum(values) / len(values)
    if avg < 0.0:
        # Numerically negative average BS shouldn't happen (BS ≥ 0), but if a
        # caller passes ABS by mistake, we forward to compute_abi's sign rule.
        return 100.0 * (1.0 + math.sqrt(-avg))
    return 100.0 * (1.0 - math.sqrt(avg))


def compute_abi(per_question_abs: Iterable[float]) -> float | None:
    """Adjusted Brier Index with the following sign convention.

    $\\overline{ABS} = \\frac{1}{N}\\sum_q ABS_q$ where $ABS_q = BS_q - \\gamma_q$.

    * If $\\overline{ABS} \\ge 0$ (model worse than or equal to baseline on
      average): $\\mathrm{ABI} = 100\\bigl(1 - \\sqrt{\\overline{ABS}}\\bigr)$.
    * If $\\overline{ABS} < 0$ (model beats baseline on average):
      $\\mathrm{ABI} = 100\\bigl(1 + \\sqrt{|\\overline{ABS}|}\\bigr)$,
      keeping the metric monotone so "more negative ABS → higher ABI".
    """
    values = [float(v) for v in per_question_abs]
    if not values:
        return None
    avg = sum(values) / len(values)
    if avg >= 0.0:
        return 100.0 * (1.0 - math.sqrt(avg))
    return 100.0 * (1.0 + math.sqrt(-avg))


@dataclass(frozen=True)
class PerQuestionScore:
    """Bundle of per-question proper scores. `bs_dec` and `mbs` are None on multi."""

    question_id: str
    choice_type: str
    k: int
    bs_lab: float
    bs_dec: float | None
    nll: float
    mbs: float | None
    probs: list[float]
    obs: list[int]
    is_fallback: bool


def per_question_scores_for(
    *,
    question_id: str,
    choice_type: str,
    probs: list[float],
    obs: list[int],
    is_fallback: bool = False,
) -> PerQuestionScore:
    """Compute the per-question score bundle in one shot.

    Centralising this guarantees `bs_dec` and `mbs` are gated identically on
    `choice_type` everywhere; `flatten.py` callers don't need to remember the
    per-metric branching.
    """
    bs_lab = brier_score_lab(probs, obs)
    if _is_single(choice_type):
        bs_dec: float | None = brier_score_dec(probs, obs)
    else:
        bs_dec = None
    nll_v = nll(probs, obs, choice_type)
    mbs_v = mbs(probs, obs, choice_type)
    return PerQuestionScore(
        question_id=question_id,
        choice_type=choice_type,
        k=len(probs),
        bs_lab=bs_lab,
        bs_dec=bs_dec,
        nll=nll_v,
        mbs=mbs_v,
        probs=probs,
        obs=obs,
        is_fallback=is_fallback,
    )


def uniform_gamma_for(obs: list[int]) -> float:
    """Per-question $\\gamma_q$ when the baseline is the uniform prior $\\mathbf{p} = (1/k, ..., 1/k)$.

    Reduces to $\\frac{1}{k}\\sum_l (1/k - o_l)^2$ — closed form depending only on
    $k$ and the count of positive labels.
    """
    if not obs:
        return 0.0
    k = len(obs)
    inv = 1.0 / k
    return sum((inv - oi) ** 2 for oi in obs) / k


def crowd_gamma_for(
    obs: list[int],
    other_models_probs: list[list[float]],
) -> float | None:
    """Per-question $\\gamma_q$ on the leave-one-out model-crowd baseline.

    `other_models_probs` MUST exclude the model whose ABI is being computed,
    so the average reflects "what other models thought" rather than self-
    referencing. Returns None if no other model produced a probability vector
    on this question — caller should fall back to `uniform_gamma_for`.
    """
    if not other_models_probs:
        return None
    k = len(obs)
    sums = [0.0] * k
    n = len(other_models_probs)
    for vec in other_models_probs:
        if len(vec) != k:
            raise ValueError(
                f"crowd vector length {len(vec)} != observation length {k}"
            )
        for i, v in enumerate(vec):
            sums[i] += float(v)
    avg = [s / n for s in sums]
    return sum((avg[i] - obs[i]) ** 2 for i in range(k)) / k


@dataclass(frozen=True)
class ModelProbabilisticAggregate:
    """Aggregate probabilistic scores for a single model over its eligible
    per-question scores. None values mean "no eligible probability vectors";
    the writer translates that into NULL on the CSV row."""

    n_questions: int
    n_fallback: int
    fallback_share: float | None
    bi: float | None
    bi_dec: float | None
    nll: float | None
    mbs: float | None
    abi_crowd: float | None
    abi_uniform: float | None


def aggregate_probabilistic(
    per_q: list[PerQuestionScore],
    *,
    crowd_gammas: dict[str, float] | None = None,
    uniform_gammas: dict[str, float] | None = None,
) -> ModelProbabilisticAggregate:
    """Aggregate per-question scores into the row-level probabilistic columns.

    `crowd_gammas` / `uniform_gammas` map question_id → $\\gamma_q$. Missing
    keys cause that question to be skipped from the corresponding ABI; this
    is what the spec calls "if only 1 model evaluated, ABI degrades to
    uniform" — for that model, `crowd_gammas` is empty / None and we fall
    back to `abi_uniform` for both columns.
    """
    if not per_q:
        return ModelProbabilisticAggregate(
            n_questions=0,
            n_fallback=0,
            fallback_share=None,
            bi=None,
            bi_dec=None,
            nll=None,
            mbs=None,
            abi_crowd=None,
            abi_uniform=None,
        )
    n_total = len(per_q)
    n_fallback = sum(1 for s in per_q if s.is_fallback)
    fallback_share = n_fallback / n_total if n_total else None

    bs_lab_values = [s.bs_lab for s in per_q]
    bs_dec_values = [s.bs_dec for s in per_q if s.bs_dec is not None]
    nll_values = [s.nll for s in per_q]
    mbs_values = [s.mbs for s in per_q if s.mbs is not None]

    bi = brier_index(bs_lab_values)
    bi_dec = brier_index(bs_dec_values) if bs_dec_values else None
    avg_nll = sum(nll_values) / len(nll_values) if nll_values else None
    avg_mbs = sum(mbs_values) / len(mbs_values) if mbs_values else None

    crowd_abs: list[float] = []
    uniform_abs: list[float] = []
    for s in per_q:
        if uniform_gammas is not None:
            g_u = uniform_gammas.get(s.question_id)
            if g_u is not None:
                uniform_abs.append(s.bs_lab - g_u)
        if crowd_gammas is not None:
            g_c = crowd_gammas.get(s.question_id)
            if g_c is not None:
                crowd_abs.append(s.bs_lab - g_c)

    abi_uniform = compute_abi(uniform_abs) if uniform_abs else None
    if crowd_abs:
        abi_crowd = compute_abi(crowd_abs)
    else:
        # Spec scenario: "single-model run → abi_crowd MUST equal abi_uniform".
        abi_crowd = abi_uniform

    return ModelProbabilisticAggregate(
        n_questions=n_total,
        n_fallback=n_fallback,
        fallback_share=fallback_share,
        bi=bi,
        bi_dec=bi_dec,
        nll=avg_nll,
        mbs=avg_mbs,
        abi_crowd=abi_crowd,
        abi_uniform=abi_uniform,
    )


__all__ = [
    "NLL_EPS",
    "PerQuestionScore",
    "ModelProbabilisticAggregate",
    "brier_score_lab",
    "brier_score_dec",
    "nll",
    "mbs",
    "brier_index",
    "compute_abi",
    "per_question_scores_for",
    "uniform_gamma_for",
    "crowd_gamma_for",
    "aggregate_probabilistic",
]
