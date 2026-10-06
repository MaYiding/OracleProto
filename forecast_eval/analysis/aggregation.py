"""K-trial aggregators for the probabilistic family.

Three aggregators take the per-question $K$-sample probability vectors and
return a single aggregated probability vector:

* `arithmetic_mean` — default; per-element average.
* `logit_space_mean` — paper §C.9 default. For `single` choice_type:
  $\\hat{p}_l = \\mathrm{softmax}(\\overline{\\log p}_l)$ (geometric mean
  normalised onto the simplex). For `multi`: per-label
  $\\hat{p}_l = \\sigma(\\overline{\\mathrm{logit}\\,p}_l)$.
* `loo_shrinkage` — scan $\\alpha \\in \\{0, 0.1, \\dots, 1.0\\}$ on the
  shrinkage parameter and return the optimal $\\alpha^*$ plus the full BI
  curve. Used as a diagnostic ("does this dataset benefit from shrinking
  toward prior?"), not as the default aggregator.

`majority_vote_v4_letter` returns a logit-space mean argmax: $K$ floating-
point logits almost never tie, so letter-set ties are essentially
impossible.

All math is in pure Python — same convention as `proper_score.py`. The clip
floor `NLL_EPS = 10^-3` is reused so log/logit operations stay finite even
when a $0$ or $1$ probability slips through.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..prompts import index_to_letter
from .proper_score import NLL_EPS, brier_score_lab


def _clip(p: float, eps: float = NLL_EPS) -> float:
    return min(max(float(p), eps), 1.0 - eps)


def _logit(p: float, eps: float = NLL_EPS) -> float:
    pc = _clip(p, eps)
    return math.log(pc / (1.0 - pc))


def _sigmoid(x: float) -> float:
    # Numerically stable on both signs (overflow-safe for very large |x|).
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _softmax(xs: list[float]) -> list[float]:
    if not xs:
        return []
    mx = max(xs)
    exps = [math.exp(x - mx) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def _validate_predictions(predictions: list[list[float]]) -> int:
    if not predictions:
        raise ValueError("aggregator requires at least one prediction vector")
    k = len(predictions[0])
    if k == 0:
        raise ValueError("aggregator received an empty probability vector")
    for v in predictions:
        if len(v) != k:
            raise ValueError(
                f"prediction length mismatch across trials: expected {k} got {len(v)}"
            )
    return k


def arithmetic_mean(predictions: list[list[float]]) -> list[float]:
    """Per-element arithmetic mean of $K$ probability vectors. $K=1$ is the identity."""
    k = _validate_predictions(predictions)
    n = len(predictions)
    sums = [0.0] * k
    for vec in predictions:
        for i, p in enumerate(vec):
            sums[i] += float(p)
    return [s / n for s in sums]


def logit_space_mean(
    predictions: list[list[float]], choice_type: str
) -> list[float]:
    """Logit-space K-trial mean (paper §C.9 default).

    `single`: softmax of the mean log probability — equivalent to a Bayesian
    model average over independent posteriors with the simplex constraint
    re-imposed.

    `multi`: per-label sigmoid of the mean logit — each option is an
    independent Bernoulli with its own posterior.

    $K=1$ reduces to the input within $\\epsilon$ clipping precision.
    """
    k = _validate_predictions(predictions)
    n = len(predictions)
    if choice_type == "single":
        log_sums = [0.0] * k
        for vec in predictions:
            for i, p in enumerate(vec):
                log_sums[i] += math.log(_clip(p))
        log_means = [s / n for s in log_sums]
        return _softmax(log_means)
    # multi: per-label sigmoid(mean(logit p))
    logit_sums = [0.0] * k
    for vec in predictions:
        for i, p in enumerate(vec):
            logit_sums[i] += _logit(p)
    return [_sigmoid(s / n) for s in logit_sums]


def _per_question_aggregated_logits(
    predictions_per_q: list[list[list[float]]],
    choice_type: str,
) -> list[list[float]]:
    """Pre-compute the per-question aggregated log-prob (single) or logit (multi).

    Used by `loo_shrinkage` to avoid recomputing $\\overline{\\log p}$ /
    $\\overline{\\mathrm{logit}\\,p}$ inside the alpha grid scan.
    """
    out: list[list[float]] = []
    for preds in predictions_per_q:
        k = _validate_predictions(preds)
        n = len(preds)
        if choice_type == "single":
            sums = [0.0] * k
            for vec in preds:
                for i, p in enumerate(vec):
                    sums[i] += math.log(_clip(p))
            out.append([s / n for s in sums])
        else:
            sums = [0.0] * k
            for vec in preds:
                for i, p in enumerate(vec):
                    sums[i] += _logit(p)
            out.append([s / n for s in sums])
    return out


def shrinkage_predict(
    aggregated_logits: list[float], alpha: float, choice_type: str
) -> list[float]:
    """Apply $\\alpha \\in [0, 1]$ on top of an aggregated logit vector.

    $\\alpha = 1$ → full logit-space mean. $\\alpha = 0$ → uniform (single) /
    $0.5$ per label (multi). Used by `loo_shrinkage` to interpolate between
    "trust the K-trial signal" and "back off to prior".
    """
    scaled = [float(alpha) * v for v in aggregated_logits]
    if choice_type == "single":
        return _softmax(scaled)
    return [_sigmoid(s) for s in scaled]


@dataclass(frozen=True)
class ShrinkageResult:
    """Return type for `loo_shrinkage`.

    `curve` lists (alpha, mean_bs, bi) tuples for the full grid, sorted by
    alpha. `alpha_star` is the alpha minimizing mean BS (== maximizing BI).
    """

    alpha_star: float
    bi_star: float
    mean_bs_star: float
    curve: list[tuple[float, float, float]]
    n_questions: int
    choice_type: str


def _default_alpha_grid() -> list[float]:
    return [round(0.1 * i, 1) for i in range(11)]


def loo_shrinkage(
    predictions_per_q: list[list[list[float]]],
    observations_per_q: list[list[int]],
    choice_type: str,
    alpha_grid: list[float] | None = None,
) -> ShrinkageResult:
    """Scan $\\alpha$ on $\\{0, 0.1, \\dots, 1.0\\}$ and return $\\alpha^*$.

    For each $\\alpha$ on the grid, the per-question prediction is
    $\\mathrm{softmax}(\\alpha \\cdot \\overline{\\log p})$ (single) or
    $\\sigma(\\alpha \\cdot \\overline{\\mathrm{logit}\\,p})$ (multi). The
    selection criterion is mean per-question label-wise Brier Score:
    $\\alpha^* = \\arg\\min_\\alpha \\overline{BS}_{\\text{lab}}(\\alpha)$.

    With a single global $\\alpha$, leave-one-out cross-validation reduces
    algebraically to the unweighted mean BS over the full $\\mathcal{Q}$ —
    LOO refinement labels the spec parity but produces the same $\\alpha^*$.
    """
    if not predictions_per_q:
        raise ValueError("loo_shrinkage requires at least one question")
    if len(predictions_per_q) != len(observations_per_q):
        raise ValueError(
            f"predictions/observations length mismatch: "
            f"{len(predictions_per_q)} vs {len(observations_per_q)}"
        )
    grid = list(alpha_grid) if alpha_grid is not None else _default_alpha_grid()
    if not grid:
        raise ValueError("alpha_grid must be non-empty")

    aggregated = _per_question_aggregated_logits(predictions_per_q, choice_type)
    n_q = len(predictions_per_q)

    curve: list[tuple[float, float, float]] = []
    best_alpha = grid[0]
    best_bs = float("inf")
    for alpha in grid:
        bs_total = 0.0
        for agg_l, obs in zip(aggregated, observations_per_q):
            pred = shrinkage_predict(agg_l, alpha, choice_type)
            bs_total += brier_score_lab(pred, obs)
        mean_bs = bs_total / n_q
        bi = 100.0 * (1.0 - math.sqrt(max(0.0, mean_bs)))
        curve.append((float(alpha), mean_bs, bi))
        if mean_bs < best_bs - 1e-12:
            best_bs = mean_bs
            best_alpha = float(alpha)

    best_bi = 100.0 * (1.0 - math.sqrt(max(0.0, best_bs)))
    return ShrinkageResult(
        alpha_star=best_alpha,
        bi_star=best_bi,
        mean_bs_star=best_bs,
        curve=curve,
        n_questions=n_q,
        choice_type=choice_type,
    )


def majority_vote_v4_letter(
    aggregated_probs: list[float],
    choice_type: str,
    threshold: float = 0.5,
) -> frozenset[str]:
    """Project an aggregated probability vector onto a predicted letter set.

    `single`: argmax → single letter. Ties only happen at exact equality,
    which is essentially impossible after a logit-space K-trial mean across
    floating-point predictions.

    `multi`: per-label thresholding $p_l \\ge \\tau$ → letter set.
    """
    if not aggregated_probs:
        return frozenset()
    if choice_type == "single":
        best_idx = 0
        best_p = aggregated_probs[0]
        for i, p in enumerate(aggregated_probs):
            if p > best_p:
                best_p = p
                best_idx = i
        return frozenset({index_to_letter(best_idx)})
    return frozenset(
        index_to_letter(i)
        for i, p in enumerate(aggregated_probs)
        if p >= threshold
    )


def majority_vote_accuracy_v4(
    predictions_per_q: list[list[list[float]]],
    gt_per_q: list[frozenset[str]],
    choice_type_per_q: list[str],
    threshold: float = 0.5,
) -> tuple[int, int]:
    """Logit-space majority vote accuracy across a question set.

    For each question:
    1. Aggregate the $K$ probability vectors via `logit_space_mean`.
    2. Project onto a letter prediction via `majority_vote_v4_letter`.
    3. Compare to ground truth (set equality).

    Returns `(n_correct, n_resolvable)`. Skips questions with no predictions
    (which become "unresolvable").
    """
    if not (
        len(predictions_per_q) == len(gt_per_q) == len(choice_type_per_q)
    ):
        raise ValueError("predictions / gt / choice_type length mismatch")
    n_correct = 0
    n_resolvable = 0
    for preds, gt, ctype in zip(
        predictions_per_q, gt_per_q, choice_type_per_q
    ):
        if not preds:
            continue
        try:
            agg = logit_space_mean(preds, ctype)
        except ValueError:
            continue
        n_resolvable += 1
        if majority_vote_v4_letter(agg, ctype, threshold) == gt:
            n_correct += 1
    return n_correct, n_resolvable


__all__ = [
    "ShrinkageResult",
    "arithmetic_mean",
    "logit_space_mean",
    "shrinkage_predict",
    "loo_shrinkage",
    "majority_vote_v4_letter",
    "majority_vote_accuracy_v4",
]
