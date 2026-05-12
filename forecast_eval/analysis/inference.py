"""Statistical inference for model comparisons.

BS-paired bootstrap on per-question Brier (consumed by `grid.py` and the
probabilistic family):

* `paired_bootstrap` — $B = 5000$ paired resamples on per-question Brier
  scores; returns 95% CI and two-sided p-value.
* `holm_bonferroni` — FWER control across multi-comparison families;
  returns adjusted p-values in original order.
* `difficulty_tertile` — tertile split on per-question $\\gamma_q$.
* `paired_bootstrap_by_difficulty` — per-tertile `paired_bootstrap`.
* `posterior_a_better_than_b` — Monte-Carlo $\\Pr(\\mathrm{BI}_A > \\mathrm{BI}_B)$.
* `posterior_normal_fit` — closed-form normal approximation.
* `pairwise_paired_bootstrap` — every model pair × Holm correction.

Multi-metric paired bootstrap (parameterised over `MetricFn`):

* `metric_paired_bootstrap(metric_fn, samples_a_by_q, samples_b_by_q, gt_map, ...)`
  — for any metric (FSS / Acc / MV_Acc / Fleiss κ / EBI). Returns ΔMean,
  95% CI, two-sided p, Cohen's d.
* `pairwise_metric_bootstrap(samples_by_model_by_q, gt_map, metric_fns)`
  — orchestrate over (metric × model pair) cartesian product.
* `DEFAULT_METRIC_FNS` — the 5 default metric wrappers.
* `cohens_d_from_bootstrap` — effect size from the resampled delta dist.

Bootstrap iteration uses unique resample-keys (`f"{qid}__bs{j}"`) so a
question sampled with replacement contributes its full sample-set
multiple times to the metric — without dict-key collisions and without
losing the "paired" structure (the same index hits both A and B).

Pure Python: `random` for the RNG, `math.erf` for the normal CDF, no
numpy / scipy dependency.
"""
from __future__ import annotations

import dataclasses
import math
import random
from collections.abc import Callable
from dataclasses import dataclass

from .flatten import SampleRow, gt_vector


# --------------------------------------------------------------------------- #
# Paired bootstrap
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Output of `paired_bootstrap`. `bootstrap_means` is the resampled mean
    delta distribution — kept around so callers can derive posterior
    probabilities without repeating the bootstrap."""

    delta_mean: float
    ci_low: float
    ci_high: float
    p_two_sided: float
    n_bootstrap: int
    n_questions: int
    bootstrap_means: list[float]


def paired_bootstrap(
    bs_a: list[float],
    bs_b: list[float],
    *,
    n_bootstrap: int = 5000,
    seed: int = 42,
    ci_alpha: float = 0.05,
) -> PairedBootstrapResult:
    """Paired bootstrap on per-question Brier-score differences.

    The two arrays MUST be aligned by question (so `bs_a[i]` and `bs_b[i]`
    refer to the same question). Each bootstrap iteration samples $N$
    indices with replacement and uses them to subset BOTH arrays — this is
    the "paired" property that controls for question-level variance, which
    paper §G.2 quantifies at 62% of total variance.

    The two-sided p-value is the standard $2 \\min(\\Pr(\\Delta \\le 0),
    \\Pr(\\Delta \\ge 0))$ on the bootstrap distribution; without recentring
    it tests "is the observed delta consistent with zero?"
    """
    if len(bs_a) != len(bs_b):
        raise ValueError(
            f"paired arrays must be the same length: {len(bs_a)} vs {len(bs_b)}"
        )
    n = len(bs_a)
    if n == 0:
        return PairedBootstrapResult(
            delta_mean=0.0, ci_low=0.0, ci_high=0.0, p_two_sided=1.0,
            n_bootstrap=0, n_questions=0, bootstrap_means=[],
        )
    deltas = [float(a) - float(b) for a, b in zip(bs_a, bs_b)]
    delta_mean = sum(deltas) / n

    rng = random.Random(seed)
    bootstrap_means: list[float] = []
    for _ in range(n_bootstrap):
        s = 0.0
        for _ in range(n):
            s += deltas[rng.randrange(n)]
        bootstrap_means.append(s / n)

    sorted_means = sorted(bootstrap_means)
    lo_idx = max(0, int(ci_alpha / 2 * n_bootstrap))
    hi_idx = min(n_bootstrap - 1, int((1 - ci_alpha / 2) * n_bootstrap) - 1)
    ci_low = sorted_means[lo_idx]
    ci_high = sorted_means[hi_idx]

    n_le_0 = sum(1 for m in bootstrap_means if m <= 0)
    n_ge_0 = sum(1 for m in bootstrap_means if m >= 0)
    p_low = n_le_0 / n_bootstrap
    p_high = n_ge_0 / n_bootstrap
    p_two = min(1.0, 2.0 * min(p_low, p_high))

    return PairedBootstrapResult(
        delta_mean=delta_mean,
        ci_low=ci_low,
        ci_high=ci_high,
        p_two_sided=p_two,
        n_bootstrap=n_bootstrap,
        n_questions=n,
        bootstrap_means=bootstrap_means,
    )


# --------------------------------------------------------------------------- #
# Holm-Bonferroni
# --------------------------------------------------------------------------- #


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni FWER control. Returns adjusted p-values aligned with input order.

    Algorithm:
    1. Sort p-values ascending; let $i = 0, 1, \\dots, n-1$ be sorted ranks.
    2. For each rank $i$: $p^*_{(i)} = (n - i) \\cdot p_{(i)}$ (clipped to 1).
    3. Apply the cumulative max from left so adjusted p-values are monotone.
    4. Map back to the original input positions.
    """
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    cur_max = 0.0
    for rank, i in enumerate(indexed):
        adj = float(p_values[i]) * (n - rank)
        if adj > 1.0:
            adj = 1.0
        if adj > cur_max:
            cur_max = adj
        adjusted[i] = cur_max
    return adjusted


# --------------------------------------------------------------------------- #
# Difficulty tertile
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DifficultyTertile:
    """A question's difficulty bucket plus the boundary $\\gamma$ thresholds."""

    by_question: dict[str, str]   # question_id → "low" / "mid" / "high"
    threshold_low: float          # γ at the low/mid boundary
    threshold_high: float         # γ at the mid/high boundary
    n_low: int
    n_mid: int
    n_high: int


def difficulty_tertile(gammas: dict[str, float]) -> DifficultyTertile:
    """Split questions into low / mid / high tertiles by per-question $\\gamma$.

    Sorts questions ascending by $\\gamma$ (lower = easier — small baseline
    BS = baseline already gets it right). Splits into thirds; ties at the
    boundary go into the lower tertile, which is a deterministic choice
    that keeps results reproducible across runs.
    """
    if not gammas:
        return DifficultyTertile(
            by_question={}, threshold_low=0.0, threshold_high=0.0,
            n_low=0, n_mid=0, n_high=0,
        )
    items = sorted(gammas.items(), key=lambda kv: kv[1])
    n = len(items)
    one_third = n // 3
    two_third = (2 * n) // 3
    by_q: dict[str, str] = {}
    for i, (q, _) in enumerate(items):
        if i < one_third:
            by_q[q] = "low"
        elif i < two_third:
            by_q[q] = "mid"
        else:
            by_q[q] = "high"
    threshold_low = items[one_third][1] if one_third < n else items[-1][1]
    threshold_high = items[two_third][1] if two_third < n else items[-1][1]
    n_low = sum(1 for v in by_q.values() if v == "low")
    n_mid = sum(1 for v in by_q.values() if v == "mid")
    n_high = sum(1 for v in by_q.values() if v == "high")
    return DifficultyTertile(
        by_question=by_q,
        threshold_low=threshold_low,
        threshold_high=threshold_high,
        n_low=n_low,
        n_mid=n_mid,
        n_high=n_high,
    )


def paired_bootstrap_by_difficulty(
    bs_a_by_q: dict[str, float],
    bs_b_by_q: dict[str, float],
    tertile: DifficultyTertile,
    *,
    n_bootstrap: int = 5000,
    seed: int = 42,
    ci_alpha: float = 0.05,
) -> dict[str, PairedBootstrapResult]:
    """Per-tertile paired bootstrap. Returns `{tier: result}` for low/mid/high."""
    out: dict[str, PairedBootstrapResult] = {}
    for tier in ("low", "mid", "high"):
        qs = [q for q, t in tertile.by_question.items() if t == tier]
        a_arr = [bs_a_by_q[q] for q in qs if q in bs_a_by_q and q in bs_b_by_q]
        b_arr = [bs_b_by_q[q] for q in qs if q in bs_a_by_q and q in bs_b_by_q]
        out[tier] = paired_bootstrap(
            a_arr, b_arr,
            n_bootstrap=n_bootstrap,
            seed=seed,
            ci_alpha=ci_alpha,
        )
    return out


# --------------------------------------------------------------------------- #
# Posterior over BI
# --------------------------------------------------------------------------- #


def _normal_cdf(z: float) -> float:
    """Standard normal CDF using `math.erf`."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def posterior_a_better_than_b(
    bs_a: list[float],
    bs_b: list[float],
    *,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> float:
    """$\\Pr(\\mathrm{BI}_A > \\mathrm{BI}_B)$ via paired bootstrap.

    BI is a monotone-decreasing function of mean BS, so the event is
    equivalent to $\\Pr(\\overline{BS}_A < \\overline{BS}_B)$. We compute it
    directly by counting bootstrap iterations where A's mean BS is smaller.
    """
    if len(bs_a) != len(bs_b):
        raise ValueError("paired arrays must be equal length")
    n = len(bs_a)
    if n == 0:
        return 0.5
    rng = random.Random(seed)
    n_a_better = 0
    for _ in range(n_bootstrap):
        sa = 0.0
        sb = 0.0
        for _ in range(n):
            j = rng.randrange(n)
            sa += bs_a[j]
            sb += bs_b[j]
        if sa < sb:
            n_a_better += 1
        elif sa == sb:
            n_a_better += 0.5  # half-credit on ties
    return n_a_better / n_bootstrap


def posterior_normal_fit(bs_a: list[float], bs_b: list[float]) -> float:
    """Normal-fit closed form: $\\Pr(\\Delta < 0)$ under $\\mathcal{N}(\\bar\\Delta, s^2/n)$.

    Reported alongside the bootstrap-based estimate as a sanity check —
    the two should agree closely on $N \\ge 100$ when the per-question
    delta distribution is roughly symmetric.
    """
    if len(bs_a) != len(bs_b):
        raise ValueError("paired arrays must be equal length")
    n = len(bs_a)
    if n == 0:
        return 0.5
    deltas = [float(a) - float(b) for a, b in zip(bs_a, bs_b)]
    mean = sum(deltas) / n
    if n < 2:
        return 1.0 if mean < 0 else (0.0 if mean > 0 else 0.5)
    var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
    if var <= 0:
        return 1.0 if mean < 0 else (0.0 if mean > 0 else 0.5)
    se = math.sqrt(var / n)
    z = -mean / se
    return _normal_cdf(z)


# --------------------------------------------------------------------------- #
# Convenience: pairwise comparison across many models
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelPairResult:
    """Bundled per-pair statistics. Fields beyond `pair` are all in BS units
    (the writer converts to BI when serialising)."""

    model_a: str
    model_b: str
    delta_bs_mean: float
    ci_low: float
    ci_high: float
    p_raw: float
    p_holm: float | None
    posterior_a_better: float
    n_questions: int


def pairwise_paired_bootstrap(
    bs_by_model_qid: dict[str, dict[str, float]],
    *,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> list[ModelPairResult]:
    """Run paired bootstrap on every ordered pair of models, return Holm-adjusted p-values.

    Pairs are ordered (model_a, model_b) for `model_a < model_b` lexicographically;
    a positive `delta_bs_mean` means A is WORSE than B (higher BS), so positive
    `posterior_a_better` reflects "P(A's mean BS smaller than B's)".

    Pairs missing a question in either model are silently dropped — the
    intersection is the only fair comparison surface.
    """
    models = sorted(bs_by_model_qid.keys())
    pairs: list[ModelPairResult] = []
    raw_p: list[float] = []
    for i, ma in enumerate(models):
        for mb in models[i + 1:]:
            common = sorted(set(bs_by_model_qid[ma]) & set(bs_by_model_qid[mb]))
            bs_a = [bs_by_model_qid[ma][q] for q in common]
            bs_b = [bs_by_model_qid[mb][q] for q in common]
            res = paired_bootstrap(
                bs_a, bs_b, n_bootstrap=n_bootstrap, seed=seed,
            )
            posterior = posterior_a_better_than_b(
                bs_a, bs_b, n_bootstrap=n_bootstrap, seed=seed,
            )
            pairs.append(
                ModelPairResult(
                    model_a=ma,
                    model_b=mb,
                    delta_bs_mean=res.delta_mean,
                    ci_low=res.ci_low,
                    ci_high=res.ci_high,
                    p_raw=res.p_two_sided,
                    p_holm=None,
                    posterior_a_better=posterior,
                    n_questions=len(common),
                )
            )
            raw_p.append(res.p_two_sided)
    if pairs:
        adjusted = holm_bonferroni(raw_p)
        # Re-emit with p_holm filled in.
        pairs = [
            ModelPairResult(
                model_a=p.model_a,
                model_b=p.model_b,
                delta_bs_mean=p.delta_bs_mean,
                ci_low=p.ci_low,
                ci_high=p.ci_high,
                p_raw=p.p_raw,
                p_holm=adjusted[idx],
                posterior_a_better=p.posterior_a_better,
                n_questions=p.n_questions,
            )
            for idx, p in enumerate(pairs)
        ]
    return pairs


# --------------------------------------------------------------------------- #
# Multi-metric paired bootstrap
# --------------------------------------------------------------------------- #


# Per-(model, question) sample-list dict keyed by question_id, plus the
# matching `gt_map`. The bootstrap subsets pass dicts keyed by unique
# resample IDs (`{qid}__bs{j}`); the gt_map alongside is rebuilt with the
# same keys, so the metric_fn never has to know whether it's looking at
# the full set or a bootstrap subset.
MetricFn = Callable[
    [dict[str, list[SampleRow]], dict[str, frozenset[str]]],
    float | None,
]


@dataclass(frozen=True)
class MetricBootstrapResult:
    """Output of `metric_paired_bootstrap`. Mirrors `PairedBootstrapResult`
    but parameterised over the metric and reports Cohen's d effect size.

    `delta_mean` is the OBSERVED delta on the full common question set
    (A - B), not the bootstrap mean. Bootstrap is used only for CI / p / d.
    `cohens_d` is `mean(bootstrap_means) / std(bootstrap_means)`; values
    above 0.8 indicate large effect, 0.5-0.8 medium, < 0.2 trivial.
    """

    metric_name: str
    model_a: str
    model_b: str
    delta_mean: float
    ci_low: float
    ci_high: float
    p_two_sided: float
    cohens_d: float
    n_questions: int
    n_bootstrap: int


def cohens_d_from_bootstrap(deltas: list[float]) -> float:
    """Cohen's d from a bootstrap delta distribution: $|\\bar{\\Delta}| / s$.

    Returns 0.0 when the bootstrap variance is degenerate (all same value)
    AND the mean is also 0 — signals "no effect". A non-zero mean with
    zero variance returns +inf (ideal certainty); we clamp to 1e9 to keep
    downstream formatting finite without losing the "very large" semantics.
    """
    n = len(deltas)
    if n == 0:
        return 0.0
    mean = sum(deltas) / n
    if n < 2:
        return 0.0
    var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
    if var <= 0:
        if abs(mean) < 1e-12:
            return 0.0
        return 1e9 if mean > 0 else -1e9
    return mean / math.sqrt(var)


def _bootstrap_subset(
    samples_by_q: dict[str, list[SampleRow]],
    gt_map: dict[str, frozenset[str]],
    qid_indices: list[str],
) -> tuple[dict[str, list[SampleRow]], dict[str, frozenset[str]]]:
    """Build a resample subset where each draw becomes a unique dict key.

    `qid_indices` may contain duplicates (with-replacement). For draw index
    $j$ we put `f"{qid}__bs{j}"` into both the sample dict and the gt dict,
    so the metric function sees as many "questions" as draws.

    Sample objects are NOT copied — we share references. This is safe
    because metric functions never mutate samples. SampleRow's
    `question_id` field is overwritten via `dataclasses.replace` only inside
    metric wrappers that need it (e.g. `_metric_fn_acc` calls `_aggregate`
    which groups by `sample.question_id`).
    """
    sub_samples: dict[str, list[SampleRow]] = {}
    sub_gt: dict[str, frozenset[str]] = {}
    for j, qid in enumerate(qid_indices):
        if qid not in samples_by_q or qid not in gt_map:
            continue
        # Unique key per draw position. Duplicate qids land under different
        # keys → both copies contribute to the metric.
        key = f"{qid}__bs{j}"
        sub_samples[key] = samples_by_q[qid]
        sub_gt[key] = gt_map[qid]
    return sub_samples, sub_gt


def metric_paired_bootstrap(
    metric_fn: MetricFn,
    samples_a_by_q: dict[str, list[SampleRow]],
    samples_b_by_q: dict[str, list[SampleRow]],
    gt_map: dict[str, frozenset[str]],
    *,
    metric_name: str,
    model_a: str = "A",
    model_b: str = "B",
    n_bootstrap: int = 5000,
    seed: int = 42,
    ci_alpha: float = 0.05,
) -> MetricBootstrapResult | None:
    """Paired bootstrap on any metric defined by `MetricFn`.

    Restricts to questions present in both A and B (the only fair surface).
    Each iteration resamples question_ids with replacement, builds matching
    A and B subsets via `_bootstrap_subset` (paired by index), evaluates
    `metric_fn` on both, and records $\\Delta = m_A - m_B$.

    Returns None if:
    * the common question set is empty;
    * the observed metric on A or B is None (metric not computable on this
      data — e.g. Fleiss κ on K=1 fixture).

    Two-sided p-value: $2 \\min(\\Pr(\\Delta_b \\le 0), \\Pr(\\Delta_b \\ge 0))$
    on the bootstrap distribution.
    """
    common = sorted(set(samples_a_by_q.keys()) & set(samples_b_by_q.keys()))
    if not common:
        return None

    # Observed delta on full common set — same dict-key-as-question convention
    # so the metric_fn doesn't need a special "full set" branch.
    full_a = {q: samples_a_by_q[q] for q in common}
    full_b = {q: samples_b_by_q[q] for q in common}
    full_gt = {q: gt_map[q] for q in common if q in gt_map}
    metric_a_obs = metric_fn(full_a, full_gt)
    metric_b_obs = metric_fn(full_b, full_gt)
    if metric_a_obs is None or metric_b_obs is None:
        return None
    delta_obs = metric_a_obs - metric_b_obs

    rng = random.Random(seed)
    n = len(common)
    bootstrap_deltas: list[float] = []
    for _ in range(n_bootstrap):
        # Same indices for A and B → "paired" property.
        idxs = [rng.randrange(n) for _ in range(n)]
        sub_qids = [common[i] for i in idxs]
        sub_a, sub_gt_a = _bootstrap_subset(samples_a_by_q, gt_map, sub_qids)
        sub_b, sub_gt_b = _bootstrap_subset(samples_b_by_q, gt_map, sub_qids)
        m_a = metric_fn(sub_a, sub_gt_a)
        m_b = metric_fn(sub_b, sub_gt_b)
        if m_a is None or m_b is None:
            # Skip degenerate iterations — should be rare for sensible n_bootstrap.
            continue
        bootstrap_deltas.append(m_a - m_b)

    if not bootstrap_deltas:
        return None

    sorted_deltas = sorted(bootstrap_deltas)
    n_b = len(sorted_deltas)
    lo_idx = max(0, int(ci_alpha / 2 * n_b))
    hi_idx = min(n_b - 1, int((1 - ci_alpha / 2) * n_b) - 1)
    ci_low = sorted_deltas[lo_idx]
    ci_high = sorted_deltas[hi_idx]

    n_le_0 = sum(1 for d in bootstrap_deltas if d <= 0)
    n_ge_0 = sum(1 for d in bootstrap_deltas if d >= 0)
    p_low = n_le_0 / n_b
    p_high = n_ge_0 / n_b
    p_two = min(1.0, 2.0 * min(p_low, p_high))

    return MetricBootstrapResult(
        metric_name=metric_name,
        model_a=model_a,
        model_b=model_b,
        delta_mean=delta_obs,
        ci_low=ci_low,
        ci_high=ci_high,
        p_two_sided=p_two,
        cohens_d=cohens_d_from_bootstrap(bootstrap_deltas),
        n_questions=n,
        n_bootstrap=n_b,
    )


def pairwise_metric_bootstrap(
    samples_by_model_by_q: dict[str, dict[str, list[SampleRow]]],
    gt_map: dict[str, frozenset[str]],
    metric_fns: dict[str, MetricFn],
    *,
    n_bootstrap: int = 5000,
    seed: int = 42,
    ci_alpha: float = 0.05,
) -> list[MetricBootstrapResult]:
    """For every (metric, ordered model pair), run `metric_paired_bootstrap`.

    Pairs are `(model_a, model_b)` for `model_a < model_b` lex. A pair is
    silently skipped when the metric is not computable on the data; the
    caller decides whether to surface that as "row missing" or "p=NA".

    Returns a flat list sorted by `(metric_name, model_a, model_b)` so the
    writer can emit a deterministic CSV without a separate sort pass.
    """
    models = sorted(samples_by_model_by_q.keys())
    out: list[MetricBootstrapResult] = []
    for metric_name in sorted(metric_fns.keys()):
        metric_fn = metric_fns[metric_name]
        for i, ma in enumerate(models):
            for mb in models[i + 1:]:
                res = metric_paired_bootstrap(
                    metric_fn,
                    samples_by_model_by_q[ma],
                    samples_by_model_by_q[mb],
                    gt_map,
                    metric_name=metric_name,
                    model_a=ma,
                    model_b=mb,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                    ci_alpha=ci_alpha,
                )
                if res is not None:
                    out.append(res)
    return out


@dataclass(frozen=True)
class SingleMetricBootstrapResult:
    """Output of `metric_single_bootstrap`. Same shape as PairedBootstrapResult
    but for a single-model single-variable bootstrap on a panel-level metric.
    """

    metric_name: str
    model: str
    point: float
    ci_low: float
    ci_high: float
    n_questions: int
    n_bootstrap: int


def metric_single_bootstrap(
    metric_fn: MetricFn,
    samples_by_q: dict[str, list[SampleRow]],
    gt_map: dict[str, frozenset[str]],
    *,
    metric_name: str,
    model: str = "M",
    n_bootstrap: int = 5000,
    seed: int = 20260512,
    ci_alpha: float = 0.05,
) -> SingleMetricBootstrapResult | None:
    """Single-variable bootstrap CI for a panel-level metric.

    Resamples question ids with replacement `B` times, evaluates
    `metric_fn` on each resampled panel, and returns a percentile CI on
    the resulting distribution. Required for non-linear metrics (composite,
    Cohen kappa, Fleiss kappa, FSS) where `mean over per-question values`
    does NOT equal the panel-level metric.
    """
    common = sorted(samples_by_q.keys())
    if not common:
        return None
    full_gt = {q: gt_map[q] for q in common if q in gt_map}
    full_samples = {q: samples_by_q[q] for q in common}
    point = metric_fn(full_samples, full_gt)
    if point is None:
        return None

    rng = random.Random(seed)
    n = len(common)
    bootstrap_values: list[float] = []
    for _ in range(n_bootstrap):
        idxs = [rng.randrange(n) for _ in range(n)]
        sub_qids = [common[i] for i in idxs]
        sub_samples, sub_gt = _bootstrap_subset(samples_by_q, gt_map, sub_qids)
        v = metric_fn(sub_samples, sub_gt)
        if v is None:
            continue
        bootstrap_values.append(v)

    if not bootstrap_values:
        return None
    sorted_values = sorted(bootstrap_values)
    n_b = len(sorted_values)
    lo_idx = max(0, int(ci_alpha / 2 * n_b))
    hi_idx = min(n_b - 1, int((1 - ci_alpha / 2) * n_b) - 1)
    return SingleMetricBootstrapResult(
        metric_name=metric_name,
        model=model,
        point=point,
        ci_low=sorted_values[lo_idx],
        ci_high=sorted_values[hi_idx],
        n_questions=n,
        n_bootstrap=n_b,
    )


@dataclass(frozen=True)
class MetricPairwiseSignificance:
    """Extends `MetricBootstrapResult` with Holm-adjusted p and the
    bootstrap posterior P(A > B). Holm is applied within each metric's
    family of `C(n, 2)` pairs; `posterior_a_better` is derived from the
    two-sided p and the sign of the observed delta.
    """

    metric: str
    model_a: str
    model_b: str
    n_questions: int
    delta_mean: float
    ci_low: float
    ci_high: float
    p_raw: float
    p_holm: float
    cohens_d: float
    posterior_a_better: float
    sig_holm_05: bool


def pairwise_metric_significance(
    samples_by_model_by_q: dict[str, dict[str, list[SampleRow]]],
    gt_map: dict[str, frozenset[str]],
    metric_fns: dict[str, MetricFn],
    *,
    n_bootstrap: int = 5000,
    seed: int = 20260512,
    ci_alpha: float = 0.05,
) -> list["MetricPairwiseSignificance"]:
    """Run `pairwise_metric_bootstrap` and then Holm-correct per metric family.

    `posterior_a_better` is the fraction of bootstrap iterations where
    delta > 0 — semantically `P(A > B | observed)` under the bootstrap.
    We derive it from `p_two_sided` and the sign of `delta_mean`: when
    delta_mean > 0, posterior = 1 - p_two/2; when delta_mean < 0,
    posterior = p_two/2. This matches the bootstrap relation
    `p_two_sided = 2 * min(P(d<=0), P(d>=0))` for real-valued metrics.
    """
    raw = pairwise_metric_bootstrap(
        samples_by_model_by_q, gt_map, metric_fns,
        n_bootstrap=n_bootstrap, seed=seed, ci_alpha=ci_alpha,
    )

    out: list[MetricPairwiseSignificance] = []
    by_metric: dict[str, list[MetricBootstrapResult]] = {}
    for r in raw:
        by_metric.setdefault(r.metric_name, []).append(r)

    for metric, results in by_metric.items():
        p_raws = [r.p_two_sided for r in results]
        p_holms = holm_bonferroni(p_raws)
        for r, p_holm in zip(results, p_holms):
            if r.delta_mean >= 0:
                posterior_a_better = 1.0 - r.p_two_sided / 2.0
            else:
                posterior_a_better = r.p_two_sided / 2.0
            out.append(MetricPairwiseSignificance(
                metric=metric,
                model_a=r.model_a,
                model_b=r.model_b,
                n_questions=r.n_questions,
                delta_mean=r.delta_mean,
                ci_low=r.ci_low,
                ci_high=r.ci_high,
                p_raw=r.p_two_sided,
                p_holm=p_holm,
                cohens_d=r.cohens_d,
                posterior_a_better=posterior_a_better,
                sig_holm_05=(p_holm < 0.05),
            ))

    return sorted(out, key=lambda r: (r.metric, r.model_a, r.model_b))


# --------------------------------------------------------------------------- #
# Default metric wrappers
# --------------------------------------------------------------------------- #


def _flatten_with_dict_key_as_qid(
    samples_by_q: dict[str, list[SampleRow]],
) -> list[SampleRow]:
    """Flatten `dict[key, list[SampleRow]]` → `list[SampleRow]` while
    rewriting each sample's `question_id` to match the dict key.

    Bootstrap subsets use synthetic keys like `qid__bs17`. Helpers like
    `_aggregate` group samples by `sample.question_id`, so without the
    rewrite a duplicated draw would silently merge into the original
    qid bucket — defeating the bootstrap. We use `dataclasses.replace`
    only when the key already differs (skip the copy on the natural full-set
    case where `key == sample.question_id`).
    """
    flat: list[SampleRow] = []
    for key, samples in samples_by_q.items():
        for s in samples:
            if s.question_id == key:
                flat.append(s)
            else:
                flat.append(dataclasses.replace(s, question_id=key))
    return flat


def _metric_fn_fss(
    samples_by_q: dict[str, list[SampleRow]],
    gt_map: dict[str, frozenset[str]],
) -> float | None:
    """Main metric. `accuracy.fss` already accepts dicts directly."""
    from .accuracy import fss

    if not samples_by_q:
        return None
    result = fss(samples_by_q, gt_map)
    return result["fss"]


def _max_sampling_n(samples_by_q: dict[str, list[SampleRow]]) -> int:
    """Largest unique sample_idx span across all questions; floor at 1."""
    n = 1
    for ss in samples_by_q.values():
        if not ss:
            continue
        n = max(n, len({s.sample_idx for s in ss}))
    return n


def _metric_fn_acc(
    samples_by_q: dict[str, list[SampleRow]],
    gt_map: dict[str, frozenset[str]],
) -> float | None:
    """Per-sample pass@1 (matches `Aggregate.pass_at_1_avg`).

    Direct sample-level computation; equivalent to running `_aggregate`
    and reading `pass_at_1_avg` but 50-100x faster on bootstrap subsets
    because it skips the 16-field Aggregate build.
    """
    if not samples_by_q:
        return None
    total = 0
    correct = 0
    for samples in samples_by_q.values():
        for s in samples:
            if not s.is_resolvable:
                continue
            total += 1
            if s.correct == 1:
                correct += 1
    if total == 0:
        return None
    return correct / total


def _metric_fn_mv_acc(
    samples_by_q: dict[str, list[SampleRow]],
    gt_map: dict[str, frozenset[str]],
) -> float | None:
    """Majority-vote accuracy. Returns None when sampling_n < 2 (no MV signal).

    Direct equivalent of `Aggregate.majority_vote_accuracy`.
    """
    from collections import Counter

    if not samples_by_q:
        return None
    sampling_n = _max_sampling_n(samples_by_q)
    if sampling_n < 2:
        return None
    hits: list[int] = []
    for qid, samples in samples_by_q.items():
        if not samples:
            continue
        gt = gt_map.get(qid)
        if gt is None:
            continue
        parsed = [s.parsed_letters for s in samples if s.is_eligible and s.parsed_letters is not None]
        if not parsed:
            continue
        counts = Counter(parsed)
        top_count = max(counts.values())
        winners = [k for k, v in counts.items() if v == top_count]
        if len(winners) != 1:
            continue
        hits.append(1 if winners[0] == gt else 0)
    if not hits:
        return None
    return sum(hits) / len(hits)


def _metric_fn_fleiss_kappa(
    samples_by_q: dict[str, list[SampleRow]],
    gt_map: dict[str, frozenset[str]],
) -> float | None:
    """Fleiss κ — needs an options_map; we synthesize from the first sample
    of each dict key. Returns None when no question has K_q ≥ 2."""
    from .consistency import fleiss_kappa

    if not samples_by_q:
        return None
    options_map = {
        key: ss[0].options
        for key, ss in samples_by_q.items()
        if ss and ss[0].options
    }
    if not options_map:
        return None
    return fleiss_kappa(samples_by_q, options_map)


def _metric_fn_ebi(
    samples_by_q: dict[str, list[SampleRow]],
    gt_map: dict[str, frozenset[str]],
) -> float | None:
    """BI on label-wise Brier — falls through to None on legacy fixtures
    that lack the `probabilities` field."""
    from .probabilistic import _aggregate_question_probs
    from .proper_score import brier_index, brier_score_lab

    if not samples_by_q:
        return None
    bs_values: list[float] = []
    for key, samples in samples_by_q.items():
        gt = gt_map.get(key)
        if gt is None or not samples or not samples[0].options:
            continue
        opts = samples[0].options
        agg_probs = _aggregate_question_probs(samples)
        if agg_probs is None:
            continue
        obs = gt_vector(gt, len(opts))
        bs_values.append(brier_score_lab(agg_probs, obs))
    if not bs_values:
        return None
    return brier_index(bs_values)


def _metric_fn_pass_any(
    samples_by_q: dict[str, list[SampleRow]],
    gt_map: dict[str, frozenset[str]],
) -> float | None:
    """OR over trials of per-trial correctness, averaged over questions.

    Direct equivalent of `Aggregate.pass_any_at_n`.
    """
    if not samples_by_q:
        return None
    hits: list[int] = []
    for samples in samples_by_q.values():
        rs = [s for s in samples if s.is_resolvable]
        if not rs:
            continue
        hits.append(1 if any(s.correct == 1 for s in rs) else 0)
    if not hits:
        return None
    return sum(hits) / len(hits)


def _metric_fn_pass_all(
    samples_by_q: dict[str, list[SampleRow]],
    gt_map: dict[str, frozenset[str]],
) -> float | None:
    """AND over trials of per-trial correctness, averaged over questions.

    Direct equivalent of `Aggregate.at_least_all_at_n` — counts a
    question as `1` only when ALL `sampling_n` trials were resolvable and
    correct (so a partial-resolvable question scores 0).
    """
    if not samples_by_q:
        return None
    sampling_n = _max_sampling_n(samples_by_q)
    hits: list[int] = []
    for samples in samples_by_q.values():
        rs = [s for s in samples if s.is_resolvable]
        if not rs:
            continue
        n_rs = len(rs)
        n_correct = sum(1 for s in rs if s.correct == 1)
        hits.append(1 if (n_rs == sampling_n and n_correct == n_rs) else 0)
    if not hits:
        return None
    return sum(hits) / len(hits)


def _metric_fn_pass_at_1(
    samples_by_q: dict[str, list[SampleRow]],
    gt_map: dict[str, frozenset[str]],
) -> float | None:
    """Strict-equality pass@1 averaged over questions (alias of `acc`).

    Kept as a named registry entry so the panel-level pairwise paired
    bootstrap can emit a row with `metric=pass_at_1` under its conventional
    name, without renaming the existing `acc` slot consumed elsewhere.
    """
    if not samples_by_q:
        return None
    return _metric_fn_acc(samples_by_q, gt_map)


def _metric_fn_cohen_kappa(
    samples_by_q: dict[str, list[SampleRow]],
    gt_map: dict[str, frozenset[str]],
) -> float | None:
    """Cohen's κ on per-sample correct/wrong outcomes.

    Delegates to `accuracy.cohen_kappa`, which already accepts the
    `(samples_by_q, gt_map)` shape used by the bootstrap pipeline.
    """
    from .accuracy import cohen_kappa

    if not samples_by_q:
        return None
    return cohen_kappa(samples_by_q, gt_map)


def _metric_fn_composite_factory(
    composite_weights: dict[str, float],
) -> MetricFn:
    """Build a composite MetricFn closed over a 3-bucket weight dict.

    `composite_weights` keys are `{"binary", "mc-single", "mc-multi"}`;
    values sum to ~1. Per question, computes the mean `exam_score` over
    eligible samples (in-base only), groups by `(question_type,
    choice_type)` into the 3 buckets, averages within each bucket, then
    takes the weighted mean across buckets — denominator drops buckets
    that contribute zero usable questions. Driven by `composite_weights`
    from the panel manifest; the code-side 4-bucket DEFAULT_WEIGHTS in
    `composite.py` remains a distinct contract knob untouched by this
    metric.
    """
    from .exam_score import exam_score

    expected_buckets = ("binary", "mc-single", "mc-multi")
    if set(composite_weights.keys()) != set(expected_buckets):
        raise ValueError(
            f"composite_weights keys must be {set(expected_buckets)}; "
            f"got {set(composite_weights.keys())}"
        )

    def _bucket_of_sample(s: SampleRow) -> str | None:
        if s.question_type in ("yes_no", "binary_named"):
            return "binary"
        if s.question_type == "multiple_choice":
            if s.choice_type == "single":
                return "mc-single"
            if s.choice_type == "multi":
                return "mc-multi"
        return None

    def _fn(
        samples_by_q: dict[str, list[SampleRow]],
        gt_map: dict[str, frozenset[str]],
    ) -> float | None:
        if not samples_by_q:
            return None
        per_bucket_e_q: dict[str, list[float]] = {b: [] for b in expected_buckets}
        for qid, ss in samples_by_q.items():
            gt = gt_map.get(qid)
            if gt is None or not ss:
                continue
            bucket = _bucket_of_sample(ss[0])
            if bucket is None:
                continue
            per_sample_scores: list[float] = []
            for s in ss:
                v = exam_score(s, gt)
                if v is None:
                    continue
                per_sample_scores.append(v)
            if not per_sample_scores:
                continue
            per_bucket_e_q[bucket].append(
                sum(per_sample_scores) / len(per_sample_scores)
            )
        total_w = 0.0
        total_v = 0.0
        for bucket, w in composite_weights.items():
            arr = per_bucket_e_q.get(bucket, [])
            if not arr:
                continue
            total_w += w
            total_v += w * (sum(arr) / len(arr))
        if total_w <= 0:
            return None
        return total_v / total_w

    return _fn


def build_default_metric_fns(
    *,
    composite_weights: dict[str, float] | None = None,
) -> dict[str, MetricFn]:
    """Build the full metric registry for panel-aware paths.

    Returns the legacy 5-metric `DEFAULT_METRIC_FNS` plus the four extra
    bootstrap metrics (pass_at_1, cohen_kappa, pass_any, pass_all). Pass
    `composite_weights` to add the 3-bucket composite under panel-manifest
    weights; omit it to leave composite out of the registry.
    """
    out = dict(DEFAULT_METRIC_FNS)
    out.update(_EXTRA_METRIC_FNS)
    if composite_weights is not None:
        out["composite"] = _metric_fn_composite_factory(composite_weights)
    return out


# Public registry for `pairwise_metric_bootstrap`. Keys become the
# `metric` column in `pairwise_bootstrap.csv`. Held to the original
# 5-metric set so the legacy non-panel analysis path keeps writing the
# same `pairwise_bootstrap.csv` schema; the 4 additional metrics
# (pass_at_1, cohen_kappa, pass_any, pass_all) and the composite metric
# enter the registry via `build_default_metric_fns()` and flow only
# through the panel-aware path.
DEFAULT_METRIC_FNS: dict[str, MetricFn] = {
    "fss": _metric_fn_fss,
    "acc": _metric_fn_acc,
    "mv_acc": _metric_fn_mv_acc,
    "fleiss_kappa": _metric_fn_fleiss_kappa,
    "ebi": _metric_fn_ebi,
}


# Internal: extra metrics surfaced only through `build_default_metric_fns`.
_EXTRA_METRIC_FNS: dict[str, MetricFn] = {
    "pass_at_1": _metric_fn_pass_at_1,
    "cohen_kappa": _metric_fn_cohen_kappa,
    "pass_any": _metric_fn_pass_any,
    "pass_all": _metric_fn_pass_all,
}


__all__ = [
    "PairedBootstrapResult",
    "DifficultyTertile",
    "ModelPairResult",
    "paired_bootstrap",
    "paired_bootstrap_by_difficulty",
    "holm_bonferroni",
    "difficulty_tertile",
    "posterior_a_better_than_b",
    "posterior_normal_fit",
    "pairwise_paired_bootstrap",
    # Multi-metric bootstrap
    "MetricFn",
    "MetricBootstrapResult",
    "metric_paired_bootstrap",
    "pairwise_metric_bootstrap",
    "cohens_d_from_bootstrap",
    "DEFAULT_METRIC_FNS",
    "build_default_metric_fns",
    "MetricPairwiseSignificance",
    "pairwise_metric_significance",
    "SingleMetricBootstrapResult",
    "metric_single_bootstrap",
]
