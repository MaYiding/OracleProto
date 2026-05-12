"""Questionwise bootstrap rank stability.

For each metric, build the `(N_q x N_models)` per-question matrix,
resample question indices with replacement `B` times, and on each
resample re-rank the models by the per-question mean. Output:

* Per-model: point_rank, median bootstrap rank, p5/p95 rank interval,
  leader_posterior = fraction of resamples where rank == 1.
* Summary: distribution of Kendall tau between each resample's rank
  vector and the point-estimate rank vector.

Stable tie-breaking: equal means break alphabetically (the secondary
sort key); deterministic for a fixed seed.
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class RankStabilityRow:
    metric: str
    model: str
    point_rank: int
    median_rank: int
    rank_p05: int
    rank_p95: int
    leader_posterior: float


RankStabilitySummary = dict[str, float]


def _ranks_from_means(means: list[tuple[str, float]]) -> dict[str, int]:
    """Higher mean -> lower rank number (rank 1 = best); ties break alphabetically."""
    ordered = sorted(means, key=lambda x: (-x[1], x[0]))
    return {model: i + 1 for i, (model, _) in enumerate(ordered)}


def _kendall_tau(rank_a: dict[str, int], rank_b: dict[str, int]) -> float:
    """Kendall tau-b between two rank vectors over the same model set."""
    keys = sorted(rank_a.keys())
    n = len(keys)
    if n < 2:
        return 1.0
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            mi, mj = keys[i], keys[j]
            sign_a = rank_a[mi] - rank_a[mj]
            sign_b = rank_b[mi] - rank_b[mj]
            if sign_a * sign_b > 0:
                concordant += 1
            elif sign_a * sign_b < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 1.0
    return (concordant - discordant) / total


def rank_stability(
    panel: dict[str, list[float]],
    *,
    metric_name: str,
    seed: int = 20260512,
    n_bootstrap: int = 5000,
    ci_alpha: float = 0.05,
) -> tuple[list[RankStabilityRow], RankStabilitySummary]:
    """Compute rank-stability rows and Kendall tau summary.

    Args:
        panel: `{model_id: [v_q1, ..., v_qN]}`; every model must share the
            same `N` (questions are paired across models).

    Returns:
        `(rows, summary)` where rows is one RankStabilityRow per model and
        summary contains kendall_tau_{mean,p05,p95}_vs_point.
    """
    models = sorted(panel.keys())
    if not models:
        return [], {}
    n_q = len(next(iter(panel.values())))
    if any(len(panel[m]) != n_q for m in models):
        raise ValueError("panel arrays must share the same length (paired by question index)")

    point_means = [(m, sum(panel[m]) / n_q) for m in models]
    point_rank = _ranks_from_means(point_means)

    rng = random.Random(seed)
    rank_traces: dict[str, list[int]] = {m: [] for m in models}
    kendall_taus: list[float] = []
    for _ in range(n_bootstrap):
        idxs = [rng.randrange(n_q) for _ in range(n_q)]
        means = []
        for m in models:
            vals = panel[m]
            s = 0.0
            for i in idxs:
                s += vals[i]
            means.append((m, s / n_q))
        rk = _ranks_from_means(means)
        for m, r in rk.items():
            rank_traces[m].append(r)
        kendall_taus.append(_kendall_tau(rk, point_rank))

    def _pct(arr: list[int], p: float) -> int:
        if not arr:
            return 0
        s = sorted(arr)
        idx = max(0, min(len(s) - 1, int(p * len(s))))
        return s[idx]

    rows: list[RankStabilityRow] = []
    for m in models:
        ranks = rank_traces[m]
        leader = sum(1 for r in ranks if r == 1) / max(1, len(ranks))
        rows.append(RankStabilityRow(
            metric=metric_name,
            model=m,
            point_rank=point_rank[m],
            median_rank=int(statistics.median(ranks)) if ranks else point_rank[m],
            rank_p05=_pct(ranks, 0.025),
            rank_p95=_pct(ranks, 0.975),
            leader_posterior=leader,
        ))

    sorted_taus = sorted(kendall_taus)
    summary: RankStabilitySummary = {
        "kendall_tau_mean_vs_point": sum(kendall_taus) / len(kendall_taus) if kendall_taus else 0.0,
        "kendall_tau_p05": sorted_taus[max(0, int(0.025 * len(sorted_taus)))] if sorted_taus else 0.0,
        "kendall_tau_p95": sorted_taus[min(len(sorted_taus) - 1, int(0.975 * len(sorted_taus)))] if sorted_taus else 0.0,
    }

    return rows, summary


def rank_stability_panel(
    samples_by_model_by_q,
    gt_map,
    metric_fn,
    *,
    metric_name: str,
    seed: int = 20260512,
    n_bootstrap: int = 5000,
    ci_alpha: float = 0.05,
):
    """Questionwise bootstrap rank stability via direct metric_fn evaluation.

    For each bootstrap draw, resamples question ids with replacement and
    evaluates `metric_fn` on the resampled panel for every model. The
    rank vector per draw is the ordering of those panel-level metric
    values; stable tie-breaking is alphabetical.

    Required for non-linear metrics (composite, Cohen kappa, Fleiss kappa,
    FSS) where ranking by mean-of-per-q-values disagrees with ranking by
    the panel-level metric. Linear metrics (pass_at_1 / pass_any /
    pass_all / mv_acc) yield identical rankings under either path.
    """
    import random
    from .inference import _bootstrap_subset

    models = sorted(samples_by_model_by_q.keys())
    if not models:
        return [], {}

    common_qids: set[str] | None = None
    for model in models:
        qids = set(samples_by_model_by_q[model].keys()) & set(gt_map.keys())
        if common_qids is None:
            common_qids = qids
        else:
            common_qids &= qids
    if not common_qids:
        return [], {}
    common = sorted(common_qids)

    # Observed point metric values per model
    point_values: list[tuple[str, float]] = []
    for model in models:
        sbq = {q: samples_by_model_by_q[model][q] for q in common}
        gt_sub = {q: gt_map[q] for q in common}
        v = metric_fn(sbq, gt_sub)
        if v is None:
            continue
        point_values.append((model, v))
    if len(point_values) < 2:
        return [], {}
    point_rank = _ranks_from_means(point_values)
    active_models = sorted(m for m, _ in point_values)

    rng = random.Random(seed)
    n_q = len(common)
    rank_traces: dict[str, list[int]] = {m: [] for m in active_models}
    kendall_taus: list[float] = []
    for _ in range(n_bootstrap):
        idxs = [rng.randrange(n_q) for _ in range(n_q)]
        sub_qids = [common[i] for i in idxs]
        draw_values: list[tuple[str, float]] = []
        for model in active_models:
            sub_samples, sub_gt = _bootstrap_subset(
                samples_by_model_by_q[model], gt_map, sub_qids,
            )
            v = metric_fn(sub_samples, sub_gt)
            if v is None:
                continue
            draw_values.append((model, v))
        if len(draw_values) < 2:
            continue
        rk = _ranks_from_means(draw_values)
        for m, r in rk.items():
            rank_traces[m].append(r)
        kendall_taus.append(_kendall_tau(rk, {m: point_rank[m] for m in rk}))

    def _pct(arr: list[int], p: float) -> int:
        if not arr:
            return 0
        s = sorted(arr)
        idx = max(0, min(len(s) - 1, int(p * len(s))))
        return s[idx]

    rows: list[RankStabilityRow] = []
    for m in active_models:
        ranks = rank_traces[m]
        leader = sum(1 for r in ranks if r == 1) / max(1, len(ranks))
        rows.append(RankStabilityRow(
            metric=metric_name,
            model=m,
            point_rank=point_rank[m],
            median_rank=int(statistics.median(ranks)) if ranks else point_rank[m],
            rank_p05=_pct(ranks, 0.025),
            rank_p95=_pct(ranks, 0.975),
            leader_posterior=leader,
        ))

    sorted_taus = sorted(kendall_taus)
    summary: RankStabilitySummary = {
        "kendall_tau_mean_vs_point": sum(kendall_taus) / len(kendall_taus) if kendall_taus else 0.0,
        "kendall_tau_p05": sorted_taus[max(0, int(0.025 * len(sorted_taus)))] if sorted_taus else 0.0,
        "kendall_tau_p95": sorted_taus[min(len(sorted_taus) - 1, int(0.975 * len(sorted_taus)))] if sorted_taus else 0.0,
    }
    return rows, summary


__all__ = [
    "RankStabilityRow",
    "RankStabilitySummary",
    "rank_stability",
    "rank_stability_panel",
]
