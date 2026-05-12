"""Benjamini-Hochberg FDR control and Holm-vs-BH sensitivity reporting.

BH (1995) controls the false discovery rate: for sorted p-values
`p_(1) <= ... <= p_(m)`, the BH-adjusted q-value is
`q_(k) = min_{j>=k} (m / j) * p_(j)`, capped at 1. The Holm-vs-BH
sensitivity reporter classifies each pair into one of {both, holm_only,
bh_only, neither} at a given alpha, so a panel-wide audit can confirm
the headline ordering is not an artifact of the family-wise vs FDR
correction choice.
"""
from __future__ import annotations


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return BH-adjusted q-values in input order.

    Classic step-up procedure with Yekutieli-style monotone-correction
    (the `min_{j>=k}` enforcement) so q-values are non-decreasing when
    sorted by raw p.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    sorted_p = [p_values[i] for i in order]
    adj_sorted: list[float] = [0.0] * m
    cur_min = 1.0
    for rank_from_top in range(m - 1, -1, -1):
        k = rank_from_top + 1
        raw_q = sorted_p[rank_from_top] * m / k
        cur_min = min(cur_min, raw_q)
        adj_sorted[rank_from_top] = max(0.0, min(1.0, cur_min))
    out = [0.0] * m
    for sorted_idx, orig_idx in enumerate(order):
        out[orig_idx] = adj_sorted[sorted_idx]
    return out


def compare_holm_vs_bh(
    p_holm: list[float],
    p_bh: list[float],
    *,
    alpha: float = 0.05,
) -> dict[str, int]:
    """Classify each pair into 4 cells based on significance at alpha.

    Both arrays must align element-wise. Returns a dict with keys
    `both`, `holm_only`, `bh_only`, `neither`.
    """
    assert len(p_holm) == len(p_bh)
    cats = {"both": 0, "holm_only": 0, "bh_only": 0, "neither": 0}
    for ph, pb in zip(p_holm, p_bh):
        s_holm = ph < alpha
        s_bh = pb < alpha
        if s_holm and s_bh:
            cats["both"] += 1
        elif s_holm and not s_bh:
            cats["holm_only"] += 1
        elif (not s_holm) and s_bh:
            cats["bh_only"] += 1
        else:
            cats["neither"] += 1
    return cats


__all__ = ["benjamini_hochberg", "compare_holm_vs_bh"]
