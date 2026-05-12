"""Pin tests for Benjamini-Hochberg FDR and Holm-vs-BH sensitivity."""
from __future__ import annotations

from forecast_eval.analysis.fdr_bh import (
    benjamini_hochberg,
    compare_holm_vs_bh,
)


def test_bh_monotone_after_sort():
    """BH q-values are monotone non-decreasing after sorting by raw p."""
    raw = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1]
    adj = benjamini_hochberg(raw)
    pairs = sorted(zip(raw, adj), key=lambda x: x[0])
    last = -1.0
    for _, q in pairs:
        assert q >= last - 1e-9
        last = q


def test_bh_bound_at_one():
    raw = [0.001, 0.5, 0.9]
    adj = benjamini_hochberg(raw)
    for q in adj:
        assert 0.0 <= q <= 1.0


def test_compare_holm_bh_categories():
    """4 categories: holm_only / bh_only / both / neither at alpha=0.05."""
    holm_p = [0.01, 0.04, 0.06, 0.20]
    bh_p =   [0.02, 0.10, 0.04, 0.30]
    cats = compare_holm_vs_bh(holm_p, bh_p, alpha=0.05)
    assert cats["both"] == 1
    assert cats["holm_only"] == 1
    assert cats["bh_only"] == 1
    assert cats["neither"] == 1
