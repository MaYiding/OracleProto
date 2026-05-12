"""Pin tests for rank-stability bootstrap."""
from __future__ import annotations

import math

from forecast_eval.analysis.rank_stability import (
    RankStabilityRow,
    RankStabilitySummary,
    rank_stability,
)


def test_rank_stability_separated_models():
    """When models have strictly separated per-q values, every bootstrap
    resample yields the same ranking — leader posterior is 1 / 0 and
    Kendall tau vs point rank is 1.0."""
    panel = {
        f"M{rank}": [0.1 * rank + 0.001 * q for q in range(80)]
        for rank in range(1, 6)
    }
    rows, summary = rank_stability(
        panel, metric_name="acc",
        seed=20260512, n_bootstrap=1000,
    )
    by_model = {r.model: r for r in rows}
    assert math.isclose(by_model["M5"].leader_posterior, 1.0, abs_tol=1e-9)
    assert math.isclose(by_model["M1"].leader_posterior, 0.0, abs_tol=1e-9)
    assert math.isclose(summary["kendall_tau_mean_vs_point"], 1.0, abs_tol=1e-3)


def test_rank_stability_degenerate_models():
    """When all models tie on every question, leader_posterior values
    must remain valid probabilities."""
    panel = {f"M{i}": [0.5] * 80 for i in range(5)}
    rows, summary = rank_stability(
        panel, metric_name="x",
        seed=20260512, n_bootstrap=500,
    )
    for r in rows:
        assert 0.0 <= r.leader_posterior <= 1.0


def test_rank_stability_row_fields():
    panel = {"M": [0.5] * 80, "N": [0.7] * 80}
    rows, summary = rank_stability(panel, metric_name="m", seed=20260512, n_bootstrap=200)
    for r in rows:
        assert isinstance(r, RankStabilityRow)
        for attr in ("metric", "model", "point_rank", "median_rank", "rank_p05", "rank_p95", "leader_posterior"):
            assert hasattr(r, attr)
        assert 1 <= r.point_rank <= 2
        assert 1 <= r.median_rank <= 2
        assert r.rank_p05 <= r.median_rank <= r.rank_p95
