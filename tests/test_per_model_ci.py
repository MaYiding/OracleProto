"""Pin tests for per-(model, metric) single-variable bootstrap CI."""
from __future__ import annotations

import math

import pytest

from forecast_eval.analysis.per_model_ci import per_model_ci, PerModelCI


def test_per_model_ci_constant_input_zero_width():
    """Bootstrap CI on a constant array collapses to a zero-width band."""
    panel = {
        "A": [0.6] * 80,
        "B": [0.4] * 80,
    }
    rows = per_model_ci(panel, metric_name="x", seed=20260512, n_bootstrap=1000)
    by_model = {r.model: r for r in rows}
    assert math.isclose(by_model["A"].point, 0.6, abs_tol=1e-9)
    assert math.isclose(by_model["A"].ci_lo, 0.6, abs_tol=1e-9)
    assert math.isclose(by_model["A"].ci_hi, 0.6, abs_tol=1e-9)
    assert math.isclose(by_model["A"].ci_half_width, 0.0, abs_tol=1e-9)


def test_per_model_ci_seed_stability():
    panel = {"M": [i / 100.0 for i in range(80)]}
    a = per_model_ci(panel, metric_name="x", seed=20260512, n_bootstrap=1000)
    b = per_model_ci(panel, metric_name="x", seed=20260512, n_bootstrap=1000)
    assert a == b


def test_per_model_ci_bounds():
    """ci_lo <= point <= ci_hi for any non-constant input."""
    panel = {"M": [i / 100.0 for i in range(80)]}
    rows = per_model_ci(panel, metric_name="x", seed=20260512, n_bootstrap=2000)
    r = rows[0]
    assert r.ci_lo <= r.point + 1e-9
    assert r.point <= r.ci_hi + 1e-9


def test_per_model_ci_dataclass_shape():
    panel = {"M": [0.5] * 80}
    rows = per_model_ci(panel, metric_name="acc", seed=20260512, n_bootstrap=1000)
    r = rows[0]
    assert isinstance(r, PerModelCI)
    for attr in ("model", "metric", "point", "ci_lo", "ci_hi", "ci_half_width", "n_questions"):
        assert hasattr(r, attr)
    assert r.metric == "acc"
    assert r.n_questions == 80
