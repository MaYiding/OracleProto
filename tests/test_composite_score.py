"""Composite difficulty-coefficient unit tests.

Coverage:
* Default weights produce correct values (hand-computed for a single model + metric);
* Overrides take effect (while other metrics still use defaults);
* A bucket with None is dropped and the rest are renormalized;
* All None -> composite = None;
* Weights summing to != 1 are normalized correctly;
* A bucket with weight 0 is dropped (equivalent to None);
* CSV columns line up with ``per_model_summary.csv``;
* An override metric name not in ``KNOWN_METRICS`` causes ``compute_composite`` to raise;
* Invalid config values cause ``Settings`` to raise at startup;
* ``bucket_of`` maps every (question_type, choice_type) combination correctly;
* Fixed Score integration is covered by ``test_analysis.py``.
"""
from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from forecast_eval import analysis
from forecast_eval import db as dbmod
from forecast_eval.analysis.composite import (
    COMPOSITE_BUCKETS,
    DEFAULT_WEIGHTS,
    KNOWN_METRICS,
    bucket_of,
    compute_composite,
)
from forecast_eval.analysis.writers import _SUMMARY_FIELDS
from forecast_eval.config import Settings


# --------------------------------------------------------------------------- #
# Pure function: bucket_of
# --------------------------------------------------------------------------- #


def test_bucket_of_collapses_yes_no_and_binary_named() -> None:
    # yes_no / binary_named are structurally always single, so choice_type
    # does not affect the bucket key.
    assert bucket_of("yes_no", "single") == "yes_no"
    assert bucket_of("binary_named", "single") == "binary_named"


def test_bucket_of_splits_multiple_choice_by_answer_mode() -> None:
    assert bucket_of("multiple_choice", "single") == "mc_single"
    assert bucket_of("multiple_choice", "multi") == "mc_multi"


def test_bucket_of_rejects_unknown_inputs() -> None:
    with pytest.raises(ValueError):
        bucket_of("multiple_choice", "trinary")
    with pytest.raises(ValueError):
        bucket_of("free_text", "single")


# --------------------------------------------------------------------------- #
# Pure function: compute_composite
# --------------------------------------------------------------------------- #


def _make_bucket_values(
    metric: str, by_bucket: dict[str, float | None]
) -> dict[str, dict[str, dict[str, float | None]]]:
    """Build ``{model: {metric: {bucket: value}}}``, simplified to a single model + single metric."""
    return {"m1": {metric: dict(by_bucket)}}


def test_default_weights_yield_expected_value() -> None:
    bv = _make_bucket_values(
        "fss",
        {
            "yes_no": 0.8,
            "binary_named": 0.6,
            "mc_single": 0.5,
            "mc_multi": 0.4,
        },
    )
    rep = compute_composite(
        bucket_values_per_model=bv,
        weights_default=DEFAULT_WEIGHTS,
        overrides={},
    )
    info = rep.per_model["m1"]["fss"]
    expected = (
        0.15 * 0.8 + 0.15 * 0.6 + 0.28 * 0.5 + 0.42 * 0.4
    ) / 1.0
    assert info.value == pytest.approx(expected)
    assert info.weights_kind == "default"
    assert tuple(sorted(info.buckets_used)) == (
        "binary_named",
        "mc_multi",
        "mc_single",
        "yes_no",
    )


def test_overrides_apply_to_one_metric_only() -> None:
    bv = {
        "m1": {
            "fss": {
                "yes_no": 1.0,
                "binary_named": 0.5,
                "mc_single": 0.0,
                "mc_multi": 0.0,
            },
            "pass_at_1_avg": {
                "yes_no": 1.0,
                "binary_named": 1.0,
                "mc_single": 0.0,
                "mc_multi": 0.0,
            },
        }
    }
    overrides = {"fss": {"mc_multi": 1.0}}  # only consider mc_multi
    rep = compute_composite(
        bucket_values_per_model=bv,
        weights_default=DEFAULT_WEIGHTS,
        overrides=overrides,
    )
    fss_info = rep.per_model["m1"]["fss"]
    assert fss_info.value == pytest.approx(0.0)  # mc_multi only = 0
    assert fss_info.weights_kind == "overridden"
    assert fss_info.buckets_used == ("mc_multi",)

    pass_info = rep.per_model["m1"]["pass_at_1_avg"]
    expected = 0.15 * 1.0 + 0.15 * 1.0 + 0.28 * 0.0 + 0.42 * 0.0
    assert pass_info.value == pytest.approx(expected)
    assert pass_info.weights_kind == "default"


def test_none_bucket_dropped_and_renormalized() -> None:
    """When binary_named=None, that bucket is dropped and the remaining buckets are renormalized."""
    bv = _make_bucket_values(
        "fss",
        {
            "yes_no": 0.8,
            "binary_named": None,
            "mc_single": 0.5,
            "mc_multi": 0.4,
        },
    )
    rep = compute_composite(
        bucket_values_per_model=bv,
        weights_default=DEFAULT_WEIGHTS,
        overrides={},
    )
    info = rep.per_model["m1"]["fss"]
    denom = 0.15 + 0.28 + 0.42
    expected = (0.15 * 0.8 + 0.28 * 0.5 + 0.42 * 0.4) / denom
    assert info.value == pytest.approx(expected)
    assert tuple(sorted(info.buckets_used)) == ("mc_multi", "mc_single", "yes_no")
    # Normalized weights should sum to 1.0
    assert sum(info.weights_used_normalized.values()) == pytest.approx(1.0)
    # Normalized weight per bucket
    assert info.weights_used_normalized["yes_no"] == pytest.approx(0.15 / denom)
    assert info.weights_used_normalized["mc_single"] == pytest.approx(0.28 / denom)
    assert info.weights_used_normalized["mc_multi"] == pytest.approx(0.42 / denom)


def test_all_none_yields_none() -> None:
    bv = _make_bucket_values(
        "fss",
        {
            "yes_no": None,
            "binary_named": None,
            "mc_single": None,
            "mc_multi": None,
        },
    )
    rep = compute_composite(
        bucket_values_per_model=bv,
        weights_default=DEFAULT_WEIGHTS,
        overrides={},
    )
    info = rep.per_model["m1"]["fss"]
    assert info.value is None
    assert info.buckets_used == ()


def test_unnormalized_weights_still_correct() -> None:
    """Weights summing to != 1 must still be normalized correctly."""
    bv = _make_bucket_values(
        "fss",
        {"mc_single": 0.5, "mc_multi": 0.0},
    )
    rep = compute_composite(
        bucket_values_per_model=bv,
        weights_default={"mc_single": 60.0, "mc_multi": 40.0},  # unnormalized
        overrides={},
    )
    info = rep.per_model["m1"]["fss"]
    expected = (60.0 * 0.5 + 40.0 * 0.0) / 100.0
    assert info.value == pytest.approx(expected)


def test_zero_weight_bucket_excluded() -> None:
    """A bucket with weight 0 behaves like None and does not participate in composition."""
    bv = _make_bucket_values(
        "fss",
        {"mc_single": 0.5, "mc_multi": 0.9},
    )
    rep = compute_composite(
        bucket_values_per_model=bv,
        weights_default={"mc_single": 1.0, "mc_multi": 0.0},
        overrides={},
    )
    info = rep.per_model["m1"]["fss"]
    assert info.value == pytest.approx(0.5)
    assert info.buckets_used == ("mc_single",)


def test_unknown_metric_in_override_raises() -> None:
    bv = _make_bucket_values("fss", {"yes_no": 0.5})
    overrides = {"unknown_metric": {"yes_no": 1.0}}
    with pytest.raises(ValueError, match="not a known metric"):
        compute_composite(
            bucket_values_per_model=bv,
            weights_default=DEFAULT_WEIGHTS,
            overrides=overrides,
        )


def test_known_metrics_align_with_summary_fields() -> None:
    """``KNOWN_METRICS`` must equal ``_SUMMARY_FIELDS`` minus the metadata columns."""
    summary_data = {f for f in _SUMMARY_FIELDS if f not in ("model", "sampling_n")}
    assert KNOWN_METRICS == summary_data


def test_default_weights_sum_to_one() -> None:
    """The four nested-difficulty coefficients must sum to one."""
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)
    assert set(DEFAULT_WEIGHTS) == set(COMPOSITE_BUCKETS)


# --------------------------------------------------------------------------- #
# Settings startup-time validation
# --------------------------------------------------------------------------- #


def _settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("MODELS", "m1")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")


def test_settings_default_composite_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _settings_env(monkeypatch)
    s = Settings()
    assert s.COMPOSITE_WEIGHTS == DEFAULT_WEIGHTS
    assert s.COMPOSITE_WEIGHT_OVERRIDES == {}


def test_settings_parses_csv_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _settings_env(monkeypatch)
    monkeypatch.setenv(
        "COMPOSITE_WEIGHTS",
        "yes_no=0.10,mc_multi=0.90",
    )
    monkeypatch.setenv(
        "COMPOSITE_WEIGHT_OVERRIDES",
        "fss=mc_single=0.3,mc_multi=0.7;cohen_kappa=mc_multi=1.0",
    )
    s = Settings()
    assert s.COMPOSITE_WEIGHTS == {"yes_no": 0.10, "mc_multi": 0.90}
    assert s.COMPOSITE_WEIGHT_OVERRIDES == {
        "fss": {"mc_single": 0.3, "mc_multi": 0.7},
        "cohen_kappa": {"mc_multi": 1.0},
    }


@pytest.mark.parametrize(
    "env_key,env_value,expected_msg",
    [
        ("COMPOSITE_WEIGHTS", "wrong_bucket=0.5", "not in"),
        ("COMPOSITE_WEIGHTS", "mc_single=-0.1,mc_multi=1.0", "must be >= 0"),
        ("COMPOSITE_WEIGHTS", "mc_single=0,mc_multi=0", "at least one weight"),
        (
            "COMPOSITE_WEIGHT_OVERRIDES",
            "fss=unknown=0.5",
            "not in",
        ),
    ],
)
def test_settings_rejects_invalid_composite_config(
    monkeypatch: pytest.MonkeyPatch,
    env_key: str,
    env_value: str,
    expected_msg: str,
) -> None:
    _settings_env(monkeypatch)
    monkeypatch.setenv(env_key, env_value)
    with pytest.raises(ValueError, match=expected_msg):
        Settings()


# --------------------------------------------------------------------------- #
