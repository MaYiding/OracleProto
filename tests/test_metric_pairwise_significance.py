"""Pin tests for `pairwise_metric_significance`:
- Holm-adjusted p column
- posterior_a_better column
- sig_holm_05 = (p_holm < 0.05) bit
- writer column schema
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from forecast_eval.analysis.flatten import SampleRow
from forecast_eval.analysis.inference import (
    DEFAULT_METRIC_FNS,
    pairwise_metric_significance,
)
from forecast_eval.analysis.writers import _write_metric_pairwise_significance_csv


def _mk(qid, picked, sidx, *, gt):
    return SampleRow(
        model="m",
        question_id=qid,
        question_type="yes_no",
        choice_type="single",
        options=["A", "B"],
        sample_idx=sidx,
        correct=1 if frozenset({picked}) == gt else 0,
        parse_ok=1,
        tool_calls_count=0,
        react_steps=0,
        prompt_tokens=0,
        completion_tokens=0,
        reasoning_tokens=0,
        latency_ms=0,
        final_answer_letters=json.dumps([picked]),
        error=None,
        created_at="2026-04-27T00:00:00Z",
        finish_reason="stop",
        nudges_used=0,
        belief_final=None,
        belief_trace=None,
        belief_parse_ok=0,
        probabilities=None,
        is_fallback=False,
    )


def _fake_samples_5_models():
    """5 models x 10 questions; deterministic probs spread to produce
    a mix of significant / non-significant pairs at alpha=0.05."""
    probs = {"M1": 0.95, "M2": 0.80, "M3": 0.60, "M4": 0.50, "M5": 0.20}
    samples_by_model: dict[str, dict[str, list]] = {}
    gt_map = {f"q{qi}": frozenset({"A"}) for qi in range(10)}
    for model, p in probs.items():
        samples_by_model[model] = {}
        for qi in range(10):
            qid = f"q{qi}"
            picked = "A" if qi < int(round(p * 10)) else "B"
            samples_by_model[model][qid] = [_mk(qid, picked, 0, gt=gt_map[qid])]
    return samples_by_model, gt_map


def test_metric_pairwise_significance_columns():
    samples_by_model, gt_map = _fake_samples_5_models()
    metric_fns = {"acc": DEFAULT_METRIC_FNS["acc"]}
    results = pairwise_metric_significance(
        samples_by_model, gt_map, metric_fns,
        n_bootstrap=500, seed=20260512,
    )
    assert results, "expected non-empty results for 5-model fake panel"
    for r in results:
        assert hasattr(r, "p_raw")
        assert hasattr(r, "p_holm")
        assert hasattr(r, "posterior_a_better")
        assert hasattr(r, "sig_holm_05")
        assert r.p_raw <= r.p_holm + 1e-9, "Holm-adjusted p must be >= raw p"
        assert r.sig_holm_05 == (r.p_holm < 0.05)
        assert 0.0 <= r.posterior_a_better <= 1.0 + 1e-9


def test_metric_pairwise_significance_holm_monotone():
    """Holm-adjusted p-values for a single metric family are monotone
    non-decreasing after sorting by raw p."""
    samples_by_model, gt_map = _fake_samples_5_models()
    metric_fns = {"acc": DEFAULT_METRIC_FNS["acc"]}
    results = pairwise_metric_significance(
        samples_by_model, gt_map, metric_fns,
        n_bootstrap=500, seed=20260512,
    )
    acc_rows = sorted([r for r in results if r.metric == "acc"], key=lambda r: r.p_raw)
    last = -1.0
    for r in acc_rows:
        assert r.p_holm >= last - 1e-9
        last = r.p_holm


def test_writer_emits_expected_columns(tmp_path):
    samples_by_model, gt_map = _fake_samples_5_models()
    metric_fns = {"acc": DEFAULT_METRIC_FNS["acc"]}
    results = pairwise_metric_significance(
        samples_by_model, gt_map, metric_fns,
        n_bootstrap=500, seed=20260512,
    )
    out = tmp_path / "metric_pairwise_significance.csv"
    _write_metric_pairwise_significance_csv(out, results)

    with out.open() as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        assert header == [
            "metric", "model_a", "model_b", "n_questions",
            "delta_mean", "ci_low", "ci_high",
            "p_raw", "p_holm", "cohens_d", "posterior_a_better", "sig_holm_05",
        ]
        rows = list(reader)
        assert rows, "writer produced no data rows"
