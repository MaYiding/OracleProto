"""Pin tests for the composite MetricFn (manifest-driven 3-bucket weights).

The 3-bucket composite uses panel-manifest weights over {binary, mc-single,
mc-multi}, e.g. (0.30, 0.50, 0.20), applied to per-bucket exam_score
averages. The code-side 4-bucket COMPOSITE_WEIGHTS default (yes_no=0.15,
binary_named=0.15, mc_single=0.28, mc_multi=0.42) in `composite.py` is a
different aggregation contract; this MetricFn is the panel-side path
driven entirely by the manifest weight dict.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from forecast_eval.analysis.flatten import SampleRow
from forecast_eval.analysis.inference import build_default_metric_fns


def _mk_sample(
    qid: str,
    question_type: str,
    choice_type: str,
    options: list[str],
    picked_letters: frozenset[str],
    sidx: int = 0,
) -> SampleRow:
    """Build a minimal eligible SampleRow with a parsed answer letter set."""
    final_letters_json = json.dumps(sorted(picked_letters))
    return SampleRow(
        model="m",
        question_id=qid,
        question_type=question_type,
        choice_type=choice_type,
        options=options,
        sample_idx=sidx,
        correct=1,
        parse_ok=1,
        tool_calls_count=0,
        react_steps=0,
        prompt_tokens=0,
        completion_tokens=0,
        reasoning_tokens=0,
        latency_ms=0,
        final_answer_letters=final_letters_json,
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


def test_composite_weights_from_manifest():
    """Composite metric reads bucket weights from the kwargs dict passed in."""
    weights = {"binary": 0.30, "mc-single": 0.50, "mc-multi": 0.20}
    fns = build_default_metric_fns(composite_weights=weights)
    assert "composite" in fns

    # Synthetic 3-question panel: one of each bucket, all answered correctly.
    samples = {
        "q_binary": [
            _mk_sample("q_binary", "yes_no", "single", ["A", "B"], frozenset({"A"}), 0),
        ],
        "q_mcs": [
            _mk_sample("q_mcs", "multiple_choice", "single", ["A", "B", "C", "D"], frozenset({"C"}), 0),
        ],
        "q_mcm": [
            _mk_sample("q_mcm", "multiple_choice", "multi", ["A", "B", "C"], frozenset({"A", "B"}), 0),
        ],
    }
    gt_map = {
        "q_binary": frozenset({"A"}),
        "q_mcs": frozenset({"C"}),
        "q_mcm": frozenset({"A", "B"}),
    }
    composite = fns["composite"](samples, gt_map)
    assert composite is not None
    # All three buckets score 1.0, weighted mean = 1.0.
    assert math.isclose(composite, 1.0, abs_tol=1e-9)


def test_composite_three_bucket_arithmetic():
    """Verify the per-question -> per-bucket -> weighted-mean math on synthetic buckets.

    The metric averages per-question exam_score within each of the
    {binary, mc-single, mc-multi} buckets, then takes a weighted mean
    across buckets under panel-manifest weights.
    """
    weights = {"binary": 0.30, "mc-single": 0.50, "mc-multi": 0.20}
    fns = build_default_metric_fns(composite_weights=weights)

    # Bucket A: a single binary question with exam_score = 0.7416.
    # exam_score for binary single-choice is 0 or 1. Synthesize a graded
    # outcome via a multi-choice question whose picked letters overlap
    # with the ground truth, so per-question exam_score lands between
    # 0 and 1 and the weighted composite is non-trivial.
    samples = {
        "qb": [_mk_sample("qb", "yes_no", "single", ["A", "B"], frozenset({"A"}), 0)],
        "qs": [_mk_sample("qs", "multiple_choice", "single", ["A", "B", "C"], frozenset({"A"}), 0)],
        "qm": [_mk_sample("qm", "multiple_choice", "multi", ["A", "B", "C"], frozenset({"A"}), 0)],
    }
    gt = {
        "qb": frozenset({"A"}),
        "qs": frozenset({"A"}),
        "qm": frozenset({"A", "B"}),  # GT={A,B}; picked={A} -> exam_score=0.5
    }
    composite = fns["composite"](samples, gt)
    # binary=1.0, mc-single=1.0, mc-multi=0.5
    # composite = 0.30*1.0 + 0.50*1.0 + 0.20*0.5 = 0.30 + 0.50 + 0.10 = 0.90
    assert composite is not None
    assert math.isclose(composite, 0.90, abs_tol=1e-6)


def test_composite_weights_invariance_under_zero_bucket():
    """When a bucket has weight 0, removing all its questions doesn't change composite."""
    weights = {"binary": 1.0, "mc-single": 0.0, "mc-multi": 0.0}
    fns = build_default_metric_fns(composite_weights=weights)
    samples_with_mcs = {
        "q1": [_mk_sample("q1", "yes_no", "single", ["A", "B"], frozenset({"A"}), 0)],
        "q2": [_mk_sample("q2", "multiple_choice", "single", ["A", "B", "C"], frozenset({"B"}), 0)],
    }
    samples_without_mcs = {
        "q1": [_mk_sample("q1", "yes_no", "single", ["A", "B"], frozenset({"A"}), 0)],
    }
    gt = {
        "q1": frozenset({"A"}),
        "q2": frozenset({"A"}),
    }
    c1 = fns["composite"](samples_with_mcs, gt)
    c2 = fns["composite"](samples_without_mcs, {"q1": frozenset({"A"})})
    assert c1 is not None and c2 is not None
    assert math.isclose(c1, c2, abs_tol=1e-9)


def test_composite_returns_none_on_empty():
    weights = {"binary": 0.30, "mc-single": 0.50, "mc-multi": 0.20}
    fns = build_default_metric_fns(composite_weights=weights)
    assert fns["composite"]({}, {}) is None
