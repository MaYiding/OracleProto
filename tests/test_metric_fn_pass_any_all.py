"""Pin tests for pass_any / pass_all bootstrap MetricFns."""
from __future__ import annotations

import json
import math

from forecast_eval.analysis.flatten import SampleRow
from forecast_eval.analysis.inference import build_default_metric_fns

DEFAULT_METRIC_FNS = build_default_metric_fns()


def _mk(qid, picked, sidx, *, gt=None):
    correct = 0
    if gt is not None and frozenset({picked}) == gt:
        correct = 1
    return SampleRow(
        model="m",
        question_id=qid,
        question_type="yes_no",
        choice_type="single",
        options=["A", "B"],
        sample_idx=sidx,
        correct=correct,
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


def test_pass_any():
    samples = {
        "q1": [
            _mk("q1", "B", 0, gt=frozenset({"A"})),
            _mk("q1", "B", 1, gt=frozenset({"A"})),
            _mk("q1", "A", 2, gt=frozenset({"A"})),
        ],
        "q2": [
            _mk("q2", "B", 0, gt=frozenset({"A"})),
            _mk("q2", "B", 1, gt=frozenset({"A"})),
            _mk("q2", "B", 2, gt=frozenset({"A"})),
        ],
    }
    gt = {"q1": frozenset({"A"}), "q2": frozenset({"A"})}
    val = DEFAULT_METRIC_FNS["pass_any"](samples, gt)
    # q1 has at least one correct trial -> 1; q2 has none -> 0. Mean = 0.5.
    assert math.isclose(val, 0.5, abs_tol=1e-9)


def test_pass_all():
    samples = {
        "q1": [
            _mk("q1", "A", 0, gt=frozenset({"A"})),
            _mk("q1", "A", 1, gt=frozenset({"A"})),
            _mk("q1", "A", 2, gt=frozenset({"A"})),
        ],
        "q2": [
            _mk("q2", "A", 0, gt=frozenset({"A"})),
            _mk("q2", "B", 1, gt=frozenset({"A"})),
            _mk("q2", "A", 2, gt=frozenset({"A"})),
        ],
    }
    gt = {"q1": frozenset({"A"}), "q2": frozenset({"A"})}
    val = DEFAULT_METRIC_FNS["pass_all"](samples, gt)
    # q1 all correct -> 1; q2 has wrong trial -> 0. Mean = 0.5.
    assert math.isclose(val, 0.5, abs_tol=1e-9)


def test_pass_any_pass_all_inequality():
    """For any input, pass_all <= pass@1 <= pass_any."""
    samples = {
        f"q{i}": [
            _mk(f"q{i}", "A" if (i + j) % 3 != 0 else "B", j, gt=frozenset({"A"}))
            for j in range(3)
        ]
        for i in range(10)
    }
    gt = {f"q{i}": frozenset({"A"}) for i in range(10)}
    p1 = DEFAULT_METRIC_FNS["pass_at_1"](samples, gt)
    pa = DEFAULT_METRIC_FNS["pass_any"](samples, gt)
    pl = DEFAULT_METRIC_FNS["pass_all"](samples, gt)
    assert pl <= p1 + 1e-9
    assert p1 <= pa + 1e-9
