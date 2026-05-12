"""Pin tests for pass_at_1 and cohen_kappa MetricFns."""
from __future__ import annotations

import json
import math

from forecast_eval.analysis.flatten import SampleRow
from forecast_eval.analysis.inference import build_default_metric_fns

DEFAULT_METRIC_FNS = build_default_metric_fns()


def _mk(
    qid: str,
    picked: str,
    sidx: int,
    *,
    qtype: str = "yes_no",
    choice_type: str = "single",
    options: tuple[str, ...] = ("A", "B"),
    correct: int | None = None,
    gt: frozenset[str] | None = None,
) -> SampleRow:
    if correct is None and gt is not None:
        correct = 1 if frozenset({picked}) == gt else 0
    return SampleRow(
        model="m",
        question_id=qid,
        question_type=qtype,
        choice_type=choice_type,
        options=list(options),
        sample_idx=sidx,
        correct=correct if correct is not None else 0,
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


def test_pass_at_1_basic():
    samples = {
        "q1": [
            _mk("q1", "A", 0, gt=frozenset({"A"})),
            _mk("q1", "A", 1, gt=frozenset({"A"})),
            _mk("q1", "B", 2, gt=frozenset({"A"})),
        ],
        "q2": [
            _mk("q2", "A", 0, gt=frozenset({"A"})),
            _mk("q2", "A", 1, gt=frozenset({"A"})),
            _mk("q2", "A", 2, gt=frozenset({"A"})),
        ],
    }
    gt = {"q1": frozenset({"A"}), "q2": frozenset({"A"})}
    fn = DEFAULT_METRIC_FNS["pass_at_1"]
    val = fn(samples, gt)
    # q1: 2/3 correct, q2: 3/3 correct -> mean = (2/3 + 1) / 2 = 5/6
    assert val is not None
    assert math.isclose(val, 5.0 / 6.0, abs_tol=1e-6)


def test_pass_at_1_empty_returns_none():
    fn = DEFAULT_METRIC_FNS["pass_at_1"]
    assert fn({}, {}) is None


def test_cohen_kappa_present():
    fn = DEFAULT_METRIC_FNS["cohen_kappa"]
    samples = {
        "q1": [
            _mk("q1", "A", 0, gt=frozenset({"A"})),
            _mk("q1", "A", 1, gt=frozenset({"A"})),
            _mk("q1", "A", 2, gt=frozenset({"A"})),
        ],
        "q2": [
            _mk("q2", "A", 0, gt=frozenset({"A"})),
            _mk("q2", "A", 1, gt=frozenset({"A"})),
            _mk("q2", "B", 2, gt=frozenset({"A"})),
        ],
    }
    gt = {"q1": frozenset({"A"}), "q2": frozenset({"A"})}
    val = fn(samples, gt)
    assert val is not None
    assert -1.0 <= val <= 1.0


def test_cohen_kappa_empty_returns_none():
    fn = DEFAULT_METRIC_FNS["cohen_kappa"]
    assert fn({}, {}) is None
