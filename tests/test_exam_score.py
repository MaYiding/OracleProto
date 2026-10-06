"""Contracts for exam-style credit, exclusion rules, and question-level averaging."""
from __future__ import annotations

import json

import pytest

from forecast_eval.analysis.exam_score import exam_score, exam_score_at_n_avg
from forecast_eval.analysis.flatten import CUTOFF, SampleRow
from forecast_eval.parser import is_correct


def _make_sample(
    *,
    question_id: str = "q1",
    sample_idx: int = 0,
    choice_type: str = "multi",
    options: list[str] | None = None,
    parse_ok: int | None = 1,
    parsed: frozenset[str] | None = None,
    error: str | None = None,
) -> SampleRow:
    """Minimal SampleRow factory for exam_score tests.

    Defaults to a multi-choice eligible sample. Pass `parsed=None` to simulate
    `final_answer_letters` missing; pass `parse_ok=0` to simulate parse failure.
    `error="skipped_training_cutoff"` triggers the cutoff path.
    """
    if options is None:
        options = ["A", "B", "C", "D"]
    final_letters_json = None
    if parsed is not None:
        final_letters_json = json.dumps(sorted(parsed))
    correct = 1 if (parse_ok == 1 and parsed is not None) else None
    return SampleRow(
        model="model_x",
        question_id=question_id,
        question_type="forecast",
        choice_type=choice_type,
        options=options,
        sample_idx=sample_idx,
        correct=correct,
        parse_ok=parse_ok,
        tool_calls_count=0,
        react_steps=0,
        prompt_tokens=0,
        completion_tokens=0,
        reasoning_tokens=0,
        latency_ms=0,
        final_answer_letters=final_letters_json,
        error=error,
        created_at="2026-04-27T00:00:00Z",
        finish_reason="stop",
        nudges_used=0,
        belief_final=None,
        belief_trace=None,
        belief_parse_ok=0,
        probabilities=None,
        is_fallback=False,
    )


# --------------------------------------------------------------------------- #
# Section §3.2 — Formula (including the user's two original examples)
# --------------------------------------------------------------------------- #


def test_user_multi_example_per_question() -> None:
    """User's original multi-choice question: GT={A,B,C}, three responses AB / ABC / AD -> e_q ~= 0.5556."""
    gt = frozenset({"A", "B", "C"})
    samples = [
        _make_sample(sample_idx=0, parsed=frozenset({"A", "B"})),
        _make_sample(sample_idx=1, parsed=frozenset({"A", "B", "C"})),
        _make_sample(sample_idx=2, parsed=frozenset({"A", "D"})),
    ]
    scores = [exam_score(s, gt) for s in samples]
    assert scores[0] == pytest.approx(2 / 3, abs=1e-4)
    assert scores[1] == pytest.approx(1.0)
    assert scores[2] == pytest.approx(0.0)
    e_q = sum(scores) / len(scores)  # type: ignore[arg-type]
    assert e_q == pytest.approx(5 / 9, abs=1e-4)


def test_user_single_example_per_question() -> None:
    """User's original single-choice question: GT={B}, three responses A / A / B -> e_q ~= 0.3333."""
    gt = frozenset({"B"})
    samples = [
        _make_sample(sample_idx=0, choice_type="single",
                     options=["A", "B"], parsed=frozenset({"A"})),
        _make_sample(sample_idx=1, choice_type="single",
                     options=["A", "B"], parsed=frozenset({"A"})),
        _make_sample(sample_idx=2, choice_type="single",
                     options=["A", "B"], parsed=frozenset({"B"})),
    ]
    scores = [exam_score(s, gt) for s in samples]
    assert scores == [0.0, 0.0, 1.0]
    e_q = sum(scores) / len(scores)  # type: ignore[arg-type]
    assert e_q == pytest.approx(1 / 3, abs=1e-4)


def test_perfect_hit_returns_1() -> None:
    s = _make_sample(parsed=frozenset({"A", "B", "C"}))
    assert exam_score(s, frozenset({"A", "B", "C"})) == pytest.approx(1.0)


def test_only_missing_partial_credit() -> None:
    """Only missed selections (FN > 0, FP = 0) -> TP/|G|."""
    s = _make_sample(parsed=frozenset({"A"}))
    assert exam_score(s, frozenset({"A", "B", "C"})) == pytest.approx(1 / 3, abs=1e-4)


def test_any_wrong_choice_returns_0() -> None:
    """Any wrong selection -> 0 (one-strike rule, even with partial correctness)."""
    s = _make_sample(parsed=frozenset({"A", "D"}))  # missed B/C and picked D
    assert exam_score(s, frozenset({"A", "B", "C"})) == pytest.approx(0.0)


def test_superset_pred_still_zero() -> None:
    """All correct + one extra wrong selection -> still 0 (FP > 0 triggers the hard gate)."""
    s = _make_sample(parsed=frozenset({"A", "B", "C", "D"}))
    assert exam_score(s, frozenset({"A", "B", "C"})) == pytest.approx(0.0)


def test_single_correct() -> None:
    s = _make_sample(choice_type="single", options=["A", "B"],
                     parsed=frozenset({"B"}))
    assert exam_score(s, frozenset({"B"})) == pytest.approx(1.0)


def test_single_wrong() -> None:
    s = _make_sample(choice_type="single", options=["A", "B"],
                     parsed=frozenset({"A"}))
    assert exam_score(s, frozenset({"B"})) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Section §3.3 — Cardinality rules (cutoff / error / parse_ok / defensive)
# --------------------------------------------------------------------------- #


def test_cutoff_returns_none() -> None:
    """is_cutoff (error == CUTOFF) -> None, dropped."""
    s = _make_sample(parsed=None, parse_ok=None, error=CUTOFF)
    assert exam_score(s, frozenset({"A"})) is None


def test_other_error_returns_none() -> None:
    """Non-cutoff errors (e.g. content policy, API timeout) -> None, dropped."""
    for err in ("content_policy", "api_timeout", "bad_request", "network"):
        s = _make_sample(parsed=None, parse_ok=None, error=err)
        assert exam_score(s, frozenset({"A"})) is None, f"error={err}"


def test_parse_ok_zero_returns_zero() -> None:
    """error=None and parse_ok=0 -> 0.0, counted in the basis ("completed but wrong")."""
    s = _make_sample(parsed=None, parse_ok=0)
    assert exam_score(s, frozenset({"A"})) == 0.0


def test_parse_ok_none_returns_zero() -> None:
    """parse_ok being None also takes the 0.0 path (joins the `parse_ok != 1` branch)."""
    s = _make_sample(parsed=None, parse_ok=None)
    assert exam_score(s, frozenset({"A"})) == 0.0


def test_parse_ok_one_but_parsed_none_returns_zero() -> None:
    """parse_ok=1 but parsed_letters resolves to None (defensive) -> 0.0."""
    # final_answer_letters is an empty string -> parsed_letters returns None
    s = SampleRow(
        model="model_x", question_id="q1", question_type="forecast",
        choice_type="multi", options=["A", "B", "C", "D"], sample_idx=0,
        correct=None, parse_ok=1,
        tool_calls_count=0, react_steps=0,
        prompt_tokens=0, completion_tokens=0, reasoning_tokens=0,
        latency_ms=0,
        final_answer_letters="",  # → parsed_letters = None
        error=None, created_at="2026-04-27T00:00:00Z",
        finish_reason="stop", nudges_used=0,
        belief_final=None, belief_trace=None, belief_parse_ok=0,
        probabilities=None, is_fallback=False,
    )
    assert s.parsed_letters is None
    assert exam_score(s, frozenset({"A"})) == 0.0


def test_normal_path_does_not_hit_defensive_branches() -> None:
    """Normal path: error=None, parse_ok=1, parsed valid -> applies the formula."""
    s = _make_sample(parsed=frozenset({"A", "B"}))
    assert exam_score(s, frozenset({"A", "B", "C"})) == pytest.approx(2 / 3, abs=1e-4)


# --------------------------------------------------------------------------- #
# Section §3.4 — Aggregation (in-question denominator = actual count in basis;
# questions with empty basis are dropped; equal weight per question)
# --------------------------------------------------------------------------- #


def test_aggregation_eligible_denominator_is_actual_in_basis() -> None:
    """User's wording: "just average over the remaining 2".
    With SAMPLING_N=3, one is a cutoff, leaving 2 in the basis -> in-question denominator = 2.
    """
    gt = frozenset({"A"})
    samples = [
        _make_sample(question_id="q1", sample_idx=0,
                     parsed=None, parse_ok=None, error=CUTOFF),  # dropped
        _make_sample(question_id="q1", sample_idx=1,
                     choice_type="single", options=["A", "B"],
                     parsed=frozenset({"A"})),
        _make_sample(question_id="q1", sample_idx=2,
                     choice_type="single", options=["A", "B"],
                     parsed=frozenset({"B"})),
    ]
    # Per-question e_q = (1.0 + 0.0) / 2 = 0.5 (cutoff excluded from the denominator)
    result = exam_score_at_n_avg(samples, {"q1": gt})
    assert result == pytest.approx(0.5)


def test_aggregation_error_excluded_from_denominator() -> None:
    """A sample that failed content moderation is excluded from the basis; the rest are averaged."""
    gt = frozenset({"A", "B", "C"})
    samples = [
        _make_sample(question_id="q1", sample_idx=0,
                     parsed=None, parse_ok=None, error="content_policy"),  # dropped
        _make_sample(question_id="q1", sample_idx=1,
                     parsed=frozenset({"A", "B", "C"})),  # 1.0
        _make_sample(question_id="q1", sample_idx=2,
                     parsed=frozenset({"A"})),  # 1/3
    ]
    # e_q = (1.0 + 1/3) / 2 = 2/3
    assert exam_score_at_n_avg(samples, {"q1": gt}) == pytest.approx(2 / 3, abs=1e-4)


def test_aggregation_parse_failure_counts_as_zero() -> None:
    """parse_ok=0 enters the basis as 0.0 — different semantics from "dropped"."""
    gt = frozenset({"A", "B", "C"})
    samples = [
        _make_sample(question_id="q1", sample_idx=0,
                     parsed=None, parse_ok=0),  # 0.0 in basis
        _make_sample(question_id="q1", sample_idx=1,
                     parsed=frozenset({"A", "B", "C"})),  # 1.0
        _make_sample(question_id="q1", sample_idx=2,
                     parsed=frozenset({"A", "B", "C"})),  # 1.0
    ]
    # e_q = (0.0 + 1.0 + 1.0) / 3 = 2/3
    assert exam_score_at_n_avg(samples, {"q1": gt}) == pytest.approx(2 / 3, abs=1e-4)


def test_aggregation_question_with_all_excluded_skipped_globally() -> None:
    """When every sample of a question is dropped -> e_q = None, the question is excluded from the global cross-question denominator."""
    gt_a = frozenset({"A"})
    gt_b = frozenset({"B"})
    samples = [
        # q1: all 3 samples are cutoffs, e_q = None
        _make_sample(question_id="q1", sample_idx=0,
                     parsed=None, parse_ok=None, error=CUTOFF),
        _make_sample(question_id="q1", sample_idx=1,
                     parsed=None, parse_ok=None, error=CUTOFF),
        # q2: both samples are 1.0, e_q = 1.0
        _make_sample(question_id="q2", sample_idx=0,
                     choice_type="single", options=["A", "B"],
                     parsed=frozenset({"B"})),
        _make_sample(question_id="q2", sample_idx=1,
                     choice_type="single", options=["A", "B"],
                     parsed=frozenset({"B"})),
    ]
    # Globally only q2 is counted -> 1.0 (q1 is not in the cross-question denominator)
    result = exam_score_at_n_avg(samples, {"q1": gt_a, "q2": gt_b})
    assert result == pytest.approx(1.0)


def test_aggregation_empty_samples_returns_none() -> None:
    assert exam_score_at_n_avg([], {}) is None


def test_aggregation_all_excluded_returns_none() -> None:
    """All samples dropped (all cutoff/error) -> globally returns None."""
    samples = [
        _make_sample(question_id="q1", sample_idx=0,
                     parsed=None, parse_ok=None, error=CUTOFF),
        _make_sample(question_id="q2", sample_idx=0,
                     parsed=None, parse_ok=None, error="content_policy"),
    ]
    assert exam_score_at_n_avg(
        samples, {"q1": frozenset({"A"}), "q2": frozenset({"B"})}
    ) is None


def test_aggregation_questions_equal_weighted_not_sample_weighted() -> None:
    """Equal weight per question (not weighted by sample count): q1 has 3 samples, q2 has 1 sample;
    the global value = (e_q1 + e_q2) / 2, not sample-weighted."""
    gt = frozenset({"A"})
    samples = [
        # q1: all 3 samples are 0.0 -> e_q1 = 0.0
        _make_sample(question_id="q1", sample_idx=0,
                     choice_type="single", options=["A", "B"],
                     parsed=frozenset({"B"})),
        _make_sample(question_id="q1", sample_idx=1,
                     choice_type="single", options=["A", "B"],
                     parsed=frozenset({"B"})),
        _make_sample(question_id="q1", sample_idx=2,
                     choice_type="single", options=["A", "B"],
                     parsed=frozenset({"B"})),
        # q2: 1 sample is 1.0 -> e_q2 = 1.0
        _make_sample(question_id="q2", sample_idx=0,
                     choice_type="single", options=["A", "B"],
                     parsed=frozenset({"A"})),
    ]
    # Equal-weight per question: (0.0 + 1.0) / 2 = 0.5
    # Sample-weighted would give (0+0+0+1)/4 = 0.25
    result = exam_score_at_n_avg(samples, {"q1": gt, "q2": gt})
    assert result == pytest.approx(0.5)


def test_aggregation_user_two_questions_global() -> None:
    """Combine the user's two examples under global aggregation (equal weight per question): (5/9 + 1/3) / 2 ~= 0.4444."""
    gt_a = frozenset({"A", "B", "C"})
    gt_b = frozenset({"B"})
    samples = [
        _make_sample(question_id="qA", sample_idx=0, parsed=frozenset({"A", "B"})),
        _make_sample(question_id="qA", sample_idx=1, parsed=frozenset({"A", "B", "C"})),
        _make_sample(question_id="qA", sample_idx=2, parsed=frozenset({"A", "D"})),
        _make_sample(question_id="qB", sample_idx=0,
                     choice_type="single", options=["A", "B"],
                     parsed=frozenset({"A"})),
        _make_sample(question_id="qB", sample_idx=1,
                     choice_type="single", options=["A", "B"],
                     parsed=frozenset({"A"})),
        _make_sample(question_id="qB", sample_idx=2,
                     choice_type="single", options=["A", "B"],
                     parsed=frozenset({"B"})),
    ]
    result = exam_score_at_n_avg(samples, {"qA": gt_a, "qB": gt_b})
    expected = (5 / 9 + 1 / 3) / 2
    assert result == pytest.approx(expected, abs=1e-4)


# --------------------------------------------------------------------------- #
# Section §3.5 — Single-choice degenerate equivalence (byte-identical to parser.is_correct)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "pred_letters,gt_letters",
    [
        ({"A"}, {"A"}),
        ({"A"}, {"B"}),
        ({"B"}, {"A"}),
        ({"B"}, {"B"}),
        ({"C"}, {"D"}),
    ],
)
def test_single_choice_degenerates_to_is_correct(pred_letters, gt_letters) -> None:
    pred = frozenset(pred_letters)
    gt = frozenset(gt_letters)
    s = _make_sample(choice_type="single", options=["A", "B", "C", "D"],
                     parsed=pred)
    expected = is_correct(pred, gt)
    assert expected is not None
    assert exam_score(s, gt) == float(expected)


# --------------------------------------------------------------------------- #
# Section §3.6 — Defensive boundaries (empty gt / qid missing from gt_map)
# --------------------------------------------------------------------------- #


def test_empty_gt_returns_zero() -> None:
    """gt is an empty frozenset -> 0.0 (should not appear in the dataset by design; guards against dirty-data NaN)."""
    s = _make_sample(parsed=frozenset({"A"}))
    assert exam_score(s, frozenset()) == 0.0


def test_gt_map_missing_question_id_skipped() -> None:
    """qid missing from gt_map (should not happen by design) -> that question is skipped and excluded from the global aggregate."""
    samples = [
        _make_sample(question_id="qA", sample_idx=0,
                     choice_type="single", options=["A", "B"],
                     parsed=frozenset({"A"})),
        _make_sample(question_id="qB_missing", sample_idx=0,
                     choice_type="single", options=["A", "B"],
                     parsed=frozenset({"B"})),
    ]
    # gt_map only contains qA -> globally only qA is counted
    result = exam_score_at_n_avg(samples, {"qA": frozenset({"A"})})
    assert result == pytest.approx(1.0)


def test_pred_empty_set_with_nonempty_gt_returns_zero() -> None:
    """pred=empty set (no FP and no TP) -> TP/|G| = 0/|G| = 0."""
    # This is the "FP=0 but TP=0" boundary — the formula yields 0/N = 0.0
    s = _make_sample(parsed=frozenset())
    assert exam_score(s, frozenset({"A", "B"})) == pytest.approx(0.0)
