"""Accuracy-side metrics: pass@1 / pass_any@N / majority_vote / breakdowns,
plus the discrete-native family (Tversky per-sample score, FSS three-step
aggregate, Cohen's κ, Hamming partial-credit). The probabilistic family
lives in `proper_score.py`.

The discrete-native functions are kept separate from the `Aggregate`
dataclass so callers that only need pass@1 / breakdowns are unaffected.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from ..prompts import index_to_letter
from .flatten import SampleRow
from . import exam_score as _exam_score


def _mean(values: Iterable[float | int]) -> float | None:
    collected = [float(v) for v in values if v is not None]
    if not collected:
        return None
    return sum(collected) / len(collected)


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(value, digits)


@dataclass
class Aggregate:
    eligible_samples: int
    eligible_questions: int
    resolvable_samples: int

    cutoff_skip_samples: int
    cutoff_skip_rate: float | None

    pass_at_1_avg: float | None
    resolvable_rate: float | None

    pass_any_at_n: float | None
    at_least_majority_at_n: float | None
    at_least_all_at_n: float | None

    exam_score_at_n_avg: float | None

    majority_vote_accuracy: float | None
    majority_vote_resolvable_rate: float | None

    parse_failure_rate: float | None
    error_rate: float | None

    avg_tool_calls: float | None
    avg_react_steps: float | None
    avg_latency_ms: float | None
    avg_prompt_tokens: float | None
    avg_completion_tokens: float | None
    avg_reasoning_tokens: float | None
    avg_nudges_used: float | None

    # Share of eligible samples whose ReAct loop exited with empty
    # `final_raw` and triggered the no-tool bail-out retry. `None` when no
    # eligible sample populates the column (legacy DB — all-NULL → division
    # denominator is 0). The column is NULL on legacy rows; we count
    # `final_answer_retry_used == 1` exactly so legacy NULLs contribute 0
    # to the numerator and the rate stays meaningful.
    final_answer_retry_rate: float | None

    def as_ordered_dict(self) -> dict[str, Any]:
        return {
            "eligible_samples": self.eligible_samples,
            "eligible_questions": self.eligible_questions,
            "resolvable_samples": self.resolvable_samples,
            "cutoff_skip_samples": self.cutoff_skip_samples,
            "cutoff_skip_rate": _round(self.cutoff_skip_rate),
            "pass_at_1_avg": _round(self.pass_at_1_avg),
            "resolvable_rate": _round(self.resolvable_rate),
            "pass_any_at_n": _round(self.pass_any_at_n),
            "at_least_majority_at_n": _round(self.at_least_majority_at_n),
            "at_least_all_at_n": _round(self.at_least_all_at_n),
            "exam_score_at_n_avg": _round(self.exam_score_at_n_avg),
            "majority_vote_accuracy": _round(self.majority_vote_accuracy),
            "majority_vote_resolvable_rate": _round(self.majority_vote_resolvable_rate),
            "parse_failure_rate": _round(self.parse_failure_rate),
            "error_rate": _round(self.error_rate),
            "avg_tool_calls": _round(self.avg_tool_calls, 2),
            "avg_react_steps": _round(self.avg_react_steps, 2),
            "avg_latency_ms": _round(self.avg_latency_ms, 1),
            "avg_prompt_tokens": _round(self.avg_prompt_tokens, 1),
            "avg_completion_tokens": _round(self.avg_completion_tokens, 1),
            "avg_reasoning_tokens": _round(self.avg_reasoning_tokens, 1),
            "avg_nudges_used": _round(self.avg_nudges_used, 2),
            "final_answer_retry_rate": _round(self.final_answer_retry_rate),
        }


def _aggregate(
    samples: list[SampleRow],
    sampling_n: int,
    gt_map: dict[str, frozenset[str]] | None = None,
) -> Aggregate:
    total = len(samples)
    cutoff_samples = [s for s in samples if s.is_cutoff]
    eligible_samples = [s for s in samples if s.is_eligible]
    resolvable_samples = [s for s in eligible_samples if s.is_resolvable]

    eligible_questions = {s.question_id for s in eligible_samples}
    by_q_resolvable: dict[str, list[SampleRow]] = {}
    by_q_all_eligible: dict[str, list[SampleRow]] = {}
    for s in eligible_samples:
        by_q_all_eligible.setdefault(s.question_id, []).append(s)
        if s.is_resolvable:
            by_q_resolvable.setdefault(s.question_id, []).append(s)

    if resolvable_samples:
        pass_at_1 = sum(1 for s in resolvable_samples if s.correct == 1) / len(resolvable_samples)
    else:
        pass_at_1 = None

    resolvable_rate = (
        len(resolvable_samples) / len(eligible_samples) if eligible_samples else None
    )

    majority_threshold = math.ceil(sampling_n / 2)

    pass_any_hits: list[int] = []
    at_least_majority_hits: list[int] = []
    at_least_all_hits: list[int] = []
    for qid, rs in by_q_resolvable.items():
        n = len(rs)
        corrects = sum(1 for s in rs if s.correct == 1)
        pass_any_hits.append(1 if corrects >= 1 else 0)
        at_least_majority_hits.append(1 if corrects >= majority_threshold else 0)
        at_least_all_hits.append(1 if (n == sampling_n and corrects == n) else 0)

    pass_any_at_n = _mean(pass_any_hits) if pass_any_hits else None
    at_least_majority_at_n = _mean(at_least_majority_hits) if at_least_majority_hits else None
    at_least_all_at_n = _mean(at_least_all_hits) if at_least_all_hits else None

    mv_resolvable_hits: list[int] = []
    mv_correct_hits: list[int] = []
    if gt_map is not None:
        for qid, rs in by_q_all_eligible.items():
            gt = gt_map.get(qid)
            if gt is None:
                continue
            parsed = [s.parsed_letters for s in rs if s.parsed_letters is not None]
            if not parsed:
                continue
            counts = Counter(parsed)
            top_count = max(counts.values())
            winners = [k for k, v in counts.items() if v == top_count]
            if len(winners) != 1:
                continue
            mv_resolvable_hits.append(1)
            mv_correct_hits.append(1 if winners[0] == gt else 0)
        majority_vote_accuracy = _mean(mv_correct_hits) if mv_correct_hits else None
        majority_vote_resolvable_rate = (
            len(mv_resolvable_hits) / len(eligible_questions) if eligible_questions else None
        )
    else:
        majority_vote_accuracy = None
        majority_vote_resolvable_rate = None

    if eligible_samples:
        parse_failure_rate = sum(
            1 for s in eligible_samples if s.parse_ok == 0 and (s.error is None)
        ) / len(eligible_samples)
        error_rate = sum(
            1 for s in eligible_samples if s.error is not None
        ) / len(eligible_samples)
    else:
        parse_failure_rate = None
        error_rate = None

    avg_tool_calls = _mean(s.tool_calls_count for s in eligible_samples)
    avg_react_steps = _mean(s.react_steps for s in eligible_samples)
    avg_latency = _mean(s.latency_ms for s in eligible_samples)
    avg_ptok = _mean(s.prompt_tokens for s in eligible_samples)
    avg_ctok = _mean(s.completion_tokens for s in eligible_samples)
    avg_rtok = _mean(s.reasoning_tokens for s in eligible_samples)
    # Pre-v3 rows have nudges_used=NULL — _mean already filters those, so a
    # mid-run schema upgrade silently averages over the v3 rows only.
    avg_nudges = _mean(s.nudges_used for s in eligible_samples)
    # `final_answer_retry_rate`: share of eligible samples that triggered
    # the bail-out retry. Denominator counts only rows where the column is
    # populated (NOT NULL); legacy DBs without the column have all NULLs
    # → denominator 0 → None (we do not pretend the feature was off; we
    # mark the column as unobserved). New rows always carry 0 or 1.
    retry_observed = [
        s.final_answer_retry_used for s in eligible_samples
        if s.final_answer_retry_used is not None
    ]
    if retry_observed:
        final_answer_retry_rate: float | None = (
            sum(1 for v in retry_observed if v == 1) / len(retry_observed)
        )
    else:
        final_answer_retry_rate = None

    exam_value = (
        _exam_score.exam_score_at_n_avg(eligible_samples, gt_map)
        if gt_map is not None
        else None
    )

    return Aggregate(
        eligible_samples=len(eligible_samples),
        eligible_questions=len(eligible_questions),
        resolvable_samples=len(resolvable_samples),
        cutoff_skip_samples=len(cutoff_samples),
        cutoff_skip_rate=(len(cutoff_samples) / total) if total else None,
        pass_at_1_avg=pass_at_1,
        resolvable_rate=resolvable_rate,
        pass_any_at_n=pass_any_at_n,
        at_least_majority_at_n=at_least_majority_at_n,
        at_least_all_at_n=at_least_all_at_n,
        exam_score_at_n_avg=exam_value,
        majority_vote_accuracy=majority_vote_accuracy,
        majority_vote_resolvable_rate=majority_vote_resolvable_rate,
        parse_failure_rate=parse_failure_rate,
        error_rate=error_rate,
        avg_tool_calls=avg_tool_calls,
        avg_react_steps=avg_react_steps,
        avg_latency_ms=avg_latency,
        avg_prompt_tokens=avg_ptok,
        avg_completion_tokens=avg_ctok,
        avg_reasoning_tokens=avg_rtok,
        avg_nudges_used=avg_nudges,
        final_answer_retry_rate=final_answer_retry_rate,
    )


def _slice_by(
    samples: list[SampleRow],
    key_fn: Callable[[SampleRow], str],
    sampling_n: int,
    gt_map: dict[str, frozenset[str]],
) -> dict[str, Aggregate]:
    buckets: dict[str, list[SampleRow]] = {}
    for s in samples:
        buckets.setdefault(key_fn(s), []).append(s)
    return {k: _aggregate(v, sampling_n, gt_map) for k, v in sorted(buckets.items())}


def _error_breakdown(samples: list[SampleRow]) -> Counter:
    """Count error codes across ALL samples (including cutoff)."""
    counter: Counter = Counter()
    for s in samples:
        if s.error is not None:
            counter[s.error] += 1
        else:
            counter["<ok>"] += 1
    return counter


def _finish_reason_breakdown(samples: list[SampleRow]) -> Counter:
    """Count `finish_reason` values across eligible samples (cutoff excluded).

    The cutoff path never invokes the LLM, so its `finish_reason` is always
    NULL — including those rows would just inflate the `<missing>` bucket.
    A NULL on an eligible sample (legacy v2 row, or pre-extraction failure)
    is rare but real, so we keep it as `<missing>` rather than silently dropping.
    """
    counter: Counter = Counter()
    for s in samples:
        if s.is_eligible:
            counter[s.finish_reason or "<missing>"] += 1
    return counter


# --------------------------------------------------------------------------- #
# Discrete-native family: Tversky / FSS / Cohen κ / Hamming
# --------------------------------------------------------------------------- #


def tversky_score(
    pred: frozenset[str],
    gt: frozenset[str],
    *,
    alpha: float = 2.0,
    beta: float = 0.5,
) -> float:
    """Tversky 1977 set similarity: $|TP|/(|TP| + α \\cdot |FP| + β \\cdot |FN|)$.

    Default $(α, β) = (2, 0.5)$ makes a multi-selection error 4× as costly
    as a missed selection — consistent with the prediction-domain intuition
    that "asserting an event will happen" is more harmful than "missing one".

    `TP=0` returns 0.0, including the "model output empty set" boundary
    (anti-conservative — conservative no-selection is NOT rewarded).

    `gt` empty (degenerate, should not occur in this dataset) returns 1.0
    iff `pred` is also empty, else 0.0 (defensive).
    """
    if not gt:
        return 1.0 if not pred else 0.0
    tp = len(pred & gt)
    if tp == 0:
        return 0.0
    fp = len(pred - gt)
    fn = len(gt - pred)
    denom = tp + alpha * fp + beta * fn
    return tp / denom


def tversky_baseline(
    k: int,
    m: int,
    *,
    alpha: float = 2.0,
    beta: float = 0.5,
) -> float:
    """Expected Tversky for a uniform random predictor on a $k$-option, $m$-positive question.

    Each label is included in $\\hat{S}$ independently with probability 0.5.
    $|TP| \\sim \\mathrm{Binom}(m, 0.5)$ and $|FP| \\sim \\mathrm{Binom}(k-m, 0.5)$
    are independent, so the expectation factorises:

    $$\\mathbb{E}[\\mathrm{Tversky}] = \\sum_{tp=1}^{m} \\sum_{fp=0}^{k-m}
        \\binom{m}{tp} 2^{-m} \\binom{k-m}{fp} 2^{-(k-m)}
        \\cdot \\frac{tp}{tp + α fp + β (m - tp)}$$

    The $tp=0$ stratum contributes 0 (Tversky returns 0 when $TP=0$). Total
    work is $O(m \\times (k - m))$; $k=35, m=10$ runs in well under 1ms.

    Returns 0.0 for degenerate $k \\le 0$, $m \\le 0$, $m > k$ inputs (the
    caller should guard, but defensive zeroing keeps FSS finite).
    """
    if k <= 0 or m <= 0 or m > k:
        return 0.0
    expectation = 0.0
    p_m = 0.5 ** m
    p_km = 0.5 ** (k - m) if k > m else 1.0
    for tp in range(1, m + 1):
        p_tp = math.comb(m, tp) * p_m
        for fp in range(k - m + 1):
            p_fp = math.comb(k - m, fp) * p_km
            denom = tp + alpha * fp + beta * (m - tp)
            expectation += p_tp * p_fp * (tp / denom)
    return expectation


def _per_question_tversky(
    samples_by_q: dict[str, list[SampleRow]],
    gt_map: dict[str, frozenset[str]],
    *,
    alpha: float = 2.0,
    beta: float = 0.5,
) -> dict[str, tuple[float, int]]:
    """Per-question $c_q$ = mean Tversky over parse_ok trials, plus $K_{\\text{eff}}$.

    Returns `{qid: (c_q, K_eff)}`. Questions with $K_{\\text{eff}} = 0$ (no
    sample produced a parsed letter set) are silently dropped — they
    contribute no signal to FSS and the chance correction would be undefined.
    """
    out: dict[str, tuple[float, int]] = {}
    for qid, samples in samples_by_q.items():
        gt = gt_map.get(qid)
        if gt is None:
            continue
        scores: list[float] = []
        for s in samples:
            if not s.is_eligible:
                continue
            pred = s.parsed_letters
            if pred is None:
                continue
            scores.append(tversky_score(pred, gt, alpha=alpha, beta=beta))
        if not scores:
            continue
        c_q = sum(scores) / len(scores)
        out[qid] = (c_q, len(scores))
    return out


def fss(
    samples: list[SampleRow] | dict[str, list[SampleRow]],
    gt_map: dict[str, frozenset[str]],
    *,
    alpha: float = 2.0,
    beta: float = 0.5,
) -> dict[str, Any]:
    """Forecast Skill Score (Tversky-based, chance-corrected).

    Three-step aggregate (plan §1.7):

    1. per-sample Tversky $\\text{score}_{q,k}$;
    2. per-question $c_q = \\frac{1}{K_{\\text{eff}}} \\sum_k \\text{score}_{q,k}$
       (parse-ok trials only; $K_{\\text{eff}} = 0$ → question dropped);
    3. chance correction $s_q = (c_q - p_{e,q}) / (1 - p_{e,q})$ where
       $p_{e,q}$ = $1/k_q$ for single-choice questions and
       `tversky_baseline(k_q, m_q)` for multi-label questions;
    4. FSS = global mean over $q$ of $s_q$.

    Returns `{"fss", "n_valid", "mean_pe", "per_question": {qid: {c_q, p_e, s_q, K_eff}}}`.
    Empty input returns FSS=None, n_valid=0; that's the "no scoreable
    questions" sentinel for the writer (CSV cell becomes blank).

    Accepts either a flat `list[SampleRow]` (the natural shape from
    `_flatten_db`, which we group by `question_id` here) or an already-grouped
    `dict[str, list[SampleRow]]`. The dict form lets `inference.metric_paired_bootstrap`
    pass bootstrap-resampled subsets keyed by a unique resample-index without
    sample-level question_id collisions.
    """
    if isinstance(samples, dict):
        by_q = samples
    else:
        by_q = {}
        for s in samples:
            by_q.setdefault(s.question_id, []).append(s)

    per_q_dict = _per_question_tversky(by_q, gt_map, alpha=alpha, beta=beta)

    per_question: dict[str, dict[str, Any]] = {}
    s_q_values: list[float] = []
    p_e_values: list[float] = []

    for qid, (c_q, k_eff) in per_q_dict.items():
        sample = by_q[qid][0]
        ctype = sample.choice_type
        options = sample.options
        if not options:
            continue
        k = len(options)
        gt = gt_map.get(qid)
        if gt is None:
            continue
        m = len(gt)

        if ctype == "single":
            # Single-choice: random guess hit rate is 1/k (plan §1.6).
            p_e = 1.0 / k
        else:
            # Multi-label: closed-form expected Tversky for a uniform
            # 0.5-per-label random predictor.
            p_e = tversky_baseline(k, m, alpha=alpha, beta=beta)

        if 1.0 - p_e > 1e-12:
            s_q = (c_q - p_e) / (1.0 - p_e)
        else:
            # Degenerate p_e ≥ 1 (shouldn't happen for legitimate k/m); treat
            # as no skill possible.
            s_q = 0.0

        per_question[qid] = {
            "c_q": c_q,
            "p_e": p_e,
            "s_q": s_q,
            "K_eff": k_eff,
        }
        s_q_values.append(s_q)
        p_e_values.append(p_e)

    if not s_q_values:
        return {
            "fss": None,
            "n_valid": 0,
            "mean_pe": None,
            "per_question": {},
        }

    fss_value = sum(s_q_values) / len(s_q_values)
    mean_pe = sum(p_e_values) / len(p_e_values)
    return {
        "fss": fss_value,
        "n_valid": len(s_q_values),
        "mean_pe": mean_pe,
        "per_question": per_question,
    }


def cohen_kappa_for_aggregate(acc: float, p_e: float) -> float | None:
    """Plain Cohen's κ on a binary outcome: $(\\mathrm{acc} - p_e) / (1 - p_e)$.

    `p_e = 1.0` returns None (degenerate, no skill is measurable). Negative
    return values are allowed (acc < p_e → worse than chance).
    """
    if abs(1.0 - p_e) < 1e-12:
        return None
    return (acc - p_e) / (1.0 - p_e)


def cohen_kappa(
    samples_by_q: dict[str, list[SampleRow]],
    gt_map: dict[str, frozenset[str]],
) -> float | None:
    """Cohen's κ on strict per-sample correct/wrong outcomes.

    Per-question $p_{e,q}$ rule:
    * `single`: $1/k_q$ (random guess hit rate);
    * `multi`: $0.5$ (per-label coin-flip simplification — a strict $0.5^{k_q}$
      chance baseline becomes vanishingly small for large $k_q$ and produces
      κ values nearly identical to raw acc, defeating the purpose of chance
      correction; per-label 0.5 keeps κ on the same order as the single-choice
      version).

    Sample-weighted (so a question with all 5 trials counts 5×). Returns None
    if no resolvable samples exist.
    """
    n_correct = 0
    n_total = 0
    p_e_sum = 0.0
    for qid, samples in samples_by_q.items():
        gt = gt_map.get(qid)
        if gt is None or not samples or not samples[0].options:
            continue
        k = len(samples[0].options)
        if k == 0:
            continue
        ctype = samples[0].choice_type
        p_e_q = 1.0 / k if ctype == "single" else 0.5
        for s in samples:
            if s.correct is None:
                continue
            n_correct += 1 if s.correct == 1 else 0
            n_total += 1
            p_e_sum += p_e_q
    if n_total == 0:
        return None
    acc = n_correct / n_total
    p_e = p_e_sum / n_total
    return cohen_kappa_for_aggregate(acc, p_e)


def hamming_score_per_question(
    pred: frozenset[str],
    gt: frozenset[str],
    options: list[str],
) -> float:
    """Hamming partial-credit: $1 - \\frac{1}{k}\\sum_l |\\hat{y}_l - o_l|$.

    Per-label 0/1 mismatch rate, then complement. Bounded $[0, 1]$. Empty
    options returns 0.0 (degenerate; caller should guard).
    """
    if not options:
        return 0.0
    k = len(options)
    mismatches = 0
    for i in range(k):
        letter = index_to_letter(i)
        in_pred = letter in pred
        in_gt = letter in gt
        if in_pred != in_gt:
            mismatches += 1
    return 1.0 - mismatches / k


def hamming_score(
    samples: list[SampleRow],
    gt_map: dict[str, frozenset[str]],
) -> float | None:
    """Mean Hamming partial-credit across multi-label samples (single-only run → None).

    Skips: ineligible samples, parse_ok=0, missing GT. A run with zero
    multi-label questions returns None — single-choice partial-credit
    degenerates to strict 0/1 (which already lives in `pass_at_1_avg`).
    """
    scores: list[float] = []
    for s in samples:
        if s.choice_type != "multi":
            continue
        if not s.is_eligible or s.parse_ok != 1:
            continue
        pred = s.parsed_letters
        if pred is None:
            continue
        gt = gt_map.get(s.question_id)
        if gt is None:
            continue
        scores.append(hamming_score_per_question(pred, gt, s.options))
    if not scores:
        return None
    return sum(scores) / len(scores)


__all__ = [
    "Aggregate",
    "_aggregate",
    "_slice_by",
    "_error_breakdown",
    "_finish_reason_breakdown",
    "_mean",
    "_round",
    # Discrete-native family
    "tversky_score",
    "tversky_baseline",
    "fss",
    "cohen_kappa",
    "cohen_kappa_for_aggregate",
    "hamming_score",
    "hamming_score_per_question",
]
