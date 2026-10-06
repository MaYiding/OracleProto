"""Exam-style partial credit with a zero-false-positive gate.

## Formula

For a single sample, denote the question's correct-answer set as $G$ and
the model's parsed option set as $\\hat S$:

$$
\\text{exam\\_score}(\\hat S, G) = \\begin{cases}
|\\hat S \\cap G| / |G| & \\text{if } \\hat S \\setminus G = \\emptyset \\\\
0 & \\text{if } \\hat S \\setminus G \\ne \\emptyset
\\end{cases}
$$

Aggregation uses two steps: per-question mean -> across-question mean.
First compute $e_q$ for each question $q$ over the in-base samples, then
take the equal-weight mean across all questions; an empty base at either
step returns `None`.

## Semantics

Reinterprets SAMPLING_N from this metric's perspective as "the number of
independent exam attempts": each attempt scores independently in 0~1, and
the arithmetic mean is taken at the end. Formally equivalent to
**"Recall under zero-FP gate"** — Recall with an added hard gate of
"any FP triggers a veto".

## Differences from existing scores

| Metric | Formula | FP penalty | FN penalty | Single-choice degeneracy |
| --- | --- | --- | --- | --- |
| `parser.is_correct` (strict) | only $\\hat S = G$ scores 1 | 0 | 0 | 0/1 |
| `tversky_score(alpha=2,beta=0.5)`+chance correction -> `fss` | soft FP/FN penalty | alpha-fold | beta-fold | with chance |
| `hamming_score` | $1 - \\text{XOR-bits}/k$ | symmetric to FN | symmetric to FP | 0/1 |
| **`exam_score` (this file)** | TP/|G|·1(FP=0) | veto | proportional deduction | 0/1 |

Any false-positive selection scores zero; otherwise credit is the
proportion of correct options selected.

The implementation depends only on the standard library and the sample-row
helpers in `flatten.py`.
"""
from __future__ import annotations

from .flatten import SampleRow, _group_by_question


def exam_score(s: SampleRow, gt: frozenset[str]) -> float | None:
    """Per-sample exam-style score; included in base -> float in [0,1],
    skipped -> `None`.

    Decision order:
      1. `is_cutoff` (question dated after training cutoff) -> `None`,
         skipped (information barrier);
      2. `error is not None` and not cutoff -> `None`, skipped
         ("process did not complete");
      3. `error is None` and `parse_ok != 1` -> `0.0`, included in base
         ("completed but answered wrong");
      4. `parsed_letters is None` (defensive) -> `0.0`, included in base;
      5. Contains FP (FP > 0) -> `0.0`, included in base;
      6. Defensive `gt` empty -> `0.0`;
      7. Otherwise -> $|\\hat S \\cap G| / |G|$, included in base.
    """
    if s.is_cutoff:
        return None
    if s.error is not None:
        return None
    if s.parse_ok != 1:
        return 0.0
    pred = s.parsed_letters
    if pred is None:
        return 0.0
    if pred - gt:
        return 0.0
    if not gt:
        return 0.0
    return len(pred & gt) / len(gt)


def exam_score_at_n_avg(
    samples: list[SampleRow],
    gt_map: dict[str, frozenset[str]],
) -> float | None:
    """Two-step aggregation: per-question mean -> across-question mean.

    Step 1 (per-question): for each question $q$, take the arithmetic mean
    of `exam_score` over in-base samples to get $e_q$; if every sample of
    that question is skipped (base of 0), $e_q = \\text{None}$ does not
    participate in the global.

    Step 2 (across-question): take the equal-weight mean across all
    questions where $e_q \\ne \\text{None}$; if the global base is 0,
    return `None` (not 0.0 / NaN / raising).

    When `gt_map` is missing this question_id, the question is skipped
    (defensive edge; not expected in practice).
    """
    by_q = _group_by_question(samples)
    e_q_values: list[float] = []
    for qid, q_samples in by_q.items():
        gt = gt_map.get(qid)
        if gt is None:
            continue
        per_sample: list[float] = []
        for s in q_samples:
            score = exam_score(s, gt)
            if score is None:
                continue
            per_sample.append(score)
        if not per_sample:
            continue
        e_q_values.append(sum(per_sample) / len(per_sample))
    if not e_q_values:
        return None
    return sum(e_q_values) / len(e_q_values)


__all__ = [
    "exam_score",
    "exam_score_at_n_avg",
]
