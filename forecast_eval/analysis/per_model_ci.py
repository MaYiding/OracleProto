"""Single-variable bootstrap CI per (model, metric).

Input: a dict `panel: {model_id: [per_q_value_1, ..., per_q_value_N]}`.
Output: list[PerModelCI] with point estimate plus a 2-sided percentile
paired-against-zero bootstrap CI per model, paired by question index.

Reuses `inference.paired_bootstrap(values, zeros)` — single-variable
bootstrap is exactly the paired-against-zero case.
"""
from __future__ import annotations

from dataclasses import dataclass

from .inference import paired_bootstrap


@dataclass(frozen=True)
class PerModelCI:
    model: str
    metric: str
    point: float
    ci_lo: float
    ci_hi: float
    ci_half_width: float
    n_questions: int


def per_model_ci(
    panel: dict[str, list[float]],
    *,
    metric_name: str,
    seed: int = 20260512,
    n_bootstrap: int = 5000,
    ci_alpha: float = 0.05,
) -> list[PerModelCI]:
    """For each model, build a PerModelCI from its per-question array.

    Args:
        panel: `{model_id: [v_q1, v_q2, ..., v_qN]}`. Missing-q questions
            should be filtered upstream so every model has the same N.
        metric_name: pinned into PerModelCI.metric for later joins.

    Returns:
        Sorted list of PerModelCI rows (alphabetical by model).
    """
    out: list[PerModelCI] = []
    for model in sorted(panel.keys()):
        values = panel[model]
        if not values:
            continue
        zeros = [0.0] * len(values)
        res = paired_bootstrap(
            values, zeros,
            n_bootstrap=n_bootstrap, seed=seed, ci_alpha=ci_alpha,
        )
        if res is None or res.n_questions == 0:
            continue
        point = sum(values) / len(values)
        ci_lo = res.ci_low
        ci_hi = res.ci_high
        out.append(PerModelCI(
            model=model,
            metric=metric_name,
            point=point,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
            ci_half_width=(ci_hi - ci_lo) / 2.0,
            n_questions=res.n_questions,
        ))
    return out


__all__ = ["PerModelCI", "per_model_ci"]
