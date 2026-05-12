"""CSV / Markdown / JSON serialisation for the analysis layer.

Probabilistic columns append at the end of `_SUMMARY_FIELDS`; accuracy
columns stay byte-identical so a legacy `per_model_summary.csv` replays
cleanly under newer code. `error_breakdown.csv` /
`finish_reason_breakdown.csv` are not touched at all.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .accuracy import Aggregate
from .aggregation import ShrinkageResult
from .behavior import (
    BeliefEvolutionRow,
    ConfidenceCalibrationRow,
    NumericConfidenceCalibrationRow,
    PDPRow,
    ReflectionABRow,
)
from .composite import CompositeReport, V5SliceResult, _v5_slice_to_columns
from .consistency import ConsistencyReport
from .inference import MetricBootstrapResult, ModelPairResult, PairedBootstrapResult
from .proper_score import ModelProbabilisticAggregate

if TYPE_CHECKING:
    # `grid` module imports writers at module scope; importing it back here
    # would create a top-level cycle. Annotations are deferred via
    # `from __future__ import annotations`, so the string-form forward
    # references in the signatures below resolve only when type checkers
    # introspect them.
    from .grid import GridCell, WinrateRow


# v3 accuracy columns — DO NOT reorder; only append at the tail.
# `final_answer_retry_rate` appends at the tail of the v3 group: keep it
# adjacent to the other "what did the harness do?" diagnostics rather
# than mixed in with FSS / probabilistic columns.
_SUMMARY_FIELDS_V3: tuple[str, ...] = (
    "model",
    "sampling_n",
    "eligible_samples",
    "eligible_questions",
    "resolvable_samples",
    "cutoff_skip_samples",
    "cutoff_skip_rate",
    "pass_at_1_avg",
    "resolvable_rate",
    "pass_any_at_n",
    "at_least_majority_at_n",
    "at_least_all_at_n",
    "exam_score_at_n_avg",  # exam-score-metric: hook 1/2 (CSV header)
    "majority_vote_accuracy",
    "majority_vote_resolvable_rate",
    "parse_failure_rate",
    "error_rate",
    "avg_tool_calls",
    "avg_react_steps",
    "avg_latency_ms",
    "avg_prompt_tokens",
    "avg_completion_tokens",
    "avg_reasoning_tokens",
    "avg_nudges_used",
    "final_answer_retry_rate",
)

# Discrete-native primary columns. Inserted between accuracy and
# probabilistic so the published main metric (FSS) sits next to pass@1 — the
# two of them together drive the "model X vs Y" comparison.
_DISCRETE_FSS_FIELDS: tuple[str, ...] = (
    "fss",
    "fss_pe_mean",
    "cohen_kappa",
    "hamming_score",
)

# K-trial consistency family. Appended right after the FSS family — these
# are the metrics that exist *because* the project does parallel sampling.
_CONSISTENCY_FIELDS: tuple[str, ...] = (
    "fleiss_kappa",
    "mean_entropy",
    "vci",
    "mvg",
)

# Probabilistic columns. Kept as companion columns with a K=5 disclaimer
# in the markdown caption. `bi_dec` only meaningful
# for single-choice subsets; on a global sheet it averages BI_dec over single
# questions only and skips multi rows (same NULL-on-multi convention as MBS).
_PROB_FIELDS_V4: tuple[str, ...] = (
    "bi",
    "bi_dec",
    "nll",
    "mbs",
    "abi_crowd",
    "abi_uniform",
    "fallback_share",
)

_SUMMARY_FIELDS: tuple[str, ...] = (
    _SUMMARY_FIELDS_V3
    + _DISCRETE_FSS_FIELDS
    + _CONSISTENCY_FIELDS
    + _PROB_FIELDS_V4
)


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def _prob_dict(agg: ModelProbabilisticAggregate | None) -> dict[str, Any]:
    """Map a probabilistic aggregate onto the CSV/MD column names.

    Returns all-None when `agg is None` (caller didn't compute probabilistic
    metrics for this row, e.g. a slice with zero scoreable questions). This
    keeps the CSV row width constant.
    """
    if agg is None:
        return {k: None for k in _PROB_FIELDS_V4}
    return {
        "bi": _round(agg.bi),
        "bi_dec": _round(agg.bi_dec),
        "nll": _round(agg.nll),
        "mbs": _round(agg.mbs),
        "abi_crowd": _round(agg.abi_crowd),
        "abi_uniform": _round(agg.abi_uniform),
        "fallback_share": _round(agg.fallback_share),
    }


def _fss_dict(fss_result: dict[str, Any] | None) -> dict[str, Any]:
    """Project a single model's `accuracy.fss(...)` output onto two columns:
    `fss` and `fss_pe_mean`. None when no scoreable questions."""
    if fss_result is None:
        return {"fss": None, "fss_pe_mean": None}
    return {
        "fss": _round(fss_result.get("fss")),
        "fss_pe_mean": _round(fss_result.get("mean_pe")),
    }


def _consistency_dict(rep: ConsistencyReport | None) -> dict[str, Any]:
    """Project a `ConsistencyReport` onto its 4 CSV column values."""
    if rep is None:
        return {k: None for k in _CONSISTENCY_FIELDS}
    return {
        "fleiss_kappa": _round(rep.fleiss_kappa),
        "mean_entropy": _round(rep.mean_entropy),
        "vci": _round(rep.vci),
        "mvg": _round(rep.mvg),
    }


def _v5_extras_dict(
    fss_result: dict[str, Any] | None,
    cohen_kappa_value: float | None,
    hamming_score_value: float | None,
    consistency: ConsistencyReport | None,
) -> dict[str, Any]:
    """Bundle the 8 discrete-native columns (FSS family + Consistency family) for one row.

    Splits stay aligned with `_DISCRETE_FSS_FIELDS + _CONSISTENCY_FIELDS` so
    the CSV writer doesn't need to remember per-field alignment.
    """
    out = {}
    out.update(_fss_dict(fss_result))
    out["cohen_kappa"] = _round(cohen_kappa_value)
    out["hamming_score"] = _round(hamming_score_value)
    out.update(_consistency_dict(consistency))
    return out


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)
    return path


def _write_per_model_summary_csv(
    path: Path,
    per_model: dict[str, tuple[int, Aggregate]],
    prob: dict[str, ModelProbabilisticAggregate] | None = None,
    fss_per_model: dict[str, dict[str, Any]] | None = None,
    cohen_kappa_per_model: dict[str, float | None] | None = None,
    hamming_per_model: dict[str, float | None] | None = None,
    consistency_per_model: dict[str, ConsistencyReport] | None = None,
) -> Path:
    """Per-model headline CSV with accuracy + discrete-native + probabilistic
    columns in that order.

    Discrete-native dict args are keyed by model name. Missing keys land as
    None (column present, value blank) — keeps CSV width constant across rows.
    """
    header = list(_SUMMARY_FIELDS)
    rows: list[list[Any]] = []
    for model, (sampling_n, agg) in per_model.items():
        prob_agg = prob.get(model) if prob else None
        fss_result = fss_per_model.get(model) if fss_per_model else None
        cohen_v = cohen_kappa_per_model.get(model) if cohen_kappa_per_model else None
        hamming_v = hamming_per_model.get(model) if hamming_per_model else None
        consistency = (
            consistency_per_model.get(model) if consistency_per_model else None
        )
        row_dict = {
            "model": model,
            "sampling_n": sampling_n,
            **agg.as_ordered_dict(),
            **_v5_extras_dict(fss_result, cohen_v, hamming_v, consistency),
            **_prob_dict(prob_agg),
        }
        rows.append([row_dict.get(k) for k in header])
    return _write_csv(path, header, rows)


def _write_slice_csv(
    path: Path,
    slice_header_field: str,
    per_model: dict[str, tuple[int, dict[str, Aggregate]]],
    prob: dict[str, dict[str, ModelProbabilisticAggregate]] | None = None,
    v5_slice: dict[str, dict[str, V5SliceResult]] | None = None,
) -> Path:
    """Per-(model, slice_key) CSV.

    Slice tables carry the full ``_SUMMARY_FIELDS`` schema: accuracy /
    discrete family / consistency family / probabilistic — every column from
    ``per_model_summary.csv`` is present, computed on the slice subset.

    Discrete-native + consistency columns require per-bucket recompute via
    :func:`composite.slice_v5_metrics_by_bucket`; when ``v5_slice=None``
    (legacy callers, or buckets without slice data) those columns fall back
    to ``None`` so the header schema stays uniform.
    """
    header = ["model", slice_header_field, "sampling_n", *[
        f for f in _SUMMARY_FIELDS if f not in ("model", "sampling_n")
    ]]
    rows: list[list[Any]] = []
    for model, (sampling_n, agg_map) in per_model.items():
        prob_for_model = prob.get(model) if prob else None
        v5_for_model = v5_slice.get(model) if v5_slice else None
        for key, agg in agg_map.items():
            prob_agg = prob_for_model.get(key) if prob_for_model else None
            v5_res = v5_for_model.get(key) if v5_for_model else None
            v5_cols = _v5_slice_to_columns(v5_res)
            row_dict = {
                "model": model,
                slice_header_field: key,
                "sampling_n": sampling_n,
                **agg.as_ordered_dict(),
                # Discrete-native + consistency columns recomputed on the bucket
                # subset (None when v5_slice is absent or the bucket carries no
                # parsed trials).
                **{k: _round(v5_cols.get(k)) for k in _DISCRETE_FSS_FIELDS},
                **{k: _round(v5_cols.get(k)) for k in _CONSISTENCY_FIELDS},
                **_prob_dict(prob_agg),
            }
            rows.append([row_dict.get(k) for k in header])
    return _write_csv(path, header, rows)


# --------------------------------------------------------------------------- #
# composite-score-by-subtype writers
# --------------------------------------------------------------------------- #


def _write_per_model_composite_csv(
    path: Path,
    report: CompositeReport,
    sampling_n_by_model: dict[str, int],
) -> Path:
    """Composite-score summary table with one row per (model).

    Column order = ``model`` / ``sampling_n`` / ``weights_kind`` / the
    remaining data columns from ``_SUMMARY_FIELDS``. Downstream scripts
    reading this table only need to swap "the original
    ``per_model_summary.csv`` path" for this table's path; column names
    align one-to-one (aside from the added ``weights_kind``).

    ``weights_kind``:
    * ``default`` — every metric for this (model) uses the global default
      weights;
    * ``overridden`` — any metric uses an override value from
      ``COMPOSITE_WEIGHT_OVERRIDES_*``.
    """
    data_fields = [
        f for f in _SUMMARY_FIELDS if f not in ("model", "sampling_n")
    ]
    header = ["model", "sampling_n", "weights_kind", *data_fields]
    rows: list[list[Any]] = []
    for model in sorted(report.per_model.keys()):
        sampling_n = sampling_n_by_model.get(model, "")
        kind = "overridden" if report.is_overridden(model) else "default"
        per_metric = report.per_model[model]
        cells = [model, sampling_n, kind]
        for field in data_fields:
            info = per_metric.get(field)
            cells.append(_round(info.value) if info is not None else None)
        rows.append(cells)
    return _write_csv(path, header, rows)


def _write_composite_meta_json(
    path: Path,
    report: CompositeReport,
) -> Path:
    """Audit trail: write a weights snapshot plus every (model, metric)'s
    buckets_used / weights_used_normalized / value / bucket_values.

    Coexists with ``per_model_composite.csv`` in the same analysis directory;
    for any column in the CSV, the JSON shows which buckets the composite
    actually used, the normalized weights, and the raw per-bucket slice
    values.
    """
    per_model: dict[str, dict[str, Any]] = {}
    for model in sorted(report.per_model.keys()):
        per_metric_out: dict[str, Any] = {}
        for metric, info in sorted(report.per_model[model].items()):
            per_metric_out[metric] = {
                "value": _round(info.value),
                "buckets_used": list(info.buckets_used),
                "weights_used_normalized": {
                    b: round(w, 6)
                    for b, w in info.weights_used_normalized.items()
                },
                "bucket_values": {
                    b: (_round(v) if isinstance(v, float) else v)
                    for b, v in info.bucket_values.items()
                },
                "weights_kind": info.weights_kind,
            }
        per_model[model] = per_metric_out
    payload: dict[str, Any] = {
        "weights_default": dict(report.weights_default),
        "overrides": {
            m: dict(sub) for m, sub in report.overrides.items()
        },
        "per_model": per_model,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _write_error_breakdown_csv(
    path: Path,
    per_model: dict[str, tuple[int, Counter]],
) -> Path:
    header = ["model", "error_kind", "count", "share_of_total_samples"]
    rows: list[list[Any]] = []
    for model, (total, counter) in per_model.items():
        for kind, count in sorted(counter.items()):
            share = count / total if total else 0.0
            rows.append([model, kind, count, round(share, 4)])
    return _write_csv(path, header, rows)


def _write_finish_reason_breakdown_csv(
    path: Path,
    per_model: dict[str, Counter],
) -> Path:
    """Write `finish_reason` distribution per model. The denominator is the
    eligible sample count for that model (= sum of the counter values, since
    `_finish_reason_breakdown` already filters cutoff rows out)."""
    header = ["model", "finish_reason", "count", "share_of_eligible"]
    rows: list[list[Any]] = []
    for model, counter in per_model.items():
        eligible_total = sum(counter.values())
        for reason, count in sorted(counter.items()):
            share = count / eligible_total if eligible_total else 0.0
            rows.append([model, reason, count, round(share, 4)])
    return _write_csv(path, header, rows)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_per_model_summary_md(
    path: Path,
    per_model: dict[str, tuple[int, Aggregate]],
    prob: dict[str, ModelProbabilisticAggregate] | None = None,
    fss_per_model: dict[str, dict[str, Any]] | None = None,
    cohen_kappa_per_model: dict[str, float | None] | None = None,
    consistency_per_model: dict[str, ConsistencyReport] | None = None,
    confidence_conflict_models: set[str] | None = None,
) -> Path:
    """Markdown summary: accuracy + discrete-native + companion probabilistic.

    Layout:
    * FSS sits next to pass@1 — the published main comparison metric;
    * Cohen κ / Fleiss κ / mean entropy / VCI / MVG follow as discrete
      diagnostics;
    * BI / NLL / MBS / ABI columns are kept as **companion** metrics with
      a footnote disclaimer about K=5 limiting probability resolution to
      6 discrete levels.
    * `conflict*` marker is preserved (linguistic vs numeric confidence
      divergence).
    * The `cal*` marker and BI_cal/NLL_cal/ECE_* columns are gone:
      calibration is deprecated under K=5 resolution.
    """
    lines = ["# Per-model summary (discrete-native primary, probabilistic as companion)", ""]
    header = [
        "model", "N",
        "eligible_Q", "eligible_S", "cutoff_S",
        "pass@1", "FSS", "Cohen_κ",
        "Fleiss_κ", "H̄", "VCI", "MVG",
        "pass_any@N", "≥majority", "≥all",
        "exam_score_at_n_avg",  # exam-score-metric: hook 2/2 (markdown table column)
        "majority_acc", "parse_fail", "error_rate",
        # Bail-out retry frequency. NULL on legacy DBs renders as "—".
        "retry_rate",
        "BI†", "NLL†", "MBS†", "ABI_crowd†", "ABI_unif†", "fallback%",
        "avg_tool", "avg_steps", "avg_nudges", "avg_latency_ms",
        "avg_p/c/r_tokens",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for model, (sampling_n, agg) in per_model.items():
        row_dict = agg.as_ordered_dict()
        prob_dict = _prob_dict(prob.get(model) if prob else None)
        v5_extras = _v5_extras_dict(
            fss_per_model.get(model) if fss_per_model else None,
            cohen_kappa_per_model.get(model) if cohen_kappa_per_model else None,
            None,  # hamming not in markdown (CSV-only)
            consistency_per_model.get(model) if consistency_per_model else None,
        )
        # Surface fallback share as a percentage so reviewers can spot
        # "this model never produced a parsed belief, take the metrics with
        # a grain of salt" without doing arithmetic in their head.
        fallback_pct = (
            f"{prob_dict['fallback_share'] * 100:.1f}%"
            if prob_dict.get("fallback_share") is not None
            else "—"
        )
        # `conflict*` marker survives.
        markers: list[str] = []
        if confidence_conflict_models and model in confidence_conflict_models:
            markers.append("conflict*")
        model_cell = f"{model} {' '.join(markers)}".rstrip() if markers else model
        tok_cell = f"{_fmt(row_dict['avg_prompt_tokens'])} / {_fmt(row_dict['avg_completion_tokens'])} / {_fmt(row_dict['avg_reasoning_tokens'])}"
        cells = [
            model_cell,
            str(sampling_n),
            _fmt(row_dict["eligible_questions"]),
            _fmt(row_dict["eligible_samples"]),
            _fmt(row_dict["cutoff_skip_samples"]),
            _fmt(row_dict["pass_at_1_avg"]),
            _fmt(v5_extras["fss"]),
            _fmt(v5_extras["cohen_kappa"]),
            _fmt(v5_extras["fleiss_kappa"]),
            _fmt(v5_extras["mean_entropy"]),
            _fmt(v5_extras["vci"]),
            _fmt(v5_extras["mvg"]),
            _fmt(row_dict["pass_any_at_n"]),
            _fmt(row_dict["at_least_majority_at_n"]),
            _fmt(row_dict["at_least_all_at_n"]),
            _fmt(row_dict["exam_score_at_n_avg"]),
            _fmt(row_dict["majority_vote_accuracy"]),
            _fmt(row_dict["parse_failure_rate"]),
            _fmt(row_dict["error_rate"]),
            _fmt(row_dict["final_answer_retry_rate"]),
            _fmt(prob_dict["bi"]),
            _fmt(prob_dict["nll"]),
            _fmt(prob_dict["mbs"]),
            _fmt(prob_dict["abi_crowd"]),
            _fmt(prob_dict["abi_uniform"]),
            fallback_pct,
            _fmt(row_dict["avg_tool_calls"]),
            _fmt(row_dict["avg_react_steps"]),
            _fmt(row_dict["avg_nudges_used"]),
            _fmt(row_dict["avg_latency_ms"]),
            tok_cell,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "† Probabilistic metrics are computed from empirical vote frequencies "
        "over K=5 parallel trials, yielding only 6 discrete probability levels "
        "per label. These values serve as ordinal companions to the primary "
        "discrete metrics and should not be interpreted as continuous "
        "calibration diagnostics."
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_overall_json(
    path: Path,
    *,
    run_id: str,
    sampling_n_by_model: dict[str, int],
    per_model: dict[str, Aggregate],
    per_model_by_difficulty: dict[str, dict[str, Aggregate]],
    error_breakdown: dict[str, tuple[int, Counter]],
    probabilistic_per_model: dict[str, ModelProbabilisticAggregate] | None = None,
    probabilistic_by_difficulty: (
        dict[str, dict[str, ModelProbabilisticAggregate]] | None
    ) = None,
    analysis_schema: str | None = None,
    composite: CompositeReport | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "run_id": run_id,
        "sampling_n_by_model": sampling_n_by_model,
        "per_model": {m: agg.as_ordered_dict() for m, agg in per_model.items()},
        "per_model_by_difficulty": {
            m: {k: agg.as_ordered_dict() for k, agg in by_k.items()}
            for m, by_k in per_model_by_difficulty.items()
        },
        "error_breakdown": {
            m: {"total_samples": total, "counts": dict(sorted(counter.items()))}
            for m, (total, counter) in error_breakdown.items()
        },
    }
    if probabilistic_per_model is not None:
        payload["probabilistic"] = {
            "per_model": {m: _prob_dict(agg) for m, agg in probabilistic_per_model.items()},
            "per_model_by_difficulty": {
                m: {k: _prob_dict(agg) for k, agg in by_k.items()}
                for m, by_k in (probabilistic_by_difficulty or {}).items()
            },
        }
    if analysis_schema is not None:
        payload["analysis_schema"] = analysis_schema
    if composite is not None:
        payload["composite"] = {
            "weights_default": dict(composite.weights_default),
            "overrides": {m: dict(sub) for m, sub in composite.overrides.items()},
            "per_model": {
                model: {
                    metric: {
                        "value": _round(info.value),
                        "buckets_used": list(info.buckets_used),
                        "weights_used_normalized": {
                            b: round(w, 6)
                            for b, w in info.weights_used_normalized.items()
                        },
                        "weights_kind": info.weights_kind,
                    }
                    for metric, info in sorted(per_metric.items())
                }
                for model, per_metric in sorted(composite.per_model.items())
            },
        }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _write_shrinkage_alpha_curve_csv(
    path: Path,
    per_model: dict[str, ShrinkageResult],
) -> Path:
    """Per-model α grid scan with mean BS and BI."""
    header = ["model", "alpha", "mean_bs", "bi", "n_questions", "choice_type"]
    rows: list[list[Any]] = []
    for model, res in per_model.items():
        for alpha, mean_bs, bi in res.curve:
            rows.append(
                [model, round(alpha, 4), round(mean_bs, 6),
                 round(bi, 4), res.n_questions, res.choice_type]
            )
    return _write_csv(path, header, rows)


def _bs_to_bi(mean_bs: float) -> float:
    """$100 \\cdot (1 - \\sqrt{\\overline{BS}})$ — the same convention `proper_score.brier_index` uses."""
    import math

    if mean_bs < 0:
        return 100.0 * (1.0 + math.sqrt(-mean_bs))
    return 100.0 * (1.0 - math.sqrt(mean_bs))


def _write_paired_delta_bi_csv(
    path: Path,
    pairs: list[ModelPairResult],
) -> Path:
    """Pairwise mean ΔBS converted to ΔBI for readability."""
    header = [
        "model_a", "model_b", "n_questions",
        "delta_bs", "ci_low_bs", "ci_high_bs",
        "delta_bi_approx",
        "p_raw", "p_holm", "posterior_a_better",
    ]
    rows: list[list[Any]] = []
    for p in pairs:
        # ΔBI = BI_b - BI_a (positive means b is worse — i.e. a is better).
        # Approximated by linearising around the per-pair mean BS; not the
        # same as recomputing BI from each side but adequate as a column
        # alongside the BS-domain CIs.
        delta_bi_approx = -p.delta_bs_mean * 100.0
        rows.append([
            p.model_a, p.model_b, p.n_questions,
            round(p.delta_bs_mean, 6),
            round(p.ci_low, 6),
            round(p.ci_high, 6),
            round(delta_bi_approx, 4),
            round(p.p_raw, 6),
            round(p.p_holm, 6) if p.p_holm is not None else None,
            round(p.posterior_a_better, 6),
        ])
    return _write_csv(path, header, rows)


def _write_pairwise_significance_csv(
    path: Path,
    pairs: list[ModelPairResult],
    *,
    alpha: float = 0.05,
) -> Path:
    """Significance flags at α=0.05 using Holm-adjusted p-values."""
    header = [
        "model_a", "model_b", "delta_bs", "p_raw", "p_holm",
        "is_significant_raw", "is_significant_holm",
    ]
    rows: list[list[Any]] = []
    for p in pairs:
        raw_sig = int(p.p_raw < alpha)
        holm_sig = int(p.p_holm is not None and p.p_holm < alpha)
        rows.append([
            p.model_a, p.model_b,
            round(p.delta_bs_mean, 6),
            round(p.p_raw, 6),
            round(p.p_holm, 6) if p.p_holm is not None else None,
            raw_sig, holm_sig,
        ])
    return _write_csv(path, header, rows)


def _write_posterior_pairwise_csv(
    path: Path,
    pairs: list[ModelPairResult],
) -> Path:
    """$\\Pr(\\mathrm{BI}_A > \\mathrm{BI}_B)$ from paired bootstrap."""
    header = ["model_a", "model_b", "n_questions", "prob_a_better"]
    rows: list[list[Any]] = []
    for p in pairs:
        rows.append([
            p.model_a, p.model_b, p.n_questions,
            round(p.posterior_a_better, 6),
        ])
    return _write_csv(path, header, rows)


def _write_per_model_by_difficulty_csv(
    path: Path,
    per_model_per_tier: dict[str, dict[str, ModelProbabilisticAggregate]],
) -> Path:
    """Per-(model, tier) probabilistic aggregates."""
    header = [
        "model", "difficulty_tertile", "n_questions",
        "bi", "nll", "abi_crowd", "abi_uniform", "fallback_share",
    ]
    rows: list[list[Any]] = []
    tier_order = {"low": 0, "mid": 1, "high": 2}
    for model in sorted(per_model_per_tier.keys()):
        tiers = per_model_per_tier[model]
        for tier in sorted(tiers.keys(), key=lambda t: tier_order.get(t, 99)):
            agg = tiers[tier]
            rows.append([
                model, tier, agg.n_questions,
                _round(agg.bi),
                _round(agg.nll),
                _round(agg.abi_crowd),
                _round(agg.abi_uniform),
                _round(agg.fallback_share),
            ])
    return _write_csv(path, header, rows)


def _write_paired_delta_bi_by_difficulty_csv(
    path: Path,
    by_pair_per_tier: dict[tuple[str, str], dict[str, PairedBootstrapResult]],
) -> Path:
    """Pairwise ΔBS within each difficulty tertile."""
    header = [
        "model_a", "model_b", "difficulty_tertile", "n_questions",
        "delta_bs", "ci_low_bs", "ci_high_bs",
        "delta_bi_approx", "p_two_sided",
    ]
    rows: list[list[Any]] = []
    tier_order = {"low": 0, "mid": 1, "high": 2}
    for (ma, mb) in sorted(by_pair_per_tier.keys()):
        tiers = by_pair_per_tier[(ma, mb)]
        for tier in sorted(tiers.keys(), key=lambda t: tier_order.get(t, 99)):
            res = tiers[tier]
            rows.append([
                ma, mb, tier, res.n_questions,
                round(res.delta_mean, 6),
                round(res.ci_low, 6),
                round(res.ci_high, 6),
                round(-res.delta_mean * 100.0, 4),
                round(res.p_two_sided, 6),
            ])
    return _write_csv(path, header, rows)


# --------------------------------------------------------------------------- #
# Discrete-native + consistency writers
# --------------------------------------------------------------------------- #


def _write_inter_trial_consistency_csv(
    path: Path,
    per_model: dict[str, ConsistencyReport],
) -> Path:
    """Per-model Fleiss κ / mean entropy / VCI / MVG (one row per model)."""
    header = [
        "model",
        "fleiss_kappa",
        "mean_entropy",
        "vci",
        "mvg",
        "n_questions_used",
    ]
    rows: list[list[Any]] = []
    for model in sorted(per_model.keys()):
        rep = per_model[model]
        rows.append([
            model,
            _round(rep.fleiss_kappa, 6),
            _round(rep.mean_entropy, 6),
            _round(rep.vci, 6),
            _round(rep.mvg, 6),
            rep.n_questions_used,
        ])
    return _write_csv(path, header, rows)


def _write_entropy_accuracy_bins_csv(
    path: Path,
    per_model: dict[str, list[dict[str, Any]]],
) -> Path:
    """Per-model × bucket row.

    Bucket order is fixed `low / mid / high` (with arbitrary "qN" labels for
    `n_buckets != 3`). Per-model bucket boundaries differ — `h_lo` / `h_hi`
    are model-specific, NOT a shared scale.
    """
    header = [
        "model", "bucket", "n_questions",
        "h_lo", "h_hi",
        "acc", "mv_acc", "fleiss_kappa",
    ]
    rows: list[list[Any]] = []
    bucket_order = {"low": 0, "mid": 1, "high": 2}
    for model in sorted(per_model.keys()):
        bins = per_model[model]
        sorted_bins = sorted(
            bins,
            key=lambda b: bucket_order.get(b.get("bucket_label", ""), 99),
        )
        for b in sorted_bins:
            rows.append([
                model,
                b.get("bucket_label"),
                b.get("n_questions"),
                _round(b.get("h_lo"), 6),
                _round(b.get("h_hi"), 6),
                _round(b.get("acc"), 6),
                _round(b.get("mv_acc"), 6),
                _round(b.get("fleiss_kappa"), 6),
            ])
    return _write_csv(path, header, rows)


def _write_metric_pairwise_bootstrap_csv(
    path: Path,
    results: list[MetricBootstrapResult],
    *,
    alpha: float = 0.05,
) -> Path:
    """Long-table for `pairwise_bootstrap.csv`.

    One row per (metric × ordered model pair). `sig_at_05` flags whether the
    95% CI excludes 0 (equivalently, p < α). Cohen's d gives the effect
    size — reviewers comparing 'p < 0.05 with d=0.05' (trivial effect, large
    sample) vs 'p < 0.05 with d=0.8' (large effect) read d, not p.
    """
    header = [
        "metric", "model_a", "model_b", "n_questions",
        "delta_mean", "ci_low", "ci_high",
        "p_value", "cohens_d", "sig_at_05",
    ]
    rows: list[list[Any]] = []
    for r in sorted(
        results, key=lambda x: (x.metric_name, x.model_a, x.model_b)
    ):
        sig = int(r.p_two_sided < alpha)
        rows.append([
            r.metric_name,
            r.model_a,
            r.model_b,
            r.n_questions,
            _round(r.delta_mean, 6),
            _round(r.ci_low, 6),
            _round(r.ci_high, 6),
            _round(r.p_two_sided, 6),
            _round(r.cohens_d, 4),
            sig,
        ])
    return _write_csv(path, header, rows)


# --------------------------------------------------------------------------- #
# Behavior / confidence writers
# --------------------------------------------------------------------------- #


def _write_belief_evolution_csv(
    path: Path,
    rows: list[BeliefEvolutionRow],
) -> Path:
    """Spec 25.7: per-(model, q, k) belief evolution indicators."""
    header = [
        "model",
        "question_id",
        "question_type",
        "choice_type",
        "sample_idx",
        "n_steps",
        "trial_internal_volatility",
        "convergence_step",
        "evidence_efficiency",
        "counterevidence_engaged",
        "inter_trial_variance",
    ]
    out_rows: list[list[Any]] = []
    for r in rows:
        out_rows.append([
            r.model,
            r.question_id,
            r.question_type,
            r.choice_type,
            r.sample_idx,
            r.n_steps,
            _round(r.volatility, 6),
            r.convergence_step,
            _round(r.evidence_efficiency, 6),
            r.counterevidence_engaged,
            _round(r.inter_trial_variance, 6),
        ])
    return _write_csv(path, header, out_rows)


def _write_reflection_ab_csv(
    path: Path,
    rows: list[ReflectionABRow],
) -> Path:
    """Spec 26.4: paired bootstrap CI per metric per qtype."""
    header = [
        "model",
        "question_type",
        "metric",
        "n_questions",
        "delta_mean",
        "ci_low",
        "ci_high",
        "p_value",
    ]
    out_rows: list[list[Any]] = []
    for r in rows:
        out_rows.append([
            r.model,
            r.question_type,
            r.metric,
            r.n_questions,
            _round(r.delta_mean, 6),
            _round(r.ci_low, 6),
            _round(r.ci_high, 6),
            _round(r.p_value, 6),
        ])
    return _write_csv(path, header, out_rows)


def _write_tool_usage_pdp_csv(
    path: Path,
    rows: list[PDPRow],
) -> Path:
    """Spec 27.3: per-model per-feature partial dependence."""
    header = [
        "model",
        "feature",
        "feature_value",
        "pdp_correct",
        "pdp_nll",
        "n_samples",
    ]
    out_rows: list[list[Any]] = []
    for r in rows:
        out_rows.append([
            r.model,
            r.feature,
            _round(r.feature_value, 6),
            _round(r.pdp_correct, 6),
            _round(r.pdp_nll, 6),
            r.n_samples,
        ])
    return _write_csv(path, header, out_rows)


def _write_confidence_calibration_csv(
    path: Path,
    rows: list[ConfidenceCalibrationRow],
) -> Path:
    """Spec 28.2: subjective (low/medium/high) confidence vs hit rate."""
    header = ["model", "confidence", "n_samples", "mean_max_p", "hit_rate"]
    out_rows: list[list[Any]] = []
    for r in rows:
        out_rows.append([
            r.model,
            r.confidence,
            r.n_samples,
            _round(r.mean_max_p, 6),
            _round(r.hit_rate, 6),
        ])
    return _write_csv(path, header, out_rows)


def _write_numeric_confidence_calibration_csv(
    path: Path,
    rows: list[NumericConfidenceCalibrationRow],
) -> Path:
    """Spec 28.2: max_p binning vs hit rate."""
    header = [
        "model",
        "bin_low",
        "bin_high",
        "n_samples",
        "mean_max_p",
        "hit_rate",
    ]
    out_rows: list[list[Any]] = []
    for r in rows:
        out_rows.append([
            r.model,
            _round(r.bin_low, 4),
            _round(r.bin_high, 4),
            r.n_samples,
            _round(r.mean_max_p, 6),
            _round(r.hit_rate, 6),
        ])
    return _write_csv(path, header, out_rows)


# --------------------------------------------------------------------------- #
# Grid-search writers
# --------------------------------------------------------------------------- #


def _write_grid_summary_csv(
    path: Path,
    grid: "dict[tuple[str, int, int], GridCell]",
) -> Path:
    """`grid_summary.csv` — main triplet table.

    17-column header is locked by the `search-budget-grid` spec; rows are
    sorted by `(real_model, R, C)` so paired diffs across runs stay stable.
    `ece` is currently None on every row — per-cell calibration (Platt /
    temperature) is skipped to keep the dependency surface small. Plot
    code reads `bi_mean` etc. and ignores `ece`; the column is reserved so a
    future calibration pass can fill it without breaking schema.
    """
    from .grid import _GRID_SUMMARY_HEADER

    header = list(_GRID_SUMMARY_HEADER)
    rows: list[list[Any]] = []
    for (real, R, C) in sorted(grid.keys()):
        cell = grid[(real, R, C)]
        acc = cell.accuracy_aggregate
        prob = cell.probabilistic_aggregate
        rows.append([
            real,
            R,
            C,
            cell.n_eligible,
            cell.n_total,
            _round(acc.pass_at_1_avg),
            _round(cell.acc_ci_lo),
            _round(cell.acc_ci_hi),
            _round(prob.bi),
            _round(cell.bi_ci_lo),
            _round(cell.bi_ci_hi),
            _round(prob.nll),
            None,
            _round(cell.mean_search_calls, 2),
            _round(cell.mean_latency_ms, 1),
            _round(cell.parse_ok_rate),
            _round(cell.belief_parse_ok_rate),
        ])
    return _write_csv(path, header, rows)


def _write_grid_marginal_C_csv(
    path: Path,
    cells: "list[GridCell]",
    fix_R: int,
) -> Path:
    """Per-(real_model, R fixed, C) row. Rows are pre-sorted by the caller."""
    header = [
        "real_model", "R_fixed", "C",
        "bi_mean", "bi_ci_lo", "bi_ci_hi",
        "mean_search_calls", "mean_latency_ms",
        "n_eligible",
    ]
    rows: list[list[Any]] = []
    for c in cells:
        rows.append([
            c.real_model,
            int(fix_R),
            c.C,
            _round(c.probabilistic_aggregate.bi),
            _round(c.bi_ci_lo),
            _round(c.bi_ci_hi),
            _round(c.mean_search_calls, 2),
            _round(c.mean_latency_ms, 1),
            c.n_eligible,
        ])
    return _write_csv(path, header, rows)


def _write_grid_marginal_R_csv(
    path: Path,
    cells: "list[GridCell]",
    fix_C: int,
) -> Path:
    """Symmetric to `_write_grid_marginal_C_csv` — fixes C and varies R."""
    header = [
        "real_model", "C_fixed", "R",
        "bi_mean", "bi_ci_lo", "bi_ci_hi",
        "mean_search_calls", "mean_latency_ms",
        "n_eligible",
    ]
    rows: list[list[Any]] = []
    for c in cells:
        rows.append([
            c.real_model,
            int(fix_C),
            c.R,
            _round(c.probabilistic_aggregate.bi),
            _round(c.bi_ci_lo),
            _round(c.bi_ci_hi),
            _round(c.mean_search_calls, 2),
            _round(c.mean_latency_ms, 1),
            c.n_eligible,
        ])
    return _write_csv(path, header, rows)


def _write_grid_pareto_csv(
    path: Path,
    pareto: "list[GridCell]",
    grid: "dict[tuple[str, int, int], GridCell]",
) -> Path:
    """Pareto front + `dominated_by` annotation for non-frontier cells.

    Layout: every cell appears exactly once; `dominated_by` is empty
    string when the cell is on the frontier (so `pareto` membership is
    self-evident from the column being blank). Otherwise it points at
    one specific dominator (the lex-smallest dominator) so a reviewer
    can eyeball "why was this cell dropped?"
    """
    header = [
        "real_model", "R", "C",
        "mean_search_calls", "bi_mean",
        "dominated_by",
    ]
    pareto_keys = {(c.real_model, c.R, c.C) for c in pareto}
    rows: list[list[Any]] = []
    for key in sorted(grid.keys()):
        cell = grid[key]
        is_pareto = key in pareto_keys
        bi = cell.probabilistic_aggregate.bi
        x = cell.mean_search_calls
        dominator = ""
        if not is_pareto and bi is not None and x is not None:
            for okey in sorted(grid.keys()):
                if okey == key:
                    continue
                o = grid[okey]
                ox = o.mean_search_calls
                obi = o.probabilistic_aggregate.bi
                if ox is None or obi is None:
                    continue
                x_weak = ox <= x
                y_weak = obi >= bi
                x_strict = ox < x
                y_strict = obi > bi
                if x_weak and y_weak and (x_strict or y_strict):
                    dominator = (
                        f"{o.real_model}::r{o.R}::c{o.C}"
                    )
                    break
        rows.append([
            cell.real_model, cell.R, cell.C,
            _round(x, 2),
            _round(bi),
            dominator,
        ])
    return _write_csv(path, header, rows)


def _write_grid_winrate_csv(
    path: Path,
    rows_in: "list[WinrateRow]",
) -> Path:
    """Pairwise win-count matrix in long form (one row per ordered pair).

    Columns mirror the `WinrateRow` dataclass field order — keeps the
    plot reader simple."""
    header = [
        "model_a", "model_b",
        "total_cells", "wins_a", "wins_b", "ties",
        "sig_cells_a", "sig_cells_b",
    ]
    out_rows: list[list[Any]] = []
    for r in rows_in:
        out_rows.append([
            r.model_a, r.model_b,
            r.total_cells, r.wins_a, r.wins_b, r.ties,
            r.sig_cells_a, r.sig_cells_b,
        ])
    return _write_csv(path, header, out_rows)


def _write_per_model_ci_csv(path, rows):
    """Long-form per-(model, metric) CI table for the panel statistical
    envelope. Columns: model, metric, point, ci_lo, ci_hi, ci_half_width,
    n_questions. Rows are `PerModelCI` dataclass instances.
    """
    header = ["model", "metric", "point", "ci_lo", "ci_hi", "ci_half_width", "n_questions"]
    out_rows = []
    for r in rows:
        out_rows.append([
            r.model, r.metric,
            _round(r.point), _round(r.ci_lo), _round(r.ci_hi),
            _round(r.ci_half_width), r.n_questions,
        ])
    return _write_csv(path, header, out_rows)


def _write_multi_comparison_sensitivity_csv(path, sensitivity_by_metric):
    """Holm vs BH sensitivity. Columns: metric, n_pairs, both, holm_only,
    bh_only, neither. `sensitivity_by_metric` is `{metric: {category: int}}`.
    """
    header = ["metric", "n_pairs", "both", "holm_only", "bh_only", "neither"]
    out_rows = []
    for metric in sorted(sensitivity_by_metric.keys()):
        cats = sensitivity_by_metric[metric]
        total = cats["both"] + cats["holm_only"] + cats["bh_only"] + cats["neither"]
        out_rows.append([
            metric, total,
            cats["both"], cats["holm_only"], cats["bh_only"], cats["neither"],
        ])
    return _write_csv(path, header, out_rows)


def _write_within_family_pairs_csv(path, rows, family_map):
    """Long-form within-family pairwise significance rows. Columns:
    family, metric, model_a, model_b, delta_mean, ci_low, ci_high,
    p_holm, cohens_d, sig_holm_05. Rows are MetricPairwiseSignificance
    filtered to in-family pairs.
    """
    header = [
        "family", "metric", "model_a", "model_b",
        "delta_mean", "ci_low", "ci_high",
        "p_holm", "cohens_d", "sig_holm_05",
    ]
    out_rows = []
    for r in rows:
        fam = family_map.get(r.model_a) or family_map.get(r.model_b) or "?"
        out_rows.append([
            fam, r.metric, r.model_a, r.model_b,
            _round(r.delta_mean), _round(r.ci_low), _round(r.ci_high),
            _round(r.p_holm, 4), _round(r.cohens_d), int(r.sig_holm_05),
        ])
    return _write_csv(path, header, out_rows)


def _write_rank_stability_csv(path, rows):
    """Per-(metric, model) rank stability. Columns: metric, model,
    point_rank, median_rank, rank_p05, rank_p95, leader_posterior.
    """
    header = ["metric", "model", "point_rank", "median_rank", "rank_p05", "rank_p95", "leader_posterior"]
    out_rows = []
    for r in rows:
        out_rows.append([
            r.metric, r.model, r.point_rank, r.median_rank,
            r.rank_p05, r.rank_p95, _round(r.leader_posterior, 4),
        ])
    return _write_csv(path, header, out_rows)


def _write_rank_stability_summary_json(path, summary_by_metric):
    """Write `{metric: {kendall_tau_*: float}}`."""
    import json as _json
    with open(path, "w") as f:
        _json.dump(
            {k: dict(v) for k, v in summary_by_metric.items()},
            f, indent=2, sort_keys=True,
        )
    return path


def _write_metric_pairwise_significance_csv(path, rows):
    """Headline pairwise paired bootstrap with Holm-adjusted p and posterior
    P(A > B). Columns: metric, model_a, model_b, n_questions, delta_mean,
    ci_low, ci_high, p_raw, p_holm, cohens_d, posterior_a_better,
    sig_holm_05. Rows are `MetricPairwiseSignificance` dataclasses.

    Separate from the BI-domain `pairwise_significance.csv` writer above
    — the two carry different schemas and downstream consumers.
    """
    header = [
        "metric", "model_a", "model_b", "n_questions",
        "delta_mean", "ci_low", "ci_high",
        "p_raw", "p_holm", "cohens_d", "posterior_a_better", "sig_holm_05",
    ]
    out_rows = []
    for r in rows:
        out_rows.append([
            r.metric, r.model_a, r.model_b, r.n_questions,
            _round(r.delta_mean), _round(r.ci_low), _round(r.ci_high),
            _round(r.p_raw, 4), _round(r.p_holm, 4),
            _round(r.cohens_d), _round(r.posterior_a_better, 4),
            int(r.sig_holm_05),
        ])
    return _write_csv(path, header, out_rows)


__all__ = [
    "_SUMMARY_FIELDS",
    "_DISCRETE_FSS_FIELDS",
    "_CONSISTENCY_FIELDS",
    "_write_csv",
    "_write_per_model_ci_csv",
    "_write_metric_pairwise_significance_csv",
    "_write_rank_stability_csv",
    "_write_rank_stability_summary_json",
    "_write_within_family_pairs_csv",
    "_write_multi_comparison_sensitivity_csv",
    "_write_per_model_summary_csv",
    "_write_slice_csv",
    "_write_per_model_composite_csv",
    "_write_composite_meta_json",
    "_write_error_breakdown_csv",
    "_write_finish_reason_breakdown_csv",
    "_write_per_model_summary_md",
    "_write_overall_json",
    "_write_shrinkage_alpha_curve_csv",
    "_write_paired_delta_bi_csv",
    "_write_pairwise_significance_csv",
    "_write_posterior_pairwise_csv",
    "_write_per_model_by_difficulty_csv",
    "_write_paired_delta_bi_by_difficulty_csv",
    "_write_belief_evolution_csv",
    "_write_reflection_ab_csv",
    "_write_tool_usage_pdp_csv",
    "_write_confidence_calibration_csv",
    "_write_numeric_confidence_calibration_csv",
    "_write_grid_summary_csv",
    "_write_grid_marginal_C_csv",
    "_write_grid_marginal_R_csv",
    "_write_grid_pareto_csv",
    "_write_grid_winrate_csv",
    "_fmt",
    # Discrete-native + consistency writers
    "_write_inter_trial_consistency_csv",
    "_write_entropy_accuracy_bins_csv",
    "_write_metric_pairwise_bootstrap_csv",
]
