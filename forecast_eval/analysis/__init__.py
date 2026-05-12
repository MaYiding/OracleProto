"""Post-run statistics for one evaluation run.

Reads `RUNS_ROOT/{run_id}/db/*.db` (one SQLite per model), computes the metric
suite from `ANALYSIS_DESIGN_v5.md`, and writes the results as CSV/Markdown/JSON
under `RUNS_ROOT/{run_id}/analysis/`.

Pure read side: this module never mutates the per-model DBs.

Entry points:
    * `run_analysis(run_dir: Path) -> list[Path]` — programmatic entry used by
      `evaluation.py` at the end of each run.
    * `python -m forecast_eval.analysis RUNS_ROOT/{run_id}` — CLI to re-run
      analysis against existing DBs.

Module layout:

* `flatten.py`     — `_flatten_db` pivot + `SampleRow` (incl. `probabilities`).
* `accuracy.py`    — pass@1 + FSS / Cohen κ / Hamming.
* `consistency.py` — K-trial Fleiss κ / entropy / entropy-Acc bins / VCI / MVG.
* `proper_score.py` — BS / NLL / MBS / BI / ABI (companion probabilistic).
* `aggregation.py` — K-trial aggregators + LOO shrinkage.
* `inference.py`   — BS-paired bootstrap + multi-metric paired bootstrap.
* `behavior.py`    — belief evolution + reflection A/B + tool PDP + confidence.
* `grid.py`        — grid (R, C) analysis.
* `writers.py`     — CSV / Markdown / JSON serialisation.

External callers should only import `run_analysis` from this module.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .. import db as dbmod
from .accuracy import (
    Aggregate,
    _aggregate,
    _error_breakdown,
    _finish_reason_breakdown,
    _slice_by,
    cohen_kappa,
    fss,
    hamming_score,
)
from .aggregation import ShrinkageResult, loo_shrinkage
from .composite import (
    DEFAULT_WEIGHTS,
    CompositeReport,
    V5SliceResult,
    collect_bucket_values,
    compute_composite,
    difficulty_bucket_key,
    slice_v5_metrics_by_bucket,
)
from .behavior import (
    build_belief_evolution_rows,
    confidence_calibration,
    confidence_conflict_models,
    numeric_confidence_calibration,
    reflection_ab_report,
    tool_usage_pdp,
)
from .consistency import (
    ConsistencyReport,
    build_consistency_report,
    entropy_accuracy_bins,
)
from .flatten import (
    CUTOFF,
    SampleRow,
    _ANALYSIS_FIELDS,
    _answer_gt_for,
    _flatten_db,
    _question_options_for,
    gt_vector,
)
from .inference import (
    DEFAULT_METRIC_FNS,
    DifficultyTertile,
    PairedBootstrapResult,
    difficulty_tertile,
    paired_bootstrap_by_difficulty,
    pairwise_metric_bootstrap,
    pairwise_paired_bootstrap,
)
from .probabilistic import (
    _QuestionProbabilityRow,
    _aggregate_for_subset,
    build_probabilistic_report,
)
from .proper_score import (
    ModelProbabilisticAggregate,
    brier_score_lab,
)
from .writers import (
    _SUMMARY_FIELDS,
    _write_belief_evolution_csv,
    _write_composite_meta_json,
    _write_confidence_calibration_csv,
    _write_entropy_accuracy_bins_csv,
    _write_error_breakdown_csv,
    _write_finish_reason_breakdown_csv,
    _write_inter_trial_consistency_csv,
    _write_metric_pairwise_bootstrap_csv,
    _write_numeric_confidence_calibration_csv,
    _write_overall_json,
    _write_paired_delta_bi_by_difficulty_csv,
    _write_paired_delta_bi_csv,
    _write_pairwise_significance_csv,
    _write_per_model_by_difficulty_csv,
    _write_per_model_composite_csv,
    _write_per_model_summary_csv,
    _write_per_model_summary_md,
    _write_posterior_pairwise_csv,
    _write_reflection_ab_csv,
    _write_shrinkage_alpha_curve_csv,
    _write_slice_csv,
    _write_tool_usage_pdp_csv,
)


# --------------------------------------------------------------------------- #
# Helpers (live here so __init__.py owns orchestration)
# --------------------------------------------------------------------------- #


def _shrinkage_per_model_per_ctype(
    samples_by_model: dict[str, list[SampleRow]],
    gt_map: dict[str, frozenset[str]],
) -> dict[str, ShrinkageResult]:
    """Run `loo_shrinkage` per (model, choice_type).

    The shrinkage formula differs for single (softmax) vs multi (sigmoid),
    so we split each model's questions by choice_type and run shrinkage
    independently. Returns `{f"{model}__{ctype}": result}` so the writer
    can emit one row per (model, ctype) without nested keys.
    """
    out: dict[str, ShrinkageResult] = {}
    for model, samples in samples_by_model.items():
        # Group by question, keep all eligible probabilities.
        per_q: dict[str, list[SampleRow]] = {}
        for s in samples:
            if not s.is_eligible or s.probabilities is None:
                continue
            if s.question_id not in gt_map:
                continue
            per_q.setdefault(s.question_id, []).append(s)

        # Bucket by choice_type (canonical: take first sample's ctype).
        by_ctype: dict[str, list[tuple[list[list[float]], list[int]]]] = {}
        for qid, ss in per_q.items():
            gt = gt_map.get(qid)
            if gt is None or not ss[0].options:
                continue
            k = len(ss[0].options)
            obs = gt_vector(gt, k)
            preds = [s.probabilities for s in ss if s.probabilities is not None]
            if not preds:
                continue
            ctype = ss[0].choice_type
            by_ctype.setdefault(ctype, []).append((preds, obs))

        for ctype, items in by_ctype.items():
            preds_list = [it[0] for it in items]
            obs_list = [it[1] for it in items]
            try:
                res = loo_shrinkage(preds_list, obs_list, ctype)
            except ValueError:
                continue
            out[f"{model}__{ctype}"] = res
    return out


def _bs_by_model_qid_from_rows(
    rows_by_model: dict[str, list[_QuestionProbabilityRow]],
) -> dict[str, dict[str, float]]:
    """Per-(model, question) label-wise BS — input for paired bootstrap and posterior."""
    out: dict[str, dict[str, float]] = {}
    for model, rows in rows_by_model.items():
        per_q: dict[str, float] = {}
        for r in rows:
            per_q[r.question_id] = brier_score_lab(r.probs, r.obs)
        out[model] = per_q
    return out


def _gamma_uniform_per_qid(
    rows_by_model: dict[str, list[_QuestionProbabilityRow]],
) -> dict[str, float]:
    """Union of uniform $\\gamma$ per question across models.

    Uniform $\\gamma$ depends only on the observation vector — same across
    models on the same question. We take any model's value as authoritative.
    """
    from .proper_score import uniform_gamma_for

    out: dict[str, float] = {}
    for rows in rows_by_model.values():
        for r in rows:
            if r.question_id not in out:
                out[r.question_id] = uniform_gamma_for(r.obs)
    return out


def _per_model_per_tier_aggregates(
    rows_by_model: dict[str, list[_QuestionProbabilityRow]],
    tertile: DifficultyTertile,
    crowd_gammas_by_model: dict[str, dict[str, float | None]],
    uniform_gammas: dict[str, float],
) -> dict[str, dict[str, ModelProbabilisticAggregate]]:
    """Slice each model's rows by difficulty tier and re-aggregate proper scores."""
    out: dict[str, dict[str, ModelProbabilisticAggregate]] = {}
    for model, rows in rows_by_model.items():
        per_tier_rows: dict[str, list[_QuestionProbabilityRow]] = {
            "low": [], "mid": [], "high": [],
        }
        for r in rows:
            tier = tertile.by_question.get(r.question_id)
            if tier is None:
                continue
            per_tier_rows[tier].append(r)
        out[model] = {
            tier: _aggregate_for_subset(
                tier_rows,
                crowd_gammas=crowd_gammas_by_model.get(model),
                uniform_gammas=uniform_gammas,
            )
            for tier, tier_rows in per_tier_rows.items()
        }
    return out


def _paired_bootstrap_pairs_by_difficulty(
    bs_by_model_qid: dict[str, dict[str, float]],
    tertile: DifficultyTertile,
) -> dict[tuple[str, str], dict[str, PairedBootstrapResult]]:
    """For every ordered pair of models, run `paired_bootstrap_by_difficulty`."""
    out: dict[tuple[str, str], dict[str, PairedBootstrapResult]] = {}
    models = sorted(bs_by_model_qid.keys())
    for i, ma in enumerate(models):
        for mb in models[i + 1:]:
            common = sorted(
                set(bs_by_model_qid[ma]) & set(bs_by_model_qid[mb])
            )
            bs_a = {q: bs_by_model_qid[ma][q] for q in common}
            bs_b = {q: bs_by_model_qid[mb][q] for q in common}
            tertile_subset = DifficultyTertile(
                by_question={q: t for q, t in tertile.by_question.items() if q in common},
                threshold_low=tertile.threshold_low,
                threshold_high=tertile.threshold_high,
                n_low=sum(1 for q in common if tertile.by_question.get(q) == "low"),
                n_mid=sum(1 for q in common if tertile.by_question.get(q) == "mid"),
                n_high=sum(1 for q in common if tertile.by_question.get(q) == "high"),
            )
            out[(ma, mb)] = paired_bootstrap_by_difficulty(bs_a, bs_b, tertile_subset)
    return out


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #


def run_analysis(
    run_dir: Path,
    *,
    composite_weights: dict[str, float] | None = None,
    composite_overrides: dict[str, dict[str, float]] | None = None,
) -> list[Path]:
    """Generate every analysis artefact for the run and return the file paths.

    When ``composite_weights`` / ``composite_overrides`` are ``None``, fall
    back to ``composite.DEFAULT_WEIGHTS`` + empty overrides (synonymous with
    ``Settings`` defaults), so unit tests / CLI re-runs can still produce
    composite output in environments without a ``.env``. ``evaluation.py``
    passes the corresponding fields from ``Settings`` through during online
    evaluation.
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found under {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = manifest.get("run_id", run_dir.name)
    models: list[str] = manifest["models"]
    model_files: dict[str, str] = manifest["model_files"]
    sampling_n_top: int = manifest.get("sampling_n", 1)
    # `analysis_schema` is missing on legacy manifests; treat that as
    # "v3 fallback semantics".
    analysis_schema: str | None = manifest.get("analysis_schema")

    # Resolve weight parameters. None -> default difficulty coefficients.
    weights = (
        dict(composite_weights)
        if composite_weights is not None
        else dict(DEFAULT_WEIGHTS)
    )
    overrides = (
        {m: dict(sub) for m, sub in composite_overrides.items()}
        if composite_overrides
        else {}
    )

    db_dir = run_dir / "db"
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    per_model_agg: dict[str, Aggregate] = {}
    per_model_agg_difficulty: dict[str, dict[str, Aggregate]] = {}
    per_model_error: dict[str, tuple[int, Counter]] = {}
    per_model_finish_reason: dict[str, Counter] = {}
    sampling_n_by_model: dict[str, int] = {}
    samples_by_model: dict[str, list[SampleRow]] = {}
    gt_map_global: dict[str, frozenset[str]] = {}
    options_map_global: dict[str, list[str]] = {}

    summary_payload: dict[str, tuple[int, Aggregate]] = {}
    slice_difficulty_payload: dict[str, tuple[int, dict[str, Aggregate]]] = {}

    for model in models:
        db_path = db_dir / model_files[model]
        if not db_path.exists():
            # Skip a missing DB rather than crash — user may have partial data.
            continue
        conn = dbmod.connect(db_path)
        try:
            meta_row = conn.execute(
                "SELECT sampling_n FROM run_meta ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            sampling_n = int(meta_row["sampling_n"]) if meta_row else sampling_n_top
            sampling_n_by_model[model] = sampling_n
            samples = _flatten_db(conn, sampling_n, model)
            gt_map = _answer_gt_for(conn)
            options_map = _question_options_for(conn)
            # Each per-model DB carries its own `questions` copy. Union them
            # so the probabilistic crowd baseline can use a question even if
            # one of the models skipped it (e.g. cutoff).
            for qid, gt in gt_map.items():
                gt_map_global.setdefault(qid, gt)
            for qid, opts in options_map.items():
                options_map_global.setdefault(qid, opts)
            agg = _aggregate(samples, sampling_n, gt_map=gt_map)
            agg_difficulty = _slice_by(
                samples, difficulty_bucket_key, sampling_n, gt_map,
            )
            per_model_agg[model] = agg
            per_model_agg_difficulty[model] = agg_difficulty
            per_model_error[model] = (len(samples), _error_breakdown(samples))
            per_model_finish_reason[model] = _finish_reason_breakdown(samples)
            samples_by_model[model] = samples
            summary_payload[model] = (sampling_n, agg)
            slice_difficulty_payload[model] = (sampling_n, agg_difficulty)
        finally:
            conn.close()

    prob_report = build_probabilistic_report(samples_by_model, gt_map_global)

    # ------------------------------------------------------------------ #
    # Discrete-native family per model: FSS / Cohen κ / Hamming /
    # ConsistencyReport / per-model entropy-Acc bins. All return None /
    # empty-list on K=1 fixtures or missing data — graceful degradation.
    # ------------------------------------------------------------------ #
    fss_per_model: dict[str, dict] = {}
    cohen_kappa_per_model: dict[str, float | None] = {}
    hamming_per_model: dict[str, float | None] = {}
    consistency_per_model: dict[str, ConsistencyReport] = {}
    entropy_acc_per_model: dict[str, list[dict]] = {}
    samples_by_model_by_q: dict[str, dict[str, list[SampleRow]]] = {}
    for model, samples in samples_by_model.items():
        fss_per_model[model] = fss(samples, gt_map_global)
        # cohen_kappa needs samples grouped by question.
        by_q: dict[str, list[SampleRow]] = {}
        for s in samples:
            by_q.setdefault(s.question_id, []).append(s)
        samples_by_model_by_q[model] = by_q
        cohen_kappa_per_model[model] = cohen_kappa(by_q, gt_map_global)
        hamming_per_model[model] = hamming_score(samples, gt_map_global)
        consistency_per_model[model] = build_consistency_report(
            samples, gt_map_global, options_map_global,
        )
        entropy_acc_per_model[model] = entropy_accuracy_bins(
            by_q, gt_map_global, options_map_global,
        )

    # ------------------------------------------------------------------ #
    # Re-run the discrete family (FSS / Cohen κ / Hamming / Consistency)
    # on each composite difficulty bucket. These values are then:
    #   * fed into ``per_model_by_bucket.csv`` (which carries the
    #     recomputed columns rather than NULL placeholders);
    #   * fed into :func:`compute_composite` for weighted aggregation.
    # ------------------------------------------------------------------ #
    v5_slice_by_difficulty: dict[str, dict[str, V5SliceResult]] = {}
    for model, samples in samples_by_model.items():
        v5_slice_by_difficulty[model] = slice_v5_metrics_by_bucket(
            samples, gt_map_global, options_map_global,
            key_fn=difficulty_bucket_key,
        )

    # ------------------------------------------------------------------ #
    # Aggregate bucket values -> CompositeReport.
    # ``per_model_agg_difficulty`` is already in
    # ``{model: {bucket: Aggregate}}`` form; combined with v5_slice and
    # prob_report.per_model_by_difficulty it forms the input to
    # collect_bucket_values.
    # ------------------------------------------------------------------ #
    bucket_values = collect_bucket_values(
        aggregate_slice=per_model_agg_difficulty,
        v5_slice=v5_slice_by_difficulty,
        prob_slice=prob_report.per_model_by_difficulty,
    )
    composite_report: CompositeReport | None = None
    if bucket_values:
        composite_report = compute_composite(
            bucket_values_per_model=bucket_values,
            weights_default=weights,
            overrides=overrides,
        )

    written: list[Path] = []
    # per_model_summary.csv carries accuracy + FSS + Consistency +
    # probabilistic columns; markdown synthesises them with the K=5
    # disclaimer footnote.
    if summary_payload:
        written.append(_write_per_model_summary_csv(
            analysis_dir / "per_model_summary.csv",
            summary_payload,
            prob=prob_report.per_model,
            fss_per_model=fss_per_model,
            cohen_kappa_per_model=cohen_kappa_per_model,
            hamming_per_model=hamming_per_model,
            consistency_per_model=consistency_per_model,
        ))
    if slice_difficulty_payload:
        written.append(_write_slice_csv(
            analysis_dir / "per_model_by_bucket.csv",
            "bucket",
            slice_difficulty_payload,
            prob=prob_report.per_model_by_difficulty,
            v5_slice=v5_slice_by_difficulty,
        ))
    # Difficulty-weighted composite summary table + one audit JSON.
    if composite_report is not None:
        written.append(_write_per_model_composite_csv(
            analysis_dir / "per_model_composite.csv",
            composite_report,
            sampling_n_by_model,
        ))
        written.append(_write_composite_meta_json(
            analysis_dir / "composite_meta.json",
            composite_report,
        ))
    if per_model_error:
        written.append(_write_error_breakdown_csv(
            analysis_dir / "error_breakdown.csv", per_model_error,
        ))
    if per_model_finish_reason:
        written.append(_write_finish_reason_breakdown_csv(
            analysis_dir / "finish_reason_breakdown.csv", per_model_finish_reason,
        ))
    written.append(_write_overall_json(
        analysis_dir / "overall.json",
        run_id=run_id,
        sampling_n_by_model=sampling_n_by_model,
        per_model=per_model_agg,
        per_model_by_difficulty=per_model_agg_difficulty,
        error_breakdown=per_model_error,
        probabilistic_per_model=prob_report.per_model,
        probabilistic_by_difficulty=prob_report.per_model_by_difficulty,
        analysis_schema=analysis_schema,
        composite=composite_report,
    ))

    # ------------------------------------------------------------------ #
    # K-trial consistency outputs — always produced; rows with K=1
    # carry NULL aggregates but the row is present so the writer schema
    # stays uniform.
    # ------------------------------------------------------------------ #
    if consistency_per_model:
        written.append(_write_inter_trial_consistency_csv(
            analysis_dir / "inter_trial_consistency.csv",
            consistency_per_model,
        ))
    # Skip writing entropy_accuracy_bins.csv when no model produced any
    # bucket (e.g. all-K=1 run) — empty file is more confusing than absent.
    if any(bins for bins in entropy_acc_per_model.values()):
        written.append(_write_entropy_accuracy_bins_csv(
            analysis_dir / "entropy_accuracy_bins.csv",
            entropy_acc_per_model,
        ))

    # ------------------------------------------------------------------ #
    # Multi-metric paired bootstrap on FSS / Acc / MV_Acc / Fleiss κ /
    # EBI. Skips metrics that return None on the data (e.g. EBI on
    # legacy fixtures, Fleiss κ on K=1).
    # ------------------------------------------------------------------ #
    is_panel = bool(manifest.get("is_virtual_panel"))
    if len(samples_by_model_by_q) >= 2 and not is_panel:
        # Legacy single-run pairwise: skipped on virtual panels because the
        # panel-aware block below produces a strictly richer table
        # (`metric_pairwise_significance.csv`) and the legacy call's 5-metric
        # x C(18,2) x 5000-iteration cost is duplicate work that adds 10+ min
        # on the 18-model panel without surfacing new information.
        metric_results = pairwise_metric_bootstrap(
            samples_by_model_by_q, gt_map_global, DEFAULT_METRIC_FNS,
        )
        if metric_results:
            written.append(_write_metric_pairwise_bootstrap_csv(
                analysis_dir / "pairwise_bootstrap.csv",
                metric_results,
            ))

    if len(samples_by_model_by_q) >= 2:
        # ------------------------------------------------------------------ #
        # Panel-aware statistical envelope: per-(model, metric) CI, pairwise
        # Holm + posterior, rank stability, within-family, BH sensitivity.
        # Gated on `is_virtual_panel` so non-panel runs are untouched.
        # ------------------------------------------------------------------ #
        if is_panel:
            from .inference import (
                build_default_metric_fns,
                metric_single_bootstrap,
                pairwise_metric_significance,
            )
            from .per_model_ci import PerModelCI
            from .rank_stability import rank_stability_panel
            from .within_family import filter_significance_to_within_family
            from .fdr_bh import benjamini_hochberg, compare_holm_vs_bh
            from .writers import (
                _write_per_model_ci_csv,
                _write_metric_pairwise_significance_csv,
                _write_rank_stability_csv,
                _write_rank_stability_summary_json,
                _write_within_family_pairs_csv,
                _write_multi_comparison_sensitivity_csv,
            )

            composite_weights = manifest.get("composite_weights")
            bootstrap_seed = int(manifest.get("bootstrap_seed", 20260512))
            n_bootstrap = int(manifest.get("bootstrap_iterations", 5000))
            ci_alpha = float(manifest.get("ci_alpha", 0.05))
            composite_source = "manifest" if composite_weights else "settings_default"

            full_metric_fns = build_default_metric_fns(
                composite_weights=composite_weights,
            )
            # Eight panel-wide metrics: composite + 4 headline
            # (pass_at_1 / cohen_kappa / fss / mv_acc) + 4 non-headline
            # (pass_any / pass_all / fleiss_kappa). `acc` is omitted
            # because `pass_at_1` is its alias; `acc` is reused below
            # for the 3 question-type-restricted sub-panels.
            panel_metric_names = [
                "composite",
                "pass_at_1", "cohen_kappa", "fss",
                "mv_acc", "pass_any", "pass_all", "fleiss_kappa",
            ]
            panel_metric_fns = {
                k: full_metric_fns[k] for k in panel_metric_names if k in full_metric_fns
            }

            # ---- Per-(model, metric) CI (panel-wide metrics) ----
            # Use metric_single_bootstrap (panel-level resampling) rather
            # than mean-over-per-q-values, because the headline metrics
            # are non-linear (composite is a bucket-weighted average;
            # Cohen / Fleiss kappa apply chance correction; FSS uses a
            # chance baseline). Per-q mean only matches panel value for
            # truly linear metrics like pass_at_1 / pass_any / pass_all.
            per_model_ci_rows: list[PerModelCI] = []
            for metric_name in sorted(panel_metric_fns.keys()):
                fn = panel_metric_fns[metric_name]
                for model in sorted(samples_by_model_by_q.keys()):
                    sbq = samples_by_model_by_q[model]
                    common = {q: sbq[q] for q in sbq if q in gt_map_global}
                    if not common:
                        continue
                    res = metric_single_bootstrap(
                        fn, common, gt_map_global,
                        metric_name=metric_name, model=model,
                        n_bootstrap=n_bootstrap, seed=bootstrap_seed,
                        ci_alpha=ci_alpha,
                    )
                    if res is None:
                        continue
                    per_model_ci_rows.append(PerModelCI(
                        model=res.model, metric=res.metric_name,
                        point=res.point,
                        ci_lo=res.ci_low, ci_hi=res.ci_high,
                        ci_half_width=(res.ci_high - res.ci_low) / 2.0,
                        n_questions=res.n_questions,
                    ))

            # ---- Pairwise significance (panel-wide metrics) ----
            significance_rows = list(pairwise_metric_significance(
                samples_by_model_by_q, gt_map_global, panel_metric_fns,
                n_bootstrap=n_bootstrap, seed=bootstrap_seed, ci_alpha=ci_alpha,
            ))

            # ---- Bucket slicing: acc on 3 question-type sub-panels ----
            # The 3 buckets are: Binary (yes_no + binary_named),
            # MC-Single (multiple_choice + choice_type=single), and
            # MC-Multi (multiple_choice + choice_type=multi).
            bucket_of_qid: dict[str, str | None] = {}
            for _model, _sbq in samples_by_model_by_q.items():
                for _qid, _ss in _sbq.items():
                    if not _ss or _qid in bucket_of_qid:
                        continue
                    s0 = _ss[0]
                    if s0.question_type in ("yes_no", "binary_named"):
                        bucket_of_qid[_qid] = "binary"
                    elif s0.question_type == "multiple_choice":
                        if s0.choice_type == "single":
                            bucket_of_qid[_qid] = "mc_single"
                        elif s0.choice_type == "multi":
                            bucket_of_qid[_qid] = "mc_multi"
                        else:
                            bucket_of_qid[_qid] = None
                    else:
                        bucket_of_qid[_qid] = None
            bucket_to_qids: dict[str, set[str]] = {}
            for _qid, _b in bucket_of_qid.items():
                if _b is None:
                    continue
                bucket_to_qids.setdefault(_b, set()).add(_qid)

            acc_subpanels = {
                "acc_binary": bucket_to_qids.get("binary", set()),
                "acc_mc_single": bucket_to_qids.get("mc_single", set()),
                "acc_mc_multi": bucket_to_qids.get("mc_multi", set()),
            }
            for bucket_key, qids in acc_subpanels.items():
                if not qids:
                    continue
                sub_samples = {
                    model: {q: sbq[q] for q in qids if q in sbq}
                    for model, sbq in samples_by_model_by_q.items()
                }
                sub_gt = {q: gt_map_global[q] for q in qids if q in gt_map_global}
                bucket_metric_fns = {bucket_key: full_metric_fns["acc"]}
                bucket_sig = pairwise_metric_significance(
                    sub_samples, sub_gt, bucket_metric_fns,
                    n_bootstrap=n_bootstrap, seed=bootstrap_seed, ci_alpha=ci_alpha,
                )
                significance_rows.extend(bucket_sig)

                # Per-(model, bucket_key) CI rows via panel-level bootstrap.
                # acc is linear (per-q correctness average), so mean-of-q
                # would equal panel value, but we use metric_single_bootstrap
                # uniformly to share the same code path with the non-linear
                # headline metrics above.
                for model in sorted(sub_samples.keys()):
                    sbq = sub_samples[model]
                    common_bucket = {q: sbq[q] for q in sbq if q in sub_gt}
                    if not common_bucket:
                        continue
                    res = metric_single_bootstrap(
                        full_metric_fns["acc"], common_bucket, sub_gt,
                        metric_name=bucket_key, model=model,
                        n_bootstrap=n_bootstrap, seed=bootstrap_seed,
                        ci_alpha=ci_alpha,
                    )
                    if res is None:
                        continue
                    per_model_ci_rows.append(PerModelCI(
                        model=res.model, metric=res.metric_name,
                        point=res.point,
                        ci_lo=res.ci_low, ci_hi=res.ci_high,
                        ci_half_width=(res.ci_high - res.ci_low) / 2.0,
                        n_questions=res.n_questions,
                    ))

            if per_model_ci_rows:
                written.append(_write_per_model_ci_csv(
                    analysis_dir / "per_model_ci.csv",
                    per_model_ci_rows,
                ))
            if significance_rows:
                written.append(_write_metric_pairwise_significance_csv(
                    analysis_dir / "metric_pairwise_significance.csv",
                    significance_rows,
                ))

            # ---- Rank stability across 5 metrics via panel-level bootstrap ----
            rank_metrics = ["composite", "pass_at_1", "fss", "cohen_kappa", "fleiss_kappa"]
            rank_rows_all = []
            rank_summary_all: dict = {}
            for metric_name in rank_metrics:
                if metric_name not in panel_metric_fns:
                    continue
                fn = panel_metric_fns[metric_name]
                rows, summary = rank_stability_panel(
                    samples_by_model_by_q, gt_map_global, fn,
                    metric_name=metric_name,
                    seed=bootstrap_seed, n_bootstrap=n_bootstrap,
                )
                if rows:
                    rank_rows_all.extend(rows)
                    rank_summary_all[metric_name] = summary
            if rank_rows_all:
                written.append(_write_rank_stability_csv(
                    analysis_dir / "rank_stability.csv", rank_rows_all,
                ))
                written.append(_write_rank_stability_summary_json(
                    analysis_dir / "rank_stability_summary.json",
                    rank_summary_all,
                ))

            # ---- Within-family pairs (P3) ----
            family_map = manifest.get("model_family_map", {})
            if family_map and significance_rows:
                wf_rows = filter_significance_to_within_family(significance_rows, family_map)
                if wf_rows:
                    written.append(_write_within_family_pairs_csv(
                        analysis_dir / "within_family_pairs.csv",
                        wf_rows, family_map,
                    ))

            # ---- Multi-comparison sensitivity (P5) ----
            if significance_rows:
                sens_by_metric: dict = {}
                by_metric: dict = {}
                for r in significance_rows:
                    by_metric.setdefault(r.metric, []).append(r)
                for metric, rs in by_metric.items():
                    p_holm_list = [r.p_holm for r in rs]
                    p_raw_list = [r.p_raw for r in rs]
                    p_bh_list = benjamini_hochberg(p_raw_list)
                    sens_by_metric[metric] = compare_holm_vs_bh(
                        p_holm_list, p_bh_list, alpha=ci_alpha,
                    )
                written.append(_write_multi_comparison_sensitivity_csv(
                    analysis_dir / "multi_comparison_sensitivity.csv",
                    sens_by_metric,
                ))

            # ---- analysis_meta stamp ----
            analysis_meta_path = analysis_dir / "analysis_meta.json"
            meta = {}
            if analysis_meta_path.exists():
                meta = json.loads(analysis_meta_path.read_text())
            meta.update({
                "bootstrap_seed": bootstrap_seed,
                "bootstrap_iterations": n_bootstrap,
                "ci_alpha": ci_alpha,
                "composite_weights_source": composite_source,
                "panel_metric_set": panel_metric_names,
            })
            analysis_meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))

    # ------------------------------------------------------------------ #
    # Probabilistic deliverables — kept as companion outputs; calibration
    # / reliability / Murphy decomposition removed.
    # ------------------------------------------------------------------ #
    rows_by_model = prob_report.rows_by_model
    has_any_rows = any(rows for rows in rows_by_model.values())
    if has_any_rows:
        # Aggregation: per-(model, choice_type) shrinkage curves.
        shrinkage_per_key = _shrinkage_per_model_per_ctype(
            samples_by_model, gt_map_global
        )
        if shrinkage_per_key:
            written.append(_write_shrinkage_alpha_curve_csv(
                analysis_dir / "shrinkage_alpha_curve.csv",
                shrinkage_per_key,
            ))

        # Inference: pairwise paired bootstrap + Holm + posterior (BS-based,
        # kept for grid.py dependency and the probabilistic narrative).
        from .probabilistic import _build_crowd_gammas_per_model

        crowd_gammas_by_model = _build_crowd_gammas_per_model(rows_by_model)
        uniform_gammas_global = _gamma_uniform_per_qid(rows_by_model)
        bs_by_model_qid = _bs_by_model_qid_from_rows(rows_by_model)
        if len(bs_by_model_qid) >= 2:
            pairs = pairwise_paired_bootstrap(bs_by_model_qid)
            if pairs:
                written.append(_write_paired_delta_bi_csv(
                    analysis_dir / "paired_delta_bi.csv", pairs,
                ))
                written.append(_write_pairwise_significance_csv(
                    analysis_dir / "pairwise_significance.csv", pairs,
                ))
                written.append(_write_posterior_pairwise_csv(
                    analysis_dir / "posterior_pairwise.csv", pairs,
                ))

        # Difficulty stratification: per-tier aggregates + paired bootstrap.
        if uniform_gammas_global:
            tertile = difficulty_tertile(uniform_gammas_global)
            per_model_per_tier = _per_model_per_tier_aggregates(
                rows_by_model, tertile, crowd_gammas_by_model, uniform_gammas_global,
            )
            if per_model_per_tier:
                written.append(_write_per_model_by_difficulty_csv(
                    analysis_dir / "per_model_by_difficulty.csv",
                    per_model_per_tier,
                ))
            if len(bs_by_model_qid) >= 2:
                by_pair_per_tier = _paired_bootstrap_pairs_by_difficulty(
                    bs_by_model_qid, tertile,
                )
                if by_pair_per_tier:
                    written.append(_write_paired_delta_bi_by_difficulty_csv(
                        analysis_dir / "paired_delta_bi_by_difficulty.csv",
                        by_pair_per_tier,
                    ))

    # ------------------------------------------------------------------ #
    # Behavior / reflection A/B / tool PDP / confidence deliverables.
    # All four blocks degrade gracefully on legacy fixtures where belief_trace
    # is uniformly NULL (empty CSVs / skipped writes).
    # ------------------------------------------------------------------ #
    behavior_rows = build_belief_evolution_rows(samples_by_model, gt_map_global)
    if behavior_rows:
        written.append(_write_belief_evolution_csv(
            analysis_dir / "belief_evolution.csv", behavior_rows,
        ))

    pdp_rows = tool_usage_pdp(samples_by_model, gt_map_global)
    if pdp_rows:
        written.append(_write_tool_usage_pdp_csv(
            analysis_dir / "tool_usage_pdp.csv", pdp_rows,
        ))

    confidence_rows = confidence_calibration(samples_by_model)
    numeric_confidence_rows = numeric_confidence_calibration(samples_by_model)
    # The confidence rows always have ≥1 model entry, but the values are
    # mostly None on legacy fixtures (no parsed beliefs). Suppress writing when
    # every numeric / hit_rate is None so the CSV doesn't add noise.
    if any(r.n_samples > 0 for r in confidence_rows):
        written.append(_write_confidence_calibration_csv(
            analysis_dir / "confidence_calibration.csv", confidence_rows,
        ))
    if any(r.n_samples > 0 for r in numeric_confidence_rows):
        written.append(_write_numeric_confidence_calibration_csv(
            analysis_dir / "numeric_confidence_calibration.csv",
            numeric_confidence_rows,
        ))

    # Reflection A/B requires sibling runs that match every fingerprint
    # except `reflection_protocol_hash`. We look one directory up from the
    # current run because pairs live as siblings under runs/.
    runs_root = run_dir.parent
    try:
        from .behavior import find_paired_runs

        pairs = find_paired_runs(runs_root)
        if pairs:
            ab_rows = reflection_ab_report(pairs)
            if ab_rows:
                written.append(_write_reflection_ab_csv(
                    analysis_dir / "reflection_ab.csv", ab_rows,
                ))
    except Exception:  # pragma: no cover — reflection A/B is best-effort
        pass

    conflict_models = confidence_conflict_models(confidence_rows)

    # ------------------------------------------------------------------ #
    # Grid CSVs over (R, C) cells. Skipped (returns []) when manifest has
    # no `grid` segment so legacy single-cell runs stay byte-identical.
    # Wrapped best-effort to mirror the reflection A/B pattern: a failure
    # here MUST NOT mask the main flow's outputs.
    # ------------------------------------------------------------------ #
    try:
        from .grid import run_grid_analysis

        grid_artifacts = run_grid_analysis(
            run_dir=run_dir,
            manifest=manifest,
            samples_by_model=samples_by_model,
            gt_map_global=gt_map_global,
            rows_by_model=rows_by_model,
            analysis_dir=analysis_dir,
        )
        written.extend(grid_artifacts)
    except Exception:  # pragma: no cover — grid analysis is best-effort
        import logging

        logging.getLogger(__name__).exception(
            "grid analysis failed; continuing without grid_*.csv outputs"
        )

    # `per_model_summary.md` is written last so it includes the discrete
    # family (FSS / Cohen κ / Fleiss κ / mean entropy / VCI / MVG)
    # alongside accuracy and companion probabilistic columns. The
    # `conflict*` marker (confidence A/B) survives — the `cal*` marker is
    # gone (calibration deprecated).
    if summary_payload:
        written.append(_write_per_model_summary_md(
            analysis_dir / "per_model_summary.md",
            summary_payload,
            prob=prob_report.per_model,
            fss_per_model=fss_per_model,
            cohen_kappa_per_model=cohen_kappa_per_model,
            consistency_per_model=consistency_per_model,
            confidence_conflict_models=conflict_models or None,
        ))
    return written


__all__ = [
    "run_analysis",
    # Re-exported for tests / dogfooding tools that poke at internals.
    "Aggregate",
    "SampleRow",
    "CUTOFF",
    "_ANALYSIS_FIELDS",
    "_SUMMARY_FIELDS",
]
