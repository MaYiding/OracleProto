"""Write fixed-points scores with coverage and input fingerprints."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from ..errors import AnalysisContractError, AnalysisIncompleteError
from .score_inputs import canonical_hash, load_catalog, load_model, local_path
from .scoring import CONTRACT, POINTS, credit, question_metrics, score_intervals, selection_diagnostics, summarize


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> Path:
    path = local_path(path)
    fields = fields or (list(rows[0]) if rows else [])
    temporary = local_path(path.with_suffix(".tmp"))
    with temporary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})
    temporary.replace(path)
    return path


def _write_text(path: Path, text: str) -> Path:
    path = local_path(path)
    temporary = local_path(path.with_suffix(".tmp"))
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return path


def run_analysis(
    run_dir: Path, *, allow_incomplete: bool = False,
    bootstrap_iterations: int = CONTRACT["bootstrap_iterations"],
    bootstrap_seed: int = CONTRACT["bootstrap_seed"],
    profiles: list[str] | None = None,
) -> list[Path]:
    """Missing/error slots block official scores, including in diagnostic previews."""
    run_dir = local_path(run_dir)
    if (not isinstance(bootstrap_iterations, int) or isinstance(bootstrap_iterations, bool)
            or bootstrap_iterations < 0 or 0 < bootstrap_iterations < 200):
        raise AnalysisContractError("Use zero bootstrap iterations or at least 200")
    is_catalog = run_dir.is_file()
    input_path = run_dir if is_catalog else local_path(run_dir / "manifest.json")
    manifest_text = input_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    if is_catalog:
        data = load_catalog(input_path, manifest, profiles)
        models = sorted(data)
        run_dir = input_path.parent
    else:
        models = sorted(manifest["models"] if profiles is None else profiles)
        if set(models) - set(manifest["models"]):
            raise AnalysisContractError("Requested profiles are absent from the manifest")
        data = {m: load_model(run_dir, manifest, m) for m in models}
    if not models or len(models) != len(set(models)):
        raise AnalysisContractError("A panel requires a nonempty list of distinct models")
    if len({d["sampling_n"] for d in data.values()}) != 1:
        raise AnalysisContractError("Model comparisons require the same number of repetitions")
    sampling_n = data[models[0]]["sampling_n"]
    common = sorted(set.intersection(*(set(d["questions"]) for d in data.values())))
    for qid in common:
        if len({canonical_hash(d["questions"][qid]) for d in data.values()}) != 1:
            raise AnalysisContractError(f"Models have different question metadata: {qid}")
    output = local_path(run_dir / "analysis" / "score")
    output.mkdir(parents=True, exist_ok=True)
    coverage, summaries, full_summaries, by_type, per_trial, diagnostics = [], [], [], [], [], []
    question_rows, sample_rows, common_metrics = [], [], {}
    incomplete_models = []
    for model in models:
        d = data[model]
        trials = [t for qid in sorted(d["trials"]) for t in d["trials"][qid]]
        counts = Counter(t["state"] for t in trials)
        expected = len(trials)
        completed = sum(counts[s] for s in ("valid", "invalid", "refusal"))
        if completed != expected:
            incomplete_models.append(model)
        coverage.append({
            "model": model, "sampling_n": sampling_n, "declared_questions": d["declared_questions"],
            "admissible_questions": len(d["questions"]), "cutoff_questions": d["cutoff_questions"],
            "common_questions": len(common), "expected_slots": expected, "completed_slots": completed,
            "missing_slots": counts["missing"], "error_slots": counts["error"],
            "invalid_slots": counts["invalid"], "refusal_slots": counts["refusal"],
            "valid_slots": counts["valid"], "completion_rate": completed / expected if expected else None,
            "invalid_rate": counts["invalid"] / completed if completed else None,
            "refusal_rate": counts["refusal"] / completed if completed else None,
            "error_rate": counts["error"] / expected if expected else None,
        })
        all_q = {qid: question_metrics(q, d["trials"][qid]) for qid, q in d["questions"].items()}
        qs = [all_q[qid] for qid in common]
        common_metrics[model] = qs
        summaries.append({"model": model, "sampling_n": sampling_n, **summarize(qs)})
        full_summaries.append({"model": model, "sampling_n": sampling_n,
                               **summarize([all_q[qid] for qid in sorted(all_q)])})
        for bucket in POINTS:
            by_type.append({"model": model, "bucket": bucket,
                            **summarize([q for q in qs if q["bucket"] == bucket])})
        for qid in sorted(all_q):
            question_rows.append({"model": model, "in_common_panel": qid in common, **all_q[qid]})
        common_trials = [t for t in trials if t["question_id"] in common]
        diag = selection_diagnostics(common_trials)
        multi_qs = [q for q in qs if q["bucket"] == "mc_multi"]
        diag["select_all_f1"] = statistics.fmean(q["select_all_credit"] for q in multi_qs) if multi_qs else None
        diagnostics.append({"model": model, **diag})
        for i in range(sampling_n):
            selected = [t for t in common_trials if t["sample_idx"] == i]
            complete = bool(selected) and all(t["state"] in ("valid", "invalid", "refusal") for t in selected)
            full = sum(POINTS[t["bucket"]] for t in selected)
            earned = sum(POINTS[t["bucket"]] * credit(
                frozenset(t["prediction"]) if t["prediction"] is not None else None,
                frozenset(t["gold"]), t["bucket"]) for t in selected) if complete else None
            per_trial.append({"model": model, "sample_idx": i, "questions": len(selected),
                              "full_points": full, "earned_points": earned,
                              "score": 100 * earned / full if earned is not None else None})
        trial_scores = [row["score"] for row in per_trial if row["model"] == model]
        summaries[-1]["trial_score_sd_pp"] = statistics.stdev(trial_scores) if (
            sampling_n > 1 and all(x is not None for x in trial_scores)) else None
        for field in ("tool_calls_count", "react_steps", "latency_ms", "prompt_tokens", "completion_tokens", "reasoning_tokens"):
            vals = [t[field] for t in common_trials if t[field] is not None]
            summaries[-1]["mean_" + field] = statistics.fmean(vals) if vals else None
            summaries[-1][field + "_observed_slots"] = len(vals)
        for t in trials:
            scoreable = t["state"] in ("valid", "invalid", "refusal")
            value = credit(frozenset(t["prediction"]) if t["prediction"] is not None else None,
                           frozenset(t["gold"]), t["bucket"]) if scoreable else None
            pred, gold = set(t["prediction"] or []), set(t["gold"])
            sample_rows.append({**t, "points": POINTS[t["bucket"]], "credit": value,
                                "earned_points": POINTS[t["bucket"]] * value if value is not None else None,
                                "exact": int(t["prediction"] == t["gold"]) if scoreable else None,
                                "tp": len(pred & gold) if t["state"] == "valid" else None,
                                "fp": len(pred - gold) if t["state"] == "valid" else None,
                                "fn": len(gold - pred) if t["state"] == "valid" else None})
    cis, pairs = score_intervals(common_metrics, iterations=bootstrap_iterations, seed=bootstrap_seed)
    ci_lookup = {row["model"]: row for row in cis}
    for row in summaries:
        ci = ci_lookup.get(row["model"], {})
        row.update(score_ci_low=ci.get("ci_low"), score_ci_high=ci.get("ci_high"))
    written = []
    for name, rows in (("score_summary", summaries), ("score_by_type", by_type),
                       ("score_by_trial", per_trial), ("selection_diagnostics", diagnostics),
                       ("coverage", coverage), ("full_admissible_scores", full_summaries),
                       ("question_scores", question_rows), ("sample_scores", sample_rows)):
        written.append(_write_csv(output / (name + ".csv"), rows))
    written.append(_write_csv(output / "score_intervals.csv", cis,
                             ["model", "score", "ci_low", "ci_high", "questions", "iterations"]))
    written.append(_write_csv(output / "score_pairwise.csv", pairs,
                             ["model_a", "model_b", "delta_score_pp", "ci_low", "ci_high", "questions", "interval_scope"]))
    contract = {**CONTRACT, "bootstrap_iterations": bootstrap_iterations, "bootstrap_seed": bootstrap_seed}
    implementation = {}
    for path in (Path(__file__), Path(__file__).with_name("scoring.py"), Path(__file__).with_name("score_inputs.py"),
                 Path(__file__).parents[1] / "parser.py", Path(__file__).parents[1] / "prompts.py"):
        implementation[path.name] = hashlib.sha256(local_path(path).read_bytes()).hexdigest()
    meta = {
        "contract": contract, "contract_sha256": canonical_hash(contract),
        "implementation_sha256": canonical_hash(implementation), "implementation_files": implementation,
        "source_manifest_sha256": hashlib.sha256(manifest_text.encode()).hexdigest(),
        "input_kind": "catalog" if is_catalog else "run_manifest",
        "catalog_inputs": {key: manifest[key] for key in ("source_db", "source_sha256", "observations_path", "observations_sha256")} if is_catalog else None,
        "models": models, "common_question_ids": common, "common_question_ids_sha256": canonical_hash(common),
        "sampling_n": sampling_n, "sources": {m: data[m]["sources"] for m in models},
        "type_question_counts": dict(Counter(q["bucket"] for q in common_metrics[models[0]])),
        "status": "incomplete" if incomplete_models else "complete" if common else "no_common_questions",
        "incomplete_models": incomplete_models,
        "interval_interpretation": "Conditional on observed repeats and fixed type composition; question clusters; exploratory pairwise intervals, no multiplicity-adjusted significance or Bayesian posterior.",
        "artifacts": [p.name for p in written] + ["score_report.md"],
    }
    meta["analysis_fingerprint"] = canonical_hash(meta)
    lines = ["# Forecast Score", "", f"Status: {meta['status']}. Common questions: {len(common)}. Repetitions: {sampling_n}.",
             "", "Score = 100 × earned points / available points. Question points: 1 / 1 / 2 / 3.",
             "", "| Model | Score | 95% CI | Exact@1 | Pass@3 | All@3 | Vote Score |", "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    def fmt(value: float | None) -> str:
        return "—" if value is None else f"{value:.2f}"
    for row in summaries:
        rates = [100 * row[k] if row[k] is not None else None for k in ("exact_at_1", "pass_at_3", "pass_all_at_3")]
        label = row["model"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {label} | {fmt(row['score'])} | [{fmt(row['score_ci_low'])}, {fmt(row['score_ci_high'])}] | "
                     + " | ".join(fmt(v) for v in rates) + f" | {fmt(row['vote_score'])} |")
    lines.extend(["", "Exact/pass columns above are percentages; CSV rates are in [0, 1]. Score columns are in [0, 100].",
                  "", "Pass@3 and best-of-N use gold answers to select successful samples and represent oracle ceilings. Vote uses a strict majority of whole answer sets; invalid votes abstain without reducing its denominator.",
                  "", "Read selection_diagnostics.csv with its valid-answer coverage. Precision/Recall and selection bias describe valid multi-answer submissions. Invalid answers and content-policy refusals earn zero points. Missing/infrastructure failures make the affected population unscorable.",
                  "", "Paired intervals are exploratory and unadjusted for multiple comparisons. They cannot establish a unique winner. Score is a discrete decision score, not a probability-calibration score.",
                  "", "Only files listed in scoring_meta.json belong to this scoring contract. Source databases are read-only.", ""])
    written.append(_write_text(output / "score_report.md", "\n".join(lines)))
    written.append(_write_text(output / "scoring_meta.json", json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n"))
    if (incomplete_models or not common) and not allow_incomplete:
        raise AnalysisIncompleteError(f"Panel is {meta['status']}; inspect {output / 'coverage.csv'} and sample_scores.csv")
    return written
