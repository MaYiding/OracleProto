"""Fixed question points, selection diagnostics, and repeated-answer summaries."""
from __future__ import annotations

import itertools
import math
import random
import statistics
from collections import Counter
from typing import Any

from ..errors import AnalysisContractError


POINTS = {"yes_no": 1, "binary_named": 1, "mc_single": 2, "mc_multi": 3}
CONTRACT = {
    "name": "points-f1",
    "points": POINTS,
    "single_credit": "exact_set_equality",
    "multi_credit": "2TP/(2TP+FP+FN)",
    "aggregation": "100*sum(points*mean_trial_credit)/sum(points)",
    "invalid_answer": "zero_credit_in_denominator",
    "content_policy": "zero_credit_in_denominator",
    "missing_or_infrastructure_error": "incomplete_no_official_score",
    "cutoff": "exclude_question",
    "headline_population": "intersection_of_declared_admissible_questions",
    "vote": "strict_majority_of_complete_answer_sets_invalid_votes_abstain",
    "pass_at_k": "mean(1-comb(K-c,k)/comb(K,k))_exact_match",
    "diagnostics": "question_mean_then_population_mean",
    "bootstrap": "paired_question_clusters_stratified_by_question_type",
    "bootstrap_iterations": 2000,
    "bootstrap_seed": 20261001,
    "ci_level": 0.95,
}


def bucket_of(question_type: str, choice_type: str) -> str:
    if question_type in ("yes_no", "binary_named") and choice_type == "single":
        return question_type
    if question_type == "multiple_choice" and choice_type in ("single", "multi"):
        return "mc_" + choice_type
    raise AnalysisContractError(f"Unsupported question type: {question_type}/{choice_type}")


def credit(pred: frozenset[str] | None, gold: frozenset[str], bucket: str) -> float:
    if not gold:
        raise AnalysisContractError("Gold answers must be nonempty")
    if pred is None:
        return 0.0
    if bucket != "mc_multi":
        return float(pred == gold)
    return 2.0 * len(pred & gold) / (len(pred) + len(gold))


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def random_credit(option_count: int, gold_count: int, bucket: str) -> float:
    """Expected credit: uniform single choice; independent fair coins for multi."""
    if bucket != "mc_multi":
        return 1.0 / option_count
    return sum(
        math.comb(gold_count, tp) * math.comb(option_count - gold_count, fp)
        * 2.0 * tp / (gold_count + tp + fp)
        for tp in range(gold_count + 1)
        for fp in range(option_count - gold_count + 1)
    ) / 2 ** option_count


def question_metrics(question: dict[str, Any], trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Every declared trial participates; incomplete questions have no point estimate."""
    gold = frozenset(question["gold"])
    bucket = question["bucket"]
    k = len(trials)
    complete = all(t["state"] in ("valid", "invalid", "refusal") for t in trials)
    result = {
        "question_id": question["id"], "bucket": bucket, "points": POINTS[bucket],
        "sampling_n": k, "complete": complete,
        "option_count": len(question["options"]), "gold_count": len(gold),
        "mean_credit": None, "earned_points": None, "exact_at_1": None,
        "pass_at_3": None, "pass_all_at_3": None, "majority_correct_at_3": None,
        "vote_credit": None, "vote_exact": None, "vote_coverage": None,
        "pairwise_agreement": None, "all_same": None, "all_same_wrong": None,
        "valid_repeat_panel": None, "credit_sd": None,
        "best_credit": None, "worst_credit": None,
        "random_credit": random_credit(len(question["options"]), len(gold), bucket),
        "select_all_credit": 2 * len(gold) / (len(question["options"]) + len(gold))
        if bucket == "mc_multi" else None,
    }
    if not complete:
        return result
    preds = [None if t["prediction"] is None else frozenset(t["prediction"]) for t in trials]
    credits = [credit(p, gold, bucket) for p in preds]
    exact = [int(p == gold) for p in preds]
    c = sum(exact)
    counts = Counter(p for p in preds if p is not None)
    vote = next((p for p, n in counts.items() if n > k / 2), None)
    result.update(
        mean_credit=statistics.fmean(credits), earned_points=POINTS[bucket] * statistics.fmean(credits),
        exact_at_1=c / k, vote_credit=credit(vote, gold, bucket),
        vote_exact=float(vote == gold), vote_coverage=float(vote is not None),
        credit_sd=statistics.stdev(credits) if k > 1 else None,
        best_credit=max(credits), worst_credit=min(credits),
    )
    if k >= 3:
        result.update(
            pass_at_3=1 - math.comb(k - c, 3) / math.comb(k, 3),
            pass_all_at_3=math.comb(c, 3) / math.comb(k, 3),
            majority_correct_at_3=sum(
                math.comb(c, hits) * math.comb(k - c, 3 - hits)
                for hits in (2, 3) if hits <= c and 3 - hits <= k - c
            ) / math.comb(k, 3),
        )
    if k > 1:
        valid_panel = all(p is not None for p in preds)
        result.update(
            valid_repeat_panel=float(valid_panel),
            all_same=float(valid_panel and len(counts) == 1),
            all_same_wrong=float(valid_panel and len(counts) == 1 and c == 0),
            pairwise_agreement=(sum(a == b for a, b in itertools.combinations(preds, 2))
                                / math.comb(k, 2)) if valid_panel else None,
        )
    return result


def summarize(questions: list[dict[str, Any]]) -> dict[str, Any]:
    full = sum(q["points"] for q in questions)
    complete = bool(questions) and all(q["complete"] for q in questions)
    earned = sum(q["earned_points"] for q in questions) if complete else None
    out = {
        "questions": len(questions), "complete_questions": sum(q["complete"] for q in questions),
        "full_points": full, "earned_points_mean": earned,
        "score": 100 * earned / full if earned is not None else None,
        "exact_at_1": None, "pass_at_3": None, "pass_all_at_3": None,
        "majority_correct_at_3": None, "vote_score": None, "vote_exact": None,
        "vote_coverage": None, "vote_gain_pp": None, "pairwise_agreement": None,
        "agreement_questions": 0, "valid_repeat_panel": None,
        "all_same": None, "all_same_wrong": None, "credit_sd": None,
        "best_of_n_score": None, "worst_of_n_score": None, "random_score": None,
    }
    if not complete:
        return out
    for key in ("exact_at_1", "pass_at_3", "pass_all_at_3", "majority_correct_at_3",
                "vote_exact", "vote_coverage", "pairwise_agreement", "valid_repeat_panel",
                "all_same", "all_same_wrong", "credit_sd"):
        out[key] = _mean([q[key] for q in questions if q[key] is not None])
    out["agreement_questions"] = sum(q["pairwise_agreement"] is not None for q in questions)
    for dest, source in (("vote_score", "vote_credit"), ("best_of_n_score", "best_credit"),
                         ("worst_of_n_score", "worst_credit"), ("random_score", "random_credit")):
        out[dest] = 100 * sum(q["points"] * q[source] for q in questions) / full
    out["vote_gain_pp"] = out["vote_score"] - out["score"]
    return out


def selection_diagnostics(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Selection behavior conditions on valid multi answers; expose that denominator."""
    multi = [t for t in trials if t["bucket"] == "mc_multi"]
    valid = [t for t in multi if t["state"] == "valid"]
    fields = ("precision", "recall", "f1", "false_positive_rate", "false_negative_rate",
              "extra_count", "missed_count", "cardinality_bias", "overselect", "underselect",
              "same_size_wrong", "exact", "omission_only", "commission_only", "mixed_error")
    values: dict[str, list[dict[str, float | None]]] = {}
    for t in valid:
        pred, gold = set(t["prediction"]), set(t["gold"])
        tp, fp, fn = len(pred & gold), len(pred - gold), len(gold - pred)
        negatives = t["option_count"] - len(gold)
        values.setdefault(t["question_id"], []).append({
            "precision": tp / len(pred), "recall": tp / len(gold),
            "f1": 2 * tp / (len(pred) + len(gold)),
            "false_positive_rate": fp / negatives if negatives else None,
            "false_negative_rate": fn / len(gold), "extra_count": fp, "missed_count": fn,
            "cardinality_bias": len(pred) - len(gold), "overselect": float(len(pred) > len(gold)),
            "underselect": float(len(pred) < len(gold)),
            "same_size_wrong": float(len(pred) == len(gold) and pred != gold),
            "exact": float(fp == fn == 0), "omission_only": float(fp == 0 and fn > 0),
            "commission_only": float(fp > 0 and fn == 0), "mixed_error": float(fp > 0 and fn > 0),
        })
    out = {"multi_slots": len(multi), "valid_multi_slots": len(valid),
           "valid_multi_questions": len(values),
           "valid_multi_rate": len(valid) / len(multi) if multi else None}
    for field in fields:
        per_q = [_mean([v[field] for v in vs if v[field] is not None]) for vs in values.values()]
        out[field] = _mean([v for v in per_q if v is not None])
    out["fpr_questions"] = sum(any(v["false_positive_rate"] is not None for v in vs) for vs in values.values())
    return out


def score_intervals(
    by_model: dict[str, list[dict[str, Any]]], *, iterations: int, seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Share resamples across models; retain the full set of trials within each question."""
    if iterations == 0 or not by_model:
        return [], []
    models = sorted(by_model)
    reference = by_model[models[0]]
    if len(reference) < 2 or any(not q["complete"] for qs in by_model.values() for q in qs):
        return [], []
    ids = [q["question_id"] for q in reference]
    if any([q["question_id"] for q in by_model[m]] != ids for m in models):
        raise AnalysisContractError("Paired intervals require identical question sets")
    strata = [[i for i, q in enumerate(reference) if q["bucket"] == b] for b in POINTS]
    if not any(len(s) > 1 for s in strata):
        return [], []
    denominator = sum(q["points"] for q in reference)
    points = {m: [q["earned_points"] for q in by_model[m]] for m in models}
    estimates = {m: 100 * sum(points[m]) / denominator for m in models}
    draws: dict[str, list[float]] = {m: [] for m in models}
    rng = random.Random(seed)
    for _ in range(iterations):
        indices = [rng.choice(stratum) for stratum in strata for _ in stratum]
        for model in models:
            draws[model].append(100 * sum(points[model][i] for i in indices) / denominator)

    def interval(xs: list[float]) -> tuple[float, float]:
        ordered = sorted(xs)
        def quantile(p: float) -> float:
            pos = (len(ordered) - 1) * p
            lo, hi = math.floor(pos), math.ceil(pos)
            return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
        return quantile(0.025), quantile(0.975)

    cis = []
    pairs = []
    for model in models:
        lo, hi = interval(draws[model])
        cis.append({"model": model, "score": estimates[model], "ci_low": lo, "ci_high": hi,
                    "questions": len(reference), "iterations": iterations})
    for a, b in itertools.combinations(models, 2):
        lo, hi = interval([x - y for x, y in zip(draws[a], draws[b])])
        pairs.append({"model_a": a, "model_b": b, "delta_score_pp": estimates[a] - estimates[b],
                      "ci_low": lo, "ci_high": hi, "questions": len(reference),
                      "interval_scope": "exploratory_unadjusted_95_percent"})
    return cis, pairs
