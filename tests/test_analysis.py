"""Contracts for fixed points, repeated answers, completeness, and read-only rescoring."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from forecast_eval import analysis, db
from forecast_eval.analysis.__main__ import _cli
from forecast_eval.analysis.scoring import credit, question_metrics, random_credit, score_intervals, selection_diagnostics
from forecast_eval.errors import AnalysisContractError, AnalysisIncompleteError


def _build_fixture_run(tmp_path: Path, n: int = 3) -> Path:
    run = tmp_path / "runs" / "run1"
    (run / "db").mkdir(parents=True)
    questions = [
        ("q1", "single", "yes_no", "event1", '["Yes","No"]', "A", "2026-06-01"),
        ("q2", "single", "binary_named", "event2", '["X","Y"]', "B", "2026-06-01"),
        ("q3", "single", "multiple_choice", "event3", '["x","y","z"]', "B", "2026-06-01"),
        ("q4", "multi", "multiple_choice", "event4", '["x","y","z","w"]', "A,C", "2026-06-01"),
    ]
    predictions = {"q1": [["A"], ["A"], ["B"]], "q2": [["B"]] * 3,
                   "q3": [["B"], ["A"], ["A"]], "q4": [["A", "C"], ["A"], ["A", "B", "C"]]}
    for model in ("a", "b"):
        conn = db.connect(run / "db" / (model + ".db"))
        db.init_schema(conn, sampling_n=n)
        conn.executemany("INSERT INTO questions (id,choice_type,question_type,event,options,answer,end_time,imported_at) VALUES (?,?,?,?,?,?,?,?)",
                         [(*q, "2026-07-01") for q in questions])
        db.register_run_meta(conn, run_id="run1", model=model, sampling_n=n,
                             filters_snapshot={}, config_snapshot={"SCORE_ANSWERS": False},
                             source_db_hash="s", metadata_hash="d", prompt_templates_hash="p")
        for q in questions:
            for i in range(n):
                pred = predictions[q[0]][i % 3] if model == "a" else q[5].split(",")
                row = {name: None for name, _ in db.PER_SAMPLE_COLUMNS}
                row.update(question_id=q[0], sample_idx=i, parse_ok=1, correct=None,
                           final_answer_letters=json.dumps(pred), created_at="2026-07-01T00:00:00Z",
                           tool_calls_count=2, react_steps=3, latency_ms=1000)
                db.upsert_sample_sync(conn, n, row)
        db.finish_run_meta(conn, "run1")
        conn.close()
    (run / "manifest.json").write_text(json.dumps({"run_id": "run1", "sampling_n": n,
        "models": ["a", "b"], "model_files": {"a": "a.db", "b": "b.db"},
        "filters": {"question_ids": [q[0] for q in questions]}}))
    return run


def _rows(run: Path, name: str) -> list[dict]:
    with (run / "analysis" / "score" / (name + ".csv")).open() as f:
        return list(csv.DictReader(f))


def _update(run: Path, sql: str, args=(), model="a") -> None:
    with sqlite3.connect(run / "db" / (model + ".db")) as conn:
        conn.execute(sql, args)


def test_fixed_points_and_f1_from_unscored_observations(tmp_path):
    run = _build_fixture_run(tmp_path)
    analysis.run_analysis(run, bootstrap_iterations=0)
    a, b = _rows(run, "score_summary")
    assert float(a["score"]) == pytest.approx(100 * 4.8 / 7)
    assert float(a["earned_points_mean"]) == pytest.approx(4.8)
    assert float(a["full_points"]) == 7
    assert float(b["score"]) == 100
    assert float(a["exact_at_1"]) == pytest.approx(7 / 12)
    assert float(a["pass_at_3"]) == 1
    assert float(a["pass_all_at_3"]) == .25
    assert float(a["vote_score"]) == pytest.approx(200 / 7)
    trials = [float(r["score"]) for r in _rows(run, "score_by_trial") if r["model"] == "a"]
    assert sum(trials) / 3 == pytest.approx(float(a["score"]))


def test_score_is_item_weighted_not_bucket_weighted(tmp_path):
    run = _build_fixture_run(tmp_path)
    _update(run, "INSERT INTO questions SELECT 'q5',choice_type,question_type,event,options,answer,end_time,imported_at FROM questions WHERE id='q1'")
    with sqlite3.connect(run / "db" / "a.db") as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(run_results)")]
        conn.execute("INSERT INTO run_results SELECT 'q5'," + ",".join(cols[1:]) + " FROM run_results WHERE question_id='q1'")
    manifest = json.loads((run / "manifest.json").read_text())
    manifest.update(models=["a"])
    manifest["filters"]["question_ids"].append("q5")
    (run / "manifest.json").write_text(json.dumps(manifest))
    analysis.run_analysis(run, bootstrap_iterations=0)
    assert float(_rows(run, "score_summary")[0]["score"]) == pytest.approx(100 * (4.8 + 2 / 3) / 8)


@pytest.mark.parametrize("pred, expected", [(None, 0), (frozenset("AC"), 1), (frozenset("A"), 2/3),
                                            (frozenset("ABC"), 4/5), (frozenset("BD"), 0)])
def test_multi_partial_credit(pred, expected):
    assert credit(pred, frozenset("AC"), "mc_multi") == pytest.approx(expected)


def test_diagnostics_distinguish_substitution_from_cardinality():
    def trial(pred):
        return dict(question_id="q", bucket="mc_multi", state="valid", prediction=list(pred), gold=list("AC"), option_count=4)
    rows = selection_diagnostics([trial("AB"), trial("A"), trial("ABC")])
    assert rows["cardinality_bias"] == 0
    assert rows["same_size_wrong"] == pytest.approx(1/3)
    assert rows["mixed_error"] == pytest.approx(1/3)
    assert rows["omission_only"] == pytest.approx(1/3)
    assert rows["commission_only"] == pytest.approx(1/3)
    assert rows["extra_count"] == pytest.approx(2/3)
    assert rows["missed_count"] == pytest.approx(2/3)


@pytest.mark.parametrize("update", ["s1_created_at=NULL", "s1_error='network'"])
def test_missing_or_failed_slot_never_disappears_from_denominator(tmp_path, update):
    run = _build_fixture_run(tmp_path)
    _update(run, "UPDATE run_results SET " + update + " WHERE question_id='q4'")
    with pytest.raises(AnalysisIncompleteError):
        analysis.run_analysis(run, bootstrap_iterations=0)
    assert _rows(run, "score_summary")[0]["score"] == ""
    assert _cli([str(run), "--bootstrap-iterations", "0"]) == 2
    assert _cli([str(run), "--bootstrap-iterations", "0", "--allow-incomplete"]) == 0


@pytest.mark.parametrize("update", ["s0_parse_ok=0", "s0_error='content_policy'", "s0_final_answer_letters='[\"Z\"]'",
                                   "s0_final_answer_letters='[\"A\",\"B\"]'"])
def test_invalid_and_refusal_are_zero_credit(tmp_path, update):
    run = _build_fixture_run(tmp_path)
    _update(run, "UPDATE run_results SET " + update + " WHERE question_id='q1'")
    analysis.run_analysis(run, bootstrap_iterations=0)
    assert float(_rows(run, "score_summary")[0]["score"]) == pytest.approx(100 * (4.8 - 1/3) / 7)


def test_missing_db_is_not_silently_dropped(tmp_path):
    run = _build_fixture_run(tmp_path)
    (run / "db" / "b.db").unlink()
    with pytest.raises(FileNotFoundError):
        analysis.run_analysis(run, bootstrap_iterations=0)


def test_read_only_deterministic_and_no_disused_outputs(tmp_path):
    run = _build_fixture_run(tmp_path)
    paths = list((run / "db").glob("*.db")) + [run / "manifest.json"]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    written = analysis.run_analysis(run, bootstrap_iterations=200)
    outputs = {p: p.read_bytes() for p in written}
    analysis.run_analysis(run, bootstrap_iterations=200)
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths} == before
    assert {p: p.read_bytes() for p in written} == outputs
    assert not any("rank" in p.name or "composite" in p.name or "shrinkage" in p.name for p in written)
    assert _rows(run, "score_pairwise") == []


def test_contract_fingerprint_changes_with_inference_settings(tmp_path):
    run = _build_fixture_run(tmp_path)
    analysis.run_analysis(run, bootstrap_iterations=0)
    meta = run / "analysis" / "score" / "scoring_meta.json"
    a = json.loads(meta.read_text())
    analysis.run_analysis(run, bootstrap_iterations=200)
    b = json.loads(meta.read_text())
    assert a["contract_sha256"] != b["contract_sha256"]
    assert a["sources"] == b["sources"]


def test_gold_mismatch_fails_before_comparison(tmp_path):
    run = _build_fixture_run(tmp_path)
    _update(run, "UPDATE questions SET answer='B' WHERE id='q1'")
    with pytest.raises(AnalysisContractError, match="different question"):
        analysis.run_analysis(run)


def test_cutoff_uses_common_declared_panel(tmp_path):
    run = _build_fixture_run(tmp_path)
    _update(run, "UPDATE run_results SET s0_error='skipped_training_cutoff',s1_error='skipped_training_cutoff',s2_error='skipped_training_cutoff' WHERE question_id='q4'")
    analysis.run_analysis(run, bootstrap_iterations=0)
    a, b = _rows(run, "score_summary")
    assert int(a["questions"]) == int(b["questions"]) == 3
    assert float(a["score"]) == pytest.approx(100 * (2/3 + 1 + 2/3) / 4)
    assert _rows(run, "coverage")[0]["cutoff_questions"] == "1"


def test_k_one_does_not_fabricate_three_trial_metrics(tmp_path):
    run = _build_fixture_run(tmp_path, n=1)
    analysis.run_analysis(run, bootstrap_iterations=0)
    row = _rows(run, "score_summary")[0]
    assert row["pass_at_3"] == row["pass_all_at_3"] == row["trial_score_sd_pp"] == ""


def test_pass_at_three_uses_without_replacement_estimator():
    q = dict(id="q", bucket="yes_no", gold=["A"], options=["yes", "no"])
    rows = [dict(state="valid", prediction=[p]) for p in "AABBB"]
    result = question_metrics(q, rows)
    assert result["pass_at_3"] == pytest.approx(.9)
    assert result["pass_all_at_3"] == 0
    assert result["majority_correct_at_3"] == pytest.approx(.3)


def test_invalid_repeats_are_not_agreement():
    q = dict(id="q", bucket="yes_no", gold=["A"], options=["yes", "no"])
    result = question_metrics(q, [dict(state="invalid", prediction=None)] * 3)
    assert result["pairwise_agreement"] is None
    assert result["all_same"] == result["all_same_wrong"] == result["vote_coverage"] == 0


def test_random_baseline_matches_exhaustive_small_problem():
    sets = [frozenset(c) for n in range(5) for c in itertools.combinations("ABCD", n)]
    expected = sum(credit(p, frozenset("AC"), "mc_multi") for p in sets) / len(sets)
    assert random_credit(4, 2, "mc_multi") == pytest.approx(expected)


def test_reference_union_is_read_only_and_rejects_conflicting_success(tmp_path):
    run = _build_fixture_run(tmp_path)
    ref = run.parent / "reference"
    ref.mkdir()
    source = ref / "a.db"
    shutil.copyfile(run / "db" / "a.db", source)
    snapshot = {"SCORE_ANSWERS": False, "COLLECTION_REFERENCE_DBS": {"a": [str(source)]}}
    _update(run, "UPDATE run_meta SET config_snapshot=?", (json.dumps(snapshot),))
    _update(run, "UPDATE run_results SET s1_created_at=NULL WHERE question_id='q4'")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    analysis.run_analysis(run, bootstrap_iterations=0)
    assert float(_rows(run, "score_summary")[0]["score"]) == pytest.approx(100 * 4.8 / 7)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    _update(run, "UPDATE run_results SET s0_final_answer_letters='[\"B\"]' WHERE question_id='q1'")
    with pytest.raises(AnalysisContractError, match="Conflicting duplicate"):
        analysis.run_analysis(run)


def test_paired_bootstrap_resamples_whole_questions_together():
    qs = [dict(question_id=str(i), bucket="yes_no", points=1, earned_points=float(i), complete=True)
          for i in (0, 1)]
    cis, pairs = score_intervals({"a": qs, "b": qs}, iterations=200, seed=1)
    assert cis[0]["ci_low"] == 0
    assert cis[0]["ci_high"] == 100
    assert pairs[0]["delta_score_pp"] == pairs[0]["ci_low"] == pairs[0]["ci_high"] == 0


def _catalog_fixture(tmp_path):
    import gzip
    from forecast_eval.analysis.score_inputs import canonical_hash
    run = _build_fixture_run(tmp_path)
    folder = run.parent / "collection"
    folder.mkdir()
    corpus = tmp_path / "corpus.db"
    with sqlite3.connect(run / "db" / "a.db") as source, sqlite3.connect(corpus) as target:
        source.backup(target)
        target.execute("ALTER TABLE questions RENAME TO test_cases")
    target.close()
    source.close()
    sources, records = [], []
    for model in ("a", "b"):
        with sqlite3.connect(run / "db" / (model + ".db")) as conn:
            conn.row_factory = sqlite3.Row
            meta = dict(conn.execute("SELECT * FROM run_meta").fetchone())
            for row in conn.execute("SELECT * FROM run_results"):
                for i in range(3):
                    prefix = f"s{i}_"
                    records.append(dict(source_id=model, profile_id=model, cohort="continuation",
                                        question_id=row["question_id"], sample_idx=i,
                                        result={k[len(prefix):]: row[k] for k in row.keys() if k.startswith(prefix)}))
        sources.append(dict(source_id=model, profile_id=model, cohort="continuation", path=str(run / "db" / (model + ".db")),
                            metadata=meta, observed_samples=12,
                            collection_contract_hash=canonical_hash(db.collection_contract(json.loads(meta["config_snapshot"])))))
    export = folder / "observations.jsonl.gz"
    with gzip.open(export, "wt") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    catalog = dict(source_db=str(corpus), source_sha256=hashlib.sha256(corpus.read_bytes()).hexdigest(),
                   sources=sources, coverage=[{"profiles": {"a": {}, "b": {}, "pending": {}}}],
                   anchor_question_ids=["q1", "q2"], additional_question_ids=["q3", "q4"],
                   observations_path=str(export), observations_encoding="gzip",
                   observations_sha256=hashlib.sha256(export.read_bytes()).hexdigest())
    path = folder / "catalog.json"
    path.write_text(json.dumps(catalog))
    return path, run


def test_catalog_matches_database_scoring_and_declared_profiles(tmp_path):
    path, run = _catalog_fixture(tmp_path)
    analysis.run_analysis(run, bootstrap_iterations=0)
    expected = _rows(run, "score_summary")
    analysis.run_analysis(path, profiles=["a", "b"], bootstrap_iterations=0)
    assert _rows(path.parent, "score_summary") == expected
    with pytest.raises(AnalysisIncompleteError):
        analysis.run_analysis(path, bootstrap_iterations=0)
    assert _rows(path.parent, "coverage")[-1]["missing_slots"] == "12"


def test_catalog_rejects_changed_export(tmp_path):
    path, _ = _catalog_fixture(tmp_path)
    export = path.parent / "observations.jsonl.gz"
    export.write_bytes(export.read_bytes() + b"extra")
    with pytest.raises(AnalysisContractError, match="observations hash"):
        analysis.run_analysis(path, profiles=["a", "b"], bootstrap_iterations=0)
