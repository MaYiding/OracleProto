"""Composite difficulty-coefficient unit tests.

Coverage:
* Default weights produce correct values (hand-computed for a single model + metric);
* Overrides take effect (while other metrics still use defaults);
* A bucket with None is dropped and the rest are renormalized;
* All None -> composite = None;
* Weights summing to != 1 are normalized correctly;
* A bucket with weight 0 is dropped (equivalent to None);
* CSV columns line up with ``per_model_summary.csv``;
* An override metric name not in ``KNOWN_METRICS`` causes ``compute_composite`` to raise;
* Invalid config values cause ``Settings`` to raise at startup;
* ``bucket_of`` maps every (question_type, choice_type) combination correctly;
* End-to-end: run ``run_analysis`` against a fixture run and assert that all new files are generated and sane.
"""
from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from forecast_eval import analysis
from forecast_eval import db as dbmod
from forecast_eval.analysis.composite import (
    COMPOSITE_BUCKETS,
    DEFAULT_WEIGHTS,
    KNOWN_METRICS,
    bucket_of,
    compute_composite,
)
from forecast_eval.analysis.writers import _SUMMARY_FIELDS
from forecast_eval.config import Settings


# --------------------------------------------------------------------------- #
# Pure function: bucket_of
# --------------------------------------------------------------------------- #


def test_bucket_of_collapses_yes_no_and_binary_named() -> None:
    # yes_no / binary_named are structurally always single, so choice_type
    # does not affect the bucket key.
    assert bucket_of("yes_no", "single") == "yes_no"
    assert bucket_of("binary_named", "single") == "binary_named"


def test_bucket_of_splits_multiple_choice_by_answer_mode() -> None:
    assert bucket_of("multiple_choice", "single") == "mc_single"
    assert bucket_of("multiple_choice", "multi") == "mc_multi"


def test_bucket_of_rejects_unknown_inputs() -> None:
    with pytest.raises(ValueError):
        bucket_of("multiple_choice", "trinary")
    with pytest.raises(ValueError):
        bucket_of("free_text", "single")


# --------------------------------------------------------------------------- #
# Pure function: compute_composite
# --------------------------------------------------------------------------- #


def _make_bucket_values(
    metric: str, by_bucket: dict[str, float | None]
) -> dict[str, dict[str, dict[str, float | None]]]:
    """Build ``{model: {metric: {bucket: value}}}``, simplified to a single model + single metric."""
    return {"m1": {metric: dict(by_bucket)}}


def test_default_weights_yield_expected_value() -> None:
    bv = _make_bucket_values(
        "fss",
        {
            "yes_no": 0.8,
            "binary_named": 0.6,
            "mc_single": 0.5,
            "mc_multi": 0.4,
        },
    )
    rep = compute_composite(
        bucket_values_per_model=bv,
        weights_default=DEFAULT_WEIGHTS,
        overrides={},
    )
    info = rep.per_model["m1"]["fss"]
    expected = (
        0.15 * 0.8 + 0.15 * 0.6 + 0.28 * 0.5 + 0.42 * 0.4
    ) / 1.0
    assert info.value == pytest.approx(expected)
    assert info.weights_kind == "default"
    assert tuple(sorted(info.buckets_used)) == (
        "binary_named",
        "mc_multi",
        "mc_single",
        "yes_no",
    )


def test_overrides_apply_to_one_metric_only() -> None:
    bv = {
        "m1": {
            "fss": {
                "yes_no": 1.0,
                "binary_named": 0.5,
                "mc_single": 0.0,
                "mc_multi": 0.0,
            },
            "pass_at_1_avg": {
                "yes_no": 1.0,
                "binary_named": 1.0,
                "mc_single": 0.0,
                "mc_multi": 0.0,
            },
        }
    }
    overrides = {"fss": {"mc_multi": 1.0}}  # only consider mc_multi
    rep = compute_composite(
        bucket_values_per_model=bv,
        weights_default=DEFAULT_WEIGHTS,
        overrides=overrides,
    )
    fss_info = rep.per_model["m1"]["fss"]
    assert fss_info.value == pytest.approx(0.0)  # mc_multi only = 0
    assert fss_info.weights_kind == "overridden"
    assert fss_info.buckets_used == ("mc_multi",)

    pass_info = rep.per_model["m1"]["pass_at_1_avg"]
    expected = 0.15 * 1.0 + 0.15 * 1.0 + 0.28 * 0.0 + 0.42 * 0.0
    assert pass_info.value == pytest.approx(expected)
    assert pass_info.weights_kind == "default"


def test_none_bucket_dropped_and_renormalized() -> None:
    """When binary_named=None, that bucket is dropped and the remaining buckets are renormalized."""
    bv = _make_bucket_values(
        "fss",
        {
            "yes_no": 0.8,
            "binary_named": None,
            "mc_single": 0.5,
            "mc_multi": 0.4,
        },
    )
    rep = compute_composite(
        bucket_values_per_model=bv,
        weights_default=DEFAULT_WEIGHTS,
        overrides={},
    )
    info = rep.per_model["m1"]["fss"]
    denom = 0.15 + 0.28 + 0.42
    expected = (0.15 * 0.8 + 0.28 * 0.5 + 0.42 * 0.4) / denom
    assert info.value == pytest.approx(expected)
    assert tuple(sorted(info.buckets_used)) == ("mc_multi", "mc_single", "yes_no")
    # Normalized weights should sum to 1.0
    assert sum(info.weights_used_normalized.values()) == pytest.approx(1.0)
    # Normalized weight per bucket
    assert info.weights_used_normalized["yes_no"] == pytest.approx(0.15 / denom)
    assert info.weights_used_normalized["mc_single"] == pytest.approx(0.28 / denom)
    assert info.weights_used_normalized["mc_multi"] == pytest.approx(0.42 / denom)


def test_all_none_yields_none() -> None:
    bv = _make_bucket_values(
        "fss",
        {
            "yes_no": None,
            "binary_named": None,
            "mc_single": None,
            "mc_multi": None,
        },
    )
    rep = compute_composite(
        bucket_values_per_model=bv,
        weights_default=DEFAULT_WEIGHTS,
        overrides={},
    )
    info = rep.per_model["m1"]["fss"]
    assert info.value is None
    assert info.buckets_used == ()


def test_unnormalized_weights_still_correct() -> None:
    """Weights summing to != 1 must still be normalized correctly."""
    bv = _make_bucket_values(
        "fss",
        {"mc_single": 0.5, "mc_multi": 0.0},
    )
    rep = compute_composite(
        bucket_values_per_model=bv,
        weights_default={"mc_single": 60.0, "mc_multi": 40.0},  # unnormalized
        overrides={},
    )
    info = rep.per_model["m1"]["fss"]
    expected = (60.0 * 0.5 + 40.0 * 0.0) / 100.0
    assert info.value == pytest.approx(expected)


def test_zero_weight_bucket_excluded() -> None:
    """A bucket with weight 0 behaves like None and does not participate in composition."""
    bv = _make_bucket_values(
        "fss",
        {"mc_single": 0.5, "mc_multi": 0.9},
    )
    rep = compute_composite(
        bucket_values_per_model=bv,
        weights_default={"mc_single": 1.0, "mc_multi": 0.0},
        overrides={},
    )
    info = rep.per_model["m1"]["fss"]
    assert info.value == pytest.approx(0.5)
    assert info.buckets_used == ("mc_single",)


def test_unknown_metric_in_override_raises() -> None:
    bv = _make_bucket_values("fss", {"yes_no": 0.5})
    overrides = {"unknown_metric": {"yes_no": 1.0}}
    with pytest.raises(ValueError, match="not a known metric"):
        compute_composite(
            bucket_values_per_model=bv,
            weights_default=DEFAULT_WEIGHTS,
            overrides=overrides,
        )


def test_known_metrics_align_with_summary_fields() -> None:
    """``KNOWN_METRICS`` must equal ``_SUMMARY_FIELDS`` minus the metadata columns."""
    summary_data = {f for f in _SUMMARY_FIELDS if f not in ("model", "sampling_n")}
    assert KNOWN_METRICS == summary_data


def test_default_weights_sum_to_one() -> None:
    """The four nested-difficulty coefficients must sum to one."""
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)
    assert set(DEFAULT_WEIGHTS) == set(COMPOSITE_BUCKETS)


# --------------------------------------------------------------------------- #
# Settings startup-time validation
# --------------------------------------------------------------------------- #


def _settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("MODELS", "m1")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")


def test_settings_default_composite_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _settings_env(monkeypatch)
    s = Settings()
    assert s.COMPOSITE_WEIGHTS == DEFAULT_WEIGHTS
    assert s.COMPOSITE_WEIGHT_OVERRIDES == {}


def test_settings_parses_csv_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _settings_env(monkeypatch)
    monkeypatch.setenv(
        "COMPOSITE_WEIGHTS",
        "yes_no=0.10,mc_multi=0.90",
    )
    monkeypatch.setenv(
        "COMPOSITE_WEIGHT_OVERRIDES",
        "fss=mc_single=0.3,mc_multi=0.7;cohen_kappa=mc_multi=1.0",
    )
    s = Settings()
    assert s.COMPOSITE_WEIGHTS == {"yes_no": 0.10, "mc_multi": 0.90}
    assert s.COMPOSITE_WEIGHT_OVERRIDES == {
        "fss": {"mc_single": 0.3, "mc_multi": 0.7},
        "cohen_kappa": {"mc_multi": 1.0},
    }


@pytest.mark.parametrize(
    "env_key,env_value,expected_msg",
    [
        ("COMPOSITE_WEIGHTS", "wrong_bucket=0.5", "not in"),
        ("COMPOSITE_WEIGHTS", "mc_single=-0.1,mc_multi=1.0", "must be >= 0"),
        ("COMPOSITE_WEIGHTS", "mc_single=0,mc_multi=0", "at least one weight"),
        (
            "COMPOSITE_WEIGHT_OVERRIDES",
            "fss=unknown=0.5",
            "not in",
        ),
    ],
)
def test_settings_rejects_invalid_composite_config(
    monkeypatch: pytest.MonkeyPatch,
    env_key: str,
    env_value: str,
    expected_msg: str,
) -> None:
    _settings_env(monkeypatch)
    monkeypatch.setenv(env_key, env_value)
    with pytest.raises(ValueError, match=expected_msg):
        Settings()


# --------------------------------------------------------------------------- #
# End-to-end: run_analysis file output
# --------------------------------------------------------------------------- #


def _seed_questions(conn: sqlite3.Connection) -> None:
    rows = [
        ("q1", "single", "yes_no",          "ev1", json.dumps(["Yes", "No"]),     "A", "2026-03-01"),
        ("q2", "single", "binary_named",    "ev2", json.dumps(["Alpha", "Beta"]), "B", "2026-03-02"),
        ("q3", "multi",  "multiple_choice", "ev3", json.dumps(["x", "y", "z"]),   "A, C", "2026-03-03"),
    ]
    now = dbmod.utcnow_iso()
    conn.executemany(
        "INSERT INTO questions (id, choice_type, question_type, event, options, answer, end_time, imported_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        [(*r, now) for r in rows],
    )


def _sample_dict(
    *,
    question_id: str,
    sample_idx: int,
    correct: int | None,
    parse_ok: int,
    error: str | None,
    letters: list[str] | None = None,
) -> dict:
    return {
        "question_id": question_id,
        "sample_idx": sample_idx,
        "user_prompt": "P",
        "final_answer_letters": json.dumps(sorted(letters)) if letters is not None else None,
        "final_answer_raw": "raw",
        "correct": correct,
        "parse_ok": parse_ok,
        "tool_calls_count": 2,
        "react_steps": 3,
        "prompt_tokens": 100,
        "completion_tokens": 40,
        "reasoning_tokens": 0,
        "latency_ms": 1000,
        "messages_trace": None,
        "search_calls": None,
        "error": error,
        "created_at": dbmod.utcnow_iso(),
        "finish_reason": "stop",
        "nudges_used": 0,
        "step_metrics": json.dumps([
            {"step": 0, "prompt": 100, "completion": 40, "reasoning": 0,
             "latency_ms": 1000, "finish_reason": "stop", "n_tool_calls": 2},
        ]),
        "response_id": "resp_test",
        "system_fingerprint": "fp_test",
        "service_tier": "default",
        "belief_final": None,
        "belief_trace": None,
        "belief_parse_ok": 0,
        "final_answer_retry_used": 0,
    }


def _build_fixture_run(tmp_path: Path) -> Path:
    """Two models, K=3. Model A: all single-choice correct, multi-choice two right one wrong; Model B: all wrong."""
    run_dir = tmp_path / "run1"
    db_dir = run_dir / "db"
    db_dir.mkdir(parents=True)

    def _make_conn(path: Path, model: str) -> sqlite3.Connection:
        conn = dbmod.connect(path)
        dbmod.init_schema(conn, sampling_n=3)
        _seed_questions(conn)
        dbmod.register_run_meta(
            conn,
            run_id="run1",
            model=model,
            sampling_n=3,
            filters_snapshot={},
            config_snapshot={},
            source_db_hash="a" * 64,
            metadata_hash="b" * 64,
            prompt_templates_hash="c" * 64,
        )
        return conn

    conn_a = _make_conn(db_dir / "m__a.db", "m/a")
    for qid, gt_letters, _multi in [
        ("q1", ["A"], False),  # yes_no, gt=A
        ("q2", ["B"], False),  # binary_named, gt=B
    ]:
        for i in range(3):
            dbmod.upsert_sample_sync(
                conn_a,
                3,
                _sample_dict(
                    question_id=qid, sample_idx=i, correct=1,
                    parse_ok=1, error=None, letters=gt_letters,
                ),
            )
    # q3 (multi): K=3, two correct {A,C}, one wrong {A,B}
    for i, letters in enumerate([["A", "C"], ["A", "C"], ["A", "B"]]):
        correct = 1 if set(letters) == {"A", "C"} else 0
        dbmod.upsert_sample_sync(
            conn_a,
            3,
            _sample_dict(
                question_id="q3", sample_idx=i, correct=correct,
                parse_ok=1, error=None, letters=letters,
            ),
        )
    conn_a.commit()
    conn_a.close()

    conn_b = _make_conn(db_dir / "m__b.db", "m/b")
    for qid, wrong_letters in [
        ("q1", ["B"]),  # all wrong
        ("q2", ["A"]),
    ]:
        for i in range(3):
            dbmod.upsert_sample_sync(
                conn_b,
                3,
                _sample_dict(
                    question_id=qid, sample_idx=i, correct=0,
                    parse_ok=1, error=None, letters=wrong_letters,
                ),
            )
    for i in range(3):
        dbmod.upsert_sample_sync(
            conn_b,
            3,
            _sample_dict(
                question_id="q3", sample_idx=i, correct=0,
                parse_ok=1, error=None, letters=["B"],
            ),
        )
    conn_b.commit()
    conn_b.close()

    manifest = {
        "run_id": "run1",
        "models": ["m/a", "m/b"],
        "model_files": {"m/a": "m__a.db", "m/b": "m__b.db"},
        "sampling_n": 3,
        "analysis_schema": "v5",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


def test_run_analysis_writes_composite_files(tmp_path: Path) -> None:
    run_dir = _build_fixture_run(tmp_path)
    paths = analysis.run_analysis(run_dir)
    names = {p.name for p in paths}
    assert "per_model_composite.csv" in names
    assert "composite_meta.json" in names


def test_composite_csv_columns_match_summary(tmp_path: Path) -> None:
    """``per_model_composite.csv`` column order = ``model + sampling_n + weights_kind +
    [_SUMMARY_FIELDS data columns]``."""
    run_dir = _build_fixture_run(tmp_path)
    analysis.run_analysis(run_dir)
    expected_data = [
        f for f in _SUMMARY_FIELDS if f not in ("model", "sampling_n")
    ]
    expected = ["model", "sampling_n", "weights_kind", *expected_data]
    path = run_dir / "analysis" / "per_model_composite.csv"
    with path.open() as f:
        header = next(csv.reader(f))
    assert header == expected, "per_model_composite.csv header drift"


def test_composite_meta_records_used_buckets(tmp_path: Path) -> None:
    run_dir = _build_fixture_run(tmp_path)
    analysis.run_analysis(run_dir)
    meta = json.loads(
        (run_dir / "analysis" / "composite_meta.json").read_text()
    )
    assert meta["weights_default"] == DEFAULT_WEIGHTS

    # m/a's fss has data in yes_no + binary_named + mc_multi (q3 cutoff-skipped
    # for m/a, but q3 carries multi answer letters that still produce an
    # eligible aggregate within mc_multi for m/a; if not, the bucket is
    # dropped). Whichever buckets contribute, normalized weights must sum to 1.
    info = meta["per_model"]["m/a"]["fss"]
    assert set(info["buckets_used"]).issubset(set(DEFAULT_WEIGHTS))
    total = sum(info["weights_used_normalized"].values())
    # Stored weights are rounded to 6 decimals in composite_meta.json; treat
    # rounding-band drift as the equality threshold.
    assert abs(total - 1.0) < 1e-5
    assert info["weights_kind"] == "default"


def test_composite_overrides_propagate_to_csv(tmp_path: Path) -> None:
    run_dir = _build_fixture_run(tmp_path)
    analysis.run_analysis(
        run_dir,
        composite_overrides={"fss": {"mc_multi": 1.0}},
    )
    # The weights_kind column in the composite csv should be overridden (any metric hitting an override)
    with (run_dir / "analysis" / "per_model_composite.csv").open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        assert r["weights_kind"] == "overridden"

    meta = json.loads(
        (run_dir / "analysis" / "composite_meta.json").read_text()
    )
    fss_info = meta["per_model"]["m/a"]["fss"]
    assert fss_info["weights_kind"] == "overridden"
    assert fss_info["buckets_used"] == ["mc_multi"]
    # Other metrics remain default
    pass_info = meta["per_model"]["m/a"]["pass_at_1_avg"]
    assert pass_info["weights_kind"] == "default"


def test_slice_csv_now_carries_v5_columns(tmp_path: Path) -> None:
    """``per_model_by_bucket.csv`` should carry v5 columns (FSS / Cohen κ /
    Hamming / Fleiss κ / mean_entropy / VCI / MVG); at least one row should
    have non-NULL values."""
    run_dir = _build_fixture_run(tmp_path)
    analysis.run_analysis(run_dir)
    path = run_dir / "analysis" / "per_model_by_bucket.csv"
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert rows, "per_model_by_bucket.csv empty"
    # at least one row has non-empty fss
    fss_seen = any(r.get("fss") not in ("", None) for r in rows)
    assert fss_seen, "per_model_by_bucket.csv has no fss values"


def test_overall_json_includes_composite(tmp_path: Path) -> None:
    run_dir = _build_fixture_run(tmp_path)
    analysis.run_analysis(run_dir)
    overall = json.loads(
        (run_dir / "analysis" / "overall.json").read_text()
    )
    assert "composite" in overall
    per = overall["composite"]["per_model"]["m/a"]
    assert "fss" in per
    assert "weights_kind" in per["fss"]
