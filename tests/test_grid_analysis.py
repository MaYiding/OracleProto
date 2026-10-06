"""Tests for `forecast_eval/analysis/grid.py`.

The fixture builds a tiny 2×2×2 grid (2 real_models × 2 R × 2 C = 8 virtual
slugs) entirely in memory: SampleRow / _QuestionProbabilityRow are
constructed by hand so we don't pay sqlite + flatten roundtrip cost on every
test. Cells get distinct BI values so pareto / winrate / marginals are
non-trivial.

`run_grid_analysis` is exercised with in-memory samples. The reflection-A/B
isolation test uses real DBs because `find_paired_runs` reads `run_meta` rows directly.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from forecast_eval import db as dbmod
from forecast_eval.analysis.accuracy import _aggregate
from forecast_eval.analysis.behavior import find_paired_runs
from forecast_eval.analysis.flatten import SampleRow
from forecast_eval.analysis.grid import (
    GridCell,
    WinrateRow,
    build_grid_summary,
    marginal_along_C,
    marginal_along_R,
    paired_bootstrap_per_cell,
    pareto_frontier,
    run_grid_analysis,
    winrate_matrix,
)
from forecast_eval.analysis.probabilistic import _QuestionProbabilityRow
from forecast_eval.analysis.proper_score import ModelProbabilisticAggregate


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #


def _make_sample(
    *,
    slug: str,
    qid: str,
    sample_idx: int,
    correct: int,
    probs: list[float],
    tool_calls: int = 1,
    latency_ms: int = 100,
    parse_ok: int = 1,
    belief_parse_ok: int = 1,
) -> SampleRow:
    return SampleRow(
        model=slug,
        question_id=qid,
        question_type="yes_no",
        choice_type="single",
        options=["yes", "no"],
        sample_idx=sample_idx,
        correct=correct,
        parse_ok=parse_ok,
        tool_calls_count=tool_calls,
        react_steps=2,
        prompt_tokens=200,
        completion_tokens=100,
        reasoning_tokens=0,
        latency_ms=latency_ms,
        final_answer_letters='["A"]' if correct else '["B"]',
        error=None,
        created_at="2026-04-25T12:00:00Z",
        finish_reason="stop",
        nudges_used=0,
        belief_final=json.dumps({"A": probs[0], "B": probs[1]}),
        belief_trace=None,
        belief_parse_ok=belief_parse_ok,
        probabilities=probs,
        is_fallback=False,
    )


def _make_q_row(
    *,
    slug: str,
    qid: str,
    probs: list[float],
    obs: list[int],
) -> _QuestionProbabilityRow:
    return _QuestionProbabilityRow(
        model=slug,
        question_id=qid,
        question_type="yes_no",
        choice_type="single",
        options=["yes", "no"],
        obs=obs,
        probs=probs,
        n_samples=1,
        n_fallback=0,
    )


# Grid axes — small enough to keep tests fast, large enough to exercise
# every code path (cartesian fan-out, marginal slicing, pareto across 4+
# points, winrate matrix between 2 real models).
_REAL_MODELS = ["m_a", "m_b"]
_R_LIST = [5, 10]
_C_LIST = [1, 3]
_QIDS = ["q1", "q2", "q3"]
_GT_MAP = {qid: frozenset({"A"}) for qid in _QIDS}


def _cell_probs(real_model: str, R: int, C: int, qid: str) -> list[float]:
    """Per-cell prob vector. Higher C → higher confidence in correct answer.

    `m_a` is uniformly more confident than `m_b`; both saturate around C=3.
    Different qids get slight perturbations so per-question BS is non-degenerate
    (paired bootstrap CIs collapse to zero on identical inputs)."""
    base_a = {1: 0.65, 3: 0.85}[C]
    base_b = {1: 0.55, 3: 0.75}[C]
    base = base_a if real_model == "m_a" else base_b
    # R=10 vs R=5 nudges by 0.03 (more search results → slightly better)
    base += 0.03 if R == 10 else 0.0
    # qid-level jitter so per-q BS varies
    jitter = {"q1": 0.0, "q2": 0.05, "q3": -0.03}[qid]
    p_correct = max(0.51, min(0.99, base + jitter))
    return [p_correct, 1.0 - p_correct]


def _build_fixture() -> tuple[
    dict[str, list[SampleRow]],
    dict[str, list[_QuestionProbabilityRow]],
    dict[str, frozenset[str]],
    dict[str, object],
]:
    """Construct samples_by_model + rows_by_model + gt + manifest_grid for a 2×2×2 grid."""
    samples_by_model: dict[str, list[SampleRow]] = {}
    rows_by_model: dict[str, list[_QuestionProbabilityRow]] = {}
    for real in _REAL_MODELS:
        for R in _R_LIST:
            for C in _C_LIST:
                slug = dbmod.compose_virtual_slug(real, R, C)
                samples: list[SampleRow] = []
                rows: list[_QuestionProbabilityRow] = []
                for qid in _QIDS:
                    probs = _cell_probs(real, R, C, qid)
                    obs = [1, 0]  # correct letter is "A"
                    correct = 1 if probs[0] >= 0.5 else 0
                    samples.append(_make_sample(
                        slug=slug, qid=qid, sample_idx=0,
                        correct=correct, probs=probs,
                        # tool_calls per cell varies by C so mean_search_calls differs
                        tool_calls=C,
                        # latency varies by R (bigger R = more tokens)
                        latency_ms=100 + 20 * R,
                    ))
                    rows.append(_make_q_row(
                        slug=slug, qid=qid, probs=probs, obs=obs,
                    ))
                samples_by_model[slug] = samples
                rows_by_model[slug] = rows
    manifest_grid = {
        "r_list": _R_LIST,
        "c_list": _C_LIST,
        "default_r": _R_LIST[0],
        "default_c": _C_LIST[0],
        "real_models": _REAL_MODELS,
        "n_cells": 2 * 2 * 2,
    }
    return samples_by_model, rows_by_model, _GT_MAP, manifest_grid


# --------------------------------------------------------------------------- #
# 10.2 build_grid_summary triplet decoding
# --------------------------------------------------------------------------- #


def test_build_grid_summary_decodes_triplets() -> None:
    samples_by_model, rows_by_model, gt_map, manifest_grid = _build_fixture()
    grid = build_grid_summary(
        samples_by_model, gt_map, rows_by_model, manifest_grid,
    )
    # Every virtual slug parses back to a triplet, so we get 8 cells.
    assert len(grid) == 8
    assert set(grid.keys()) == {
        ("m_a", 5, 1), ("m_a", 5, 3), ("m_a", 10, 1), ("m_a", 10, 3),
        ("m_b", 5, 1), ("m_b", 5, 3), ("m_b", 10, 1), ("m_b", 10, 3),
    }
    # Spot-check one cell's BI is a finite number (not None) and that
    # mean_search_calls reflects the per-cell tool_calls override.
    cell = grid[("m_a", 5, 3)]
    assert cell.real_model == "m_a"
    assert cell.R == 5
    assert cell.C == 3
    assert cell.probabilistic_aggregate.bi is not None
    assert cell.mean_search_calls == 3.0  # tool_calls=C=3


def test_build_grid_summary_skips_non_virtual_slugs() -> None:
    """A real-only slug (e.g. legacy v4) is silently skipped by the parser."""
    samples_by_model, rows_by_model, gt_map, manifest_grid = _build_fixture()
    # Inject a real-slug-only entry alongside the virtual slugs.
    samples_by_model["legacy_real_only"] = []
    rows_by_model["legacy_real_only"] = []
    grid = build_grid_summary(
        samples_by_model, gt_map, rows_by_model, manifest_grid,
    )
    # 8 virtual + 0 legacy = 8.
    assert len(grid) == 8


# --------------------------------------------------------------------------- #
# 10.3 marginal_along_C / R
# --------------------------------------------------------------------------- #


def test_marginal_along_C_filter_and_sort() -> None:
    samples_by_model, rows_by_model, gt_map, manifest_grid = _build_fixture()
    grid = build_grid_summary(
        samples_by_model, gt_map, rows_by_model, manifest_grid,
    )
    cells = marginal_along_C(grid, fix_R=5)
    # 2 real_models × 2 C cells = 4
    assert len(cells) == 4
    assert all(c.R == 5 for c in cells)
    # Sort key is (real_model, C) ascending.
    keys = [(c.real_model, c.C) for c in cells]
    assert keys == [("m_a", 1), ("m_a", 3), ("m_b", 1), ("m_b", 3)]


def test_marginal_along_R_filter_and_sort() -> None:
    samples_by_model, rows_by_model, gt_map, manifest_grid = _build_fixture()
    grid = build_grid_summary(
        samples_by_model, gt_map, rows_by_model, manifest_grid,
    )
    cells = marginal_along_R(grid, fix_C=1)
    assert len(cells) == 4
    assert all(c.C == 1 for c in cells)
    keys = [(c.real_model, c.R) for c in cells]
    assert keys == [("m_a", 5), ("m_a", 10), ("m_b", 5), ("m_b", 10)]


def test_marginal_returns_empty_on_unknown_axis_value() -> None:
    samples_by_model, rows_by_model, gt_map, manifest_grid = _build_fixture()
    grid = build_grid_summary(
        samples_by_model, gt_map, rows_by_model, manifest_grid,
    )
    assert marginal_along_C(grid, fix_R=999) == []
    assert marginal_along_R(grid, fix_C=999) == []


# --------------------------------------------------------------------------- #
# 10.4 Pareto frontier
# --------------------------------------------------------------------------- #


def _synthetic_cell(
    *, real_model: str, R: int, C: int,
    cost: float, bi: float,
) -> GridCell:
    """Tiny GridCell with the two attributes pareto_frontier reads."""
    prob_agg = ModelProbabilisticAggregate(
        n_questions=10, n_fallback=0, fallback_share=0.0,
        bi=bi, bi_dec=None, nll=None, mbs=None,
        abi_crowd=None, abi_uniform=None,
    )
    # Stub Aggregate is unused by pareto, but GridCell requires the field.
    acc_agg = _aggregate([], sampling_n=1, gt_map={})
    return GridCell(
        real_model=real_model, R=R, C=C,
        accuracy_aggregate=acc_agg,
        probabilistic_aggregate=prob_agg,
        n_eligible=10, n_total=10,
        mean_search_calls=cost,
        mean_latency_ms=None,
        parse_ok_rate=None, belief_parse_ok_rate=None,
        bi_ci_lo=None, bi_ci_hi=None,
        acc_ci_lo=None, acc_ci_hi=None,
    )


def test_pareto_frontier_strict_decrease_keeps_all() -> None:
    """4 cells with strictly increasing cost AND strictly increasing BI:
    every cell is non-dominated, so all 4 enter the frontier."""
    grid: dict[tuple[str, int, int], GridCell] = {
        ("m", 5, 1): _synthetic_cell(real_model="m", R=5, C=1, cost=1.0, bi=60.0),
        ("m", 5, 3): _synthetic_cell(real_model="m", R=5, C=3, cost=3.0, bi=70.0),
        ("m", 5, 5): _synthetic_cell(real_model="m", R=5, C=5, cost=5.0, bi=75.0),
        ("m", 5, 8): _synthetic_cell(real_model="m", R=5, C=8, cost=8.0, bi=76.0),
    }
    pareto = pareto_frontier(grid)
    assert len(pareto) == 4
    keys = [(c.R, c.C) for c in pareto]
    assert keys == [(5, 1), (5, 3), (5, 5), (5, 8)]


def test_pareto_frontier_drops_strictly_dominated_point() -> None:
    """Insert a (cost=8, BI=68) cell — dominated by (cost=5, BI=75) → dropped.
    The remaining frontier is the original 3 surviving cells."""
    grid: dict[tuple[str, int, int], GridCell] = {
        ("m", 5, 1): _synthetic_cell(real_model="m", R=5, C=1, cost=1.0, bi=60.0),
        ("m", 5, 3): _synthetic_cell(real_model="m", R=5, C=3, cost=3.0, bi=70.0),
        ("m", 5, 5): _synthetic_cell(real_model="m", R=5, C=5, cost=5.0, bi=75.0),
        ("m", 5, 8): _synthetic_cell(real_model="m", R=5, C=8, cost=8.0, bi=68.0),
    }
    pareto = pareto_frontier(grid)
    assert len(pareto) == 3
    keys = {(c.R, c.C) for c in pareto}
    assert keys == {(5, 1), (5, 3), (5, 5)}


def test_pareto_frontier_y_axis_nll_minimizes() -> None:
    """y_axis="nll_mean" — lower is better. Same dominance logic, opposite sign."""
    def cell_with_nll(C: int, cost: float, nll: float) -> GridCell:
        c = _synthetic_cell(real_model="m", R=5, C=C, cost=cost, bi=0.0)
        return replace(c, probabilistic_aggregate=replace(
            c.probabilistic_aggregate, nll=nll,
        ))

    grid = {
        ("m", 5, 1): cell_with_nll(1, 1.0, 1.0),
        ("m", 5, 3): cell_with_nll(3, 3.0, 0.5),  # better cost AND nll than C=8
        ("m", 5, 8): cell_with_nll(8, 8.0, 0.7),  # dominated by C=3
    }
    pareto = pareto_frontier(grid, y_axis="nll_mean")
    keys = {(c.R, c.C) for c in pareto}
    assert keys == {(5, 1), (5, 3)}


def test_pareto_frontier_rejects_unknown_axis() -> None:
    """An empty grid would short-circuit before the axis lookup runs, so
    we feed a single cell to force the axis-validation path."""
    grid = {
        ("m", 5, 1): _synthetic_cell(real_model="m", R=5, C=1, cost=1.0, bi=70.0),
    }
    with pytest.raises(ValueError, match="x_axis"):
        pareto_frontier(grid, x_axis="bogus")
    with pytest.raises(ValueError, match="y_axis"):
        pareto_frontier(grid, y_axis="bogus")


# --------------------------------------------------------------------------- #
# 10.5 paired_bootstrap_per_cell
# --------------------------------------------------------------------------- #


def test_paired_bootstrap_per_cell_returns_tuple_per_slug() -> None:
    samples_by_model, rows_by_model, gt_map, manifest_grid = _build_fixture()
    out = paired_bootstrap_per_cell(rows_by_model, n_bootstrap=200)
    assert len(out) == 8
    for slug in rows_by_model:
        triple = out[slug]
        assert triple is not None
        assert len(triple) == 3
        bi_mean, lo, hi = triple
        # CI band MUST contain (or at least bracket) the point estimate to
        # within float noise — the bootstrap mean of mean_BS converted to BI
        # is the same number as bi_mean by construction.
        assert lo <= bi_mean <= hi or abs(lo - hi) < 1e-9


def test_paired_bootstrap_per_cell_handles_empty_slug() -> None:
    out = paired_bootstrap_per_cell({"slug": []}, n_bootstrap=10)
    assert out["slug"] is None


# --------------------------------------------------------------------------- #
# 10.6 Winrate matrix
# --------------------------------------------------------------------------- #


def test_winrate_matrix_pair_count() -> None:
    """2 real_models → 1 pair (m_a vs m_b)."""
    samples_by_model, rows_by_model, gt_map, manifest_grid = _build_fixture()
    grid = build_grid_summary(
        samples_by_model, gt_map, rows_by_model, manifest_grid,
    )
    # Use a small n_bootstrap so the test stays fast.
    rows = winrate_matrix(grid, rows_by_model, n_bootstrap=200)
    assert len(rows) == 1
    r = rows[0]
    assert r.model_a == "m_a"
    assert r.model_b == "m_b"
    # m_a is constructed to be more confident on every cell, so it should
    # win every (R, C) cell. wins_a == total_cells, wins_b == 0, ties == 0.
    assert r.total_cells == 4  # 2 R × 2 C
    assert r.wins_a == 4
    assert r.wins_b == 0
    assert r.ties == 0
    # sig_cells_a + sig_cells_b ≤ total_cells (mutually exclusive)
    assert r.sig_cells_a + r.sig_cells_b <= r.total_cells


def test_winrate_matrix_complementary_under_role_swap() -> None:
    """Symmetric property: if we synthesize a 2-model grid where model_a has
    BI < model_b on every cell (i.e. a always loses), winrate row's wins_a
    must be 0 and wins_b must equal total_cells."""
    grid: dict[tuple[str, int, int], GridCell] = {}
    rows_by_model: dict[str, list[_QuestionProbabilityRow]] = {}
    for R in [5]:
        for C in [1, 3]:
            cell_a = _synthetic_cell(
                real_model="loser", R=R, C=C, cost=1.0, bi=10.0,
            )
            cell_b = _synthetic_cell(
                real_model="winner", R=R, C=C, cost=1.0, bi=70.0,
            )
            grid[("loser", R, C)] = cell_a
            grid[("winner", R, C)] = cell_b
            rows_by_model[dbmod.compose_virtual_slug("loser", R, C)] = []
            rows_by_model[dbmod.compose_virtual_slug("winner", R, C)] = []
    rows = winrate_matrix(grid, rows_by_model, n_bootstrap=50)
    assert len(rows) == 1
    r = rows[0]
    assert r.model_a == "loser"
    assert r.model_b == "winner"
    assert r.wins_a == 0
    assert r.wins_b == 2
    assert r.ties == 0


# --------------------------------------------------------------------------- #
# 10.7 Legacy v4 compat
# --------------------------------------------------------------------------- #


def test_run_grid_analysis_returns_empty_on_legacy_manifest(tmp_path: Path) -> None:
    """Manifest without a `grid` segment → run_grid_analysis returns []
    and writes nothing to disk. Legacy v4 single-cell runs stay
    byte-identical to pre-change behavior."""
    run_dir = tmp_path / "legacy_run"
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    legacy_manifest = {
        "run_id": "legacy",
        "models": ["openai/gpt-5"],
        "model_files": {"openai/gpt-5": "openai__gpt-5.db"},
        "sampling_n": 1,
        # No "grid" key — this is the v4 backward-compat path.
    }
    written = run_grid_analysis(
        run_dir=run_dir,
        manifest=legacy_manifest,
        samples_by_model={},
        gt_map_global={},
        rows_by_model={},
        analysis_dir=analysis_dir,
    )
    assert written == []
    # No grid_*.csv MUST be written.
    assert list(analysis_dir.glob("grid_*.csv")) == []


def test_run_grid_analysis_writes_5_csvs_with_grid_segment(tmp_path: Path) -> None:
    """Manifest with grid segment → 5 grid_*.csv files exist."""
    run_dir = tmp_path / "grid_run"
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    samples_by_model, rows_by_model, gt_map, manifest_grid = _build_fixture()
    manifest = {
        "run_id": "grid",
        "models": list(samples_by_model.keys()),
        "model_files": {s: f"{s}.db" for s in samples_by_model.keys()},
        "sampling_n": 1,
        "grid": manifest_grid,
    }
    written = run_grid_analysis(
        run_dir=run_dir,
        manifest=manifest,
        samples_by_model=samples_by_model,
        gt_map_global=gt_map,
        rows_by_model=rows_by_model,
        analysis_dir=analysis_dir,
    )
    names = sorted(p.name for p in written)
    assert names == [
        "grid_marginal_C.csv",
        "grid_marginal_R.csv",
        "grid_pareto.csv",
        "grid_summary.csv",
        "grid_winrate.csv",
    ]
    # grid_summary.csv has 17 columns + 8 data rows
    summary_text = (analysis_dir / "grid_summary.csv").read_text(encoding="utf-8")
    lines = summary_text.strip().split("\n")
    assert len(lines) == 9  # header + 8 cells
    header_cols = lines[0].split(",")
    assert len(header_cols) == 17
    assert header_cols[0] == "real_model"
    assert header_cols[-1] == "belief_parse_ok_rate"


# --------------------------------------------------------------------------- #
# 10.8 Reflection A/B isolation under grid run
# --------------------------------------------------------------------------- #


def _build_minimal_run_for_pairing(
    base: Path,
    *,
    run_id: str,
    model: str,
    reflection_hash: str | None,
    belief_hash: str = "bel_v1",
) -> Path:
    """Drop a manifest + run_meta DB into `base/run_id/` so find_paired_runs
    sees this run. Schema mirrors `tests/test_behavior.py::_build_minimal_run`
    minimally — we only need `run_meta` + `questions` for the bucketing path
    and the `_question_ids_in` join. No `run_results` / `belief_*` columns
    needed because find_paired_runs short-circuits before touching them.
    """
    run_dir = base / run_id
    (run_dir / "db").mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({
            "run_id": run_id,
            "models": [model],
            "model_files": {model: f"{dbmod.model_slug_safe(model)}.db"},
            "sampling_n": 1,
        }),
        encoding="utf-8",
    )
    db_path = run_dir / "db" / f"{dbmod.model_slug_safe(model)}.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
    CREATE TABLE run_meta (
        run_id TEXT,
        model TEXT,
        sampling_n INTEGER,
        source_db_hash TEXT,
        metadata_hash TEXT,
        prompt_templates_hash TEXT,
        reflection_protocol_hash TEXT,
        belief_protocol_hash TEXT,
        started_at TEXT
    );
    CREATE TABLE questions (
        id TEXT PRIMARY KEY,
        question_type TEXT,
        choice_type TEXT,
        options TEXT,
        answer TEXT
    );
    """)
    conn.execute(
        "INSERT INTO run_meta VALUES (?, ?, 1, 'srchash', 'mdhash', 'pthash', ?, ?, ?)",
        (run_id, model, reflection_hash, belief_hash, "2026-04-25T12:00:00Z"),
    )
    for qid in ["q1", "q2"]:
        conn.execute(
            "INSERT INTO questions VALUES (?, 'yes_no', 'single', ?, 'A')",
            (qid, json.dumps(["yes", "no"])),
        )
    conn.commit()
    conn.close()
    return run_dir


def test_find_paired_runs_does_not_pair_across_grid_cells(tmp_path: Path) -> None:
    """Different (R, C) cells encode as different virtual slugs. Since
    `find_paired_runs` buckets by `model` (== virtual slug), two cells
    with different (R, C) MUST land in different buckets and never pair —
    even if everything else (real_model, hashes) matches and one has
    reflection on while the other has reflection off."""
    runs_root = tmp_path / "runs"
    # cell_A: m_x with R=5, C=3, reflection on
    _build_minimal_run_for_pairing(
        runs_root, run_id="run_a_on",
        model=dbmod.compose_virtual_slug("m_x", 5, 3),
        reflection_hash="refl_v1",
    )
    # cell_B: m_x with R=10, C=3, reflection off
    _build_minimal_run_for_pairing(
        runs_root, run_id="run_b_off",
        model=dbmod.compose_virtual_slug("m_x", 10, 3),
        reflection_hash=None,
    )
    pairs = find_paired_runs(runs_root)
    # Different virtual slugs → different bucket keys → no accidental pair.
    assert pairs == []


def test_find_paired_runs_pairs_within_same_grid_cell(tmp_path: Path) -> None:
    """Counter-test: SAME (R, C) cell with reflection on/off MUST pair.
    Demonstrates the bucket-key isolation cuts both ways — grid runs don't
    block legitimate within-cell A/B pairs."""
    runs_root = tmp_path / "runs"
    same_cell_slug = dbmod.compose_virtual_slug("m_x", 5, 3)
    _build_minimal_run_for_pairing(
        runs_root, run_id="run_on",
        model=same_cell_slug,
        reflection_hash="refl_v1",
    )
    _build_minimal_run_for_pairing(
        runs_root, run_id="run_off",
        model=same_cell_slug,
        reflection_hash=None,
    )
    pairs = find_paired_runs(runs_root)
    assert len(pairs) == 1
    assert pairs[0].model == same_cell_slug
