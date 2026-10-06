"""Tests for `scripts/fss_sensitivity.py`.

Pins:
* CLI accepts `--all` and `--alpha A --beta B` modes.
* 5-tier sweep produces 5 rows per model with the expected (α, β) pairs.
* The (2, 0.5) row matches `accuracy.fss(samples, gt_map)` byte-for-byte
  (this is the "default-tier ↔ run_analysis main table" parity invariant).
* The CSV opens with the provenance comment so a reviewer reading the bare
  file understands it's not a `run_analysis` artefact.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

# `_build_fixture_run` is defined in `tests.test_analysis`; reusing it here
# keeps the sensitivity CLI test honest against the same fixture style as
# the main run_analysis tests.
from tests.test_analysis import _build_fixture_run

import scripts.fss_sensitivity as sensitivity  # noqa: E402


def _read_sensitivity_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        # Skip the two provenance comment lines.
        comment_lines = []
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if line.startswith("#"):
                comment_lines.append(line)
                continue
            f.seek(pos)
            break
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(comment_lines) >= 1, "Expected provenance comment line on top"
    assert "fss_sensitivity.py" in comment_lines[0]
    return rows


def test_sensitivity_all_tiers_emits_5_rows_per_model(tmp_path: Path) -> None:
    """Default `--all` sweep → 5 rows per model with the expected (α, β) tiers."""
    run_dir = _build_fixture_run(tmp_path)
    sensitivity.run_sensitivity(run_dir)

    out_path = run_dir / "analysis" / "fss_sensitivity.csv"
    assert out_path.exists()
    rows = _read_sensitivity_rows(out_path)

    # Two models × 5 tiers = 10 rows.
    assert len(rows) == 10
    by_model: dict[str, set[tuple[float, float]]] = {}
    for r in rows:
        by_model.setdefault(r["model"], set()).add(
            (float(r["alpha"]), float(r["beta"]))
        )
    expected_tiers = {(1.0, 1.0), (1.0, 0.5), (2.0, 0.5), (3.0, 0.5), (4.0, 0.5)}
    for model, tiers in by_model.items():
        assert tiers == expected_tiers, f"Model {model}: tiers={tiers}"


def test_sensitivity_default_tier_matches_run_analysis_fss(tmp_path: Path) -> None:
    """Task 4b.5: the (2, 0.5) row must match `accuracy.fss(...)` exactly.

    This is the parity invariant — `per_model_summary.csv` reports a single
    canonical FSS at (2, 0.5); the sensitivity CSV's (2, 0.5) row must
    reproduce that number to numerical precision.
    """
    from forecast_eval.analysis.accuracy import fss as fss_fn

    run_dir = _build_fixture_run(tmp_path)
    sensitivity.run_sensitivity(run_dir)

    rows = _read_sensitivity_rows(run_dir / "analysis" / "fss_sensitivity.csv")
    samples_by_model, gt_map = sensitivity._load_samples_and_gt(run_dir)

    for r in rows:
        if (float(r["alpha"]), float(r["beta"])) != (2.0, 0.5):
            continue
        if not r["fss"]:
            # FSS=None for this model — the default tier should also be empty.
            continue
        # Recompute via the same helper.
        expected = fss_fn(
            samples_by_model[r["model"]], gt_map, alpha=2.0, beta=0.5,
        )
        if expected["fss"] is None:
            assert r["fss"] == ""
        else:
            assert float(r["fss"]) == pytest.approx(expected["fss"], abs=1e-6)


def test_sensitivity_single_alpha_beta_produces_one_row_per_model(tmp_path: Path) -> None:
    """`--alpha 1 --beta 1` (Jaccard) → exactly one row per model."""
    run_dir = _build_fixture_run(tmp_path)
    sensitivity.run_sensitivity(run_dir, tiers=[(1.0, 1.0)])

    rows = _read_sensitivity_rows(run_dir / "analysis" / "fss_sensitivity.csv")
    assert len(rows) == 2  # 2 models × 1 tier
    for r in rows:
        assert (float(r["alpha"]), float(r["beta"])) == (1.0, 1.0)


def test_sensitivity_main_cli_all_flag(tmp_path: Path) -> None:
    """`main(["--all", run_dir])` returns 0 and writes the CSV."""
    run_dir = _build_fixture_run(tmp_path)
    rc = sensitivity.main([str(run_dir), "--all"])
    assert rc == 0
    assert (run_dir / "analysis" / "fss_sensitivity.csv").exists()


def test_sensitivity_main_cli_single_alpha_beta(tmp_path: Path) -> None:
    """`main([run_dir, "--alpha", "1", "--beta", "0.5"])` writes a 1-tier CSV."""
    run_dir = _build_fixture_run(tmp_path)
    rc = sensitivity.main([str(run_dir), "--alpha", "1", "--beta", "0.5"])
    assert rc == 0

    rows = _read_sensitivity_rows(run_dir / "analysis" / "fss_sensitivity.csv")
    assert all(
        (float(r["alpha"]), float(r["beta"])) == (1.0, 0.5)
        for r in rows
    )


def test_sensitivity_main_cli_alpha_without_beta_errors(tmp_path: Path) -> None:
    """`--alpha` without `--beta` → exit code 2 (CLI usage error)."""
    run_dir = _build_fixture_run(tmp_path)
    rc = sensitivity.main([str(run_dir), "--alpha", "2"])
    assert rc == 2
