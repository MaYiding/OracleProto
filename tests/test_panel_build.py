"""Pin tests for the public panel schema and independently created source runs."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
PANELS_DIR = REPO_ROOT / "panels"


@pytest.fixture(scope="module")
def panel():
    return json.loads((PANELS_DIR / "example.json").read_text())


@pytest.fixture
def real_panel(tmp_path, panel):
    root = tmp_path / "workspace"
    for model in panel["models"]:
        run = root / "runs" / model["source_run_id"]
        (run / "db").mkdir(parents=True)
        with sqlite3.connect(run / "db" / model["db_filename"]) as conn:
            conn.execute("CREATE TABLE identity (model TEXT)")
            conn.execute("INSERT INTO identity VALUES (?)", (model["virtual_slug"],))
        manifest = {
            "model_training_cutoffs": {model["virtual_slug"]: model["training_cutoff"]},
            "hashes": {"source_db": "dataset", "metadata": "metadata", "prompt_templates": "prompts"},
        }
        (run / "manifest.json").write_text(json.dumps(manifest))
    return panel, root


def test_panel_keys(panel):
    for key in (
        "panel_id", "bootstrap_seed", "bootstrap_iterations",
        "ci_alpha", "composite_weights", "families", "models",
    ):
        assert key in panel, f"missing top-level key {key!r}"


def test_panel_models_nonempty_and_unique(panel):
    models = panel["models"]
    assert len(models) >= 2, "panel must have at least 2 models for pairwise tests"
    slugs = [m["virtual_slug"] for m in models]
    labels = [m["display_label"] for m in models]
    assert len(set(slugs)) == len(models), "virtual_slug not unique"
    assert len(set(labels)) == len(models), "display_label not unique"


def test_panel_model_fields(panel):
    for m in panel["models"]:
        for key in ("display_label", "virtual_slug", "source_run_id", "db_filename", "family"):
            assert key in m, f"model {m.get('display_label')!r} missing {key!r}"


def test_panel_source_dbs_exist(real_panel):
    panel, root = real_panel
    for m in panel["models"]:
        path = root / "runs" / m["source_run_id"] / "db" / m["db_filename"]
        assert path.exists(), f"source DB missing: {path}"


def test_panel_composite_weights_sum_to_one(panel):
    w = panel["composite_weights"]
    assert set(w.keys()) == {"binary", "mc-single", "mc-multi"}
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_panel_seed_pins(panel):
    assert isinstance(panel["bootstrap_seed"], int)
    assert panel["bootstrap_iterations"] >= 1000
    assert 0.0 < panel["ci_alpha"] < 1.0


def test_build_panel_creates_virtual_dir(tmp_path, real_panel):
    """Run build_panel_analysis.py against a 2-model subset and check the
    output directory has the expected shape."""
    panel, root = real_panel
    subset = {**panel, "models": panel["models"][:2]}
    subset_path = tmp_path / "panel_subset.json"
    subset_path.write_text(json.dumps(subset))

    out_dir = tmp_path / "panel_out"
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "build_panel_analysis.py"),
        "--panel", str(subset_path),
        "--out", str(out_dir),
        "--mode", "copy",
    ]
    result = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, f"build failed: {result.stderr}"

    assert (out_dir / "manifest.json").exists()
    db_files = list((out_dir / "db").glob("*.db"))
    assert len(db_files) == 2

    manifest = json.loads((out_dir / "manifest.json").read_text())
    for key in (
        "panel_id", "models", "model_files", "model_training_cutoffs",
        "panel_source_runs", "composite_weights", "bootstrap_seed",
    ):
        assert key in manifest, f"manifest missing {key!r}"
    assert len(manifest["panel_source_runs"]) >= 1
    for entry in manifest["panel_source_runs"]:
        assert "run_id" in entry and "source_db_hash" in entry
    for model in subset["models"]:
        source = root / "runs" / model["source_run_id"] / "db" / model["db_filename"]
        target = out_dir / "db" / model["db_filename"]
        assert target.read_bytes() == source.read_bytes()
        assert not target.is_symlink()
        assert manifest["panel_db_hashes"][model["virtual_slug"]] == hashlib.sha256(source.read_bytes()).hexdigest()
