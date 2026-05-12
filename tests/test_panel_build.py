"""Pin tests for panel manifest schema and uniqueness.

Panel manifests live in `panels/<panel_id>.json` and are git-ignored
because each user may bundle a different evaluation slate. These tests
discover a manifest at runtime; when no manifest is present (e.g. on a
fresh clone) the tests are skipped instead of failing.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
PANELS_DIR = REPO_ROOT / "panels"


def _discover_panel() -> Path | None:
    """Prefer any evaluator-supplied panel; fall back to panels/example.json.

    Returns the lexicographically first non-example manifest if one exists,
    otherwise panels/example.json if present, otherwise None (tests skip).
    """
    if not PANELS_DIR.exists():
        return None
    all_jsons = sorted(PANELS_DIR.glob("*.json"))
    non_example = [p for p in all_jsons if p.name != "example.json"]
    if non_example:
        return non_example[0]
    example = PANELS_DIR / "example.json"
    return example if example.exists() else None


@pytest.fixture(scope="module")
def panel_info():
    """Return (manifest_dict, is_example). is_example=True means the
    only manifest we found is panels/example.json — whose `source_run_id`
    entries are placeholders and therefore can't be resolved against a
    fresh clone's `runs/` directory.
    """
    path = _discover_panel()
    if path is None:
        pytest.skip("no panel manifest found under panels/")
    return json.loads(path.read_text()), path.name == "example.json"


@pytest.fixture(scope="module")
def panel(panel_info):
    return panel_info[0]


@pytest.fixture(scope="module")
def real_panel(panel_info):
    panel, is_example = panel_info
    if is_example:
        pytest.skip("panels/example.json has placeholder source_run_id; skipping run-bound assertions")
    return panel


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
    for m in real_panel["models"]:
        path = REPO_ROOT / "runs" / m["source_run_id"] / "db" / m["db_filename"]
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
    subset = {**real_panel, "models": real_panel["models"][:2]}
    subset_path = tmp_path / "panel_subset.json"
    subset_path.write_text(json.dumps(subset))

    out_dir = tmp_path / "panel_out"
    cmd = [
        sys.executable, "scripts/build_panel_analysis.py",
        "--panel", str(subset_path),
        "--out", str(out_dir),
        "--mode", "copy",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
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
