"""Build a virtual evaluation panel by aggregating model DBs across source runs.

Usage:
    python scripts/build_panel_analysis.py \
        --panel panels/<panel_id>.json \
        --out runs/panel_<panel_id> \
        --mode copy

Output: a directory whose shape matches a single-run output (manifest.json +
db/<slug>.db x N). `forecast_eval.analysis` can consume it directly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_source_manifest(run_dir: Path) -> dict[str, Any]:
    mf = run_dir / "manifest.json"
    if not mf.exists():
        raise FileNotFoundError(f"source manifest missing: {mf}")
    return json.loads(mf.read_text())


def _collect_training_cutoffs(panel: dict[str, Any], repo_root: Path) -> dict[str, str]:
    """For each panel model, resolve the training cutoff.

    Resolution order: (1) source run manifest under the full virtual_slug,
    (2) source run manifest under the base slug (before `::`), (3) the
    panel manifest's per-model `training_cutoff` field. The third tier
    exists because some early runs persisted with empty cutoff maps; the
    panel manifest is the auditable source of record for them.
    """
    out: dict[str, str] = {}
    for m in panel["models"]:
        run_mf = _load_source_manifest(repo_root / "runs" / m["source_run_id"])
        cutoffs = run_mf.get("model_training_cutoffs", {})
        slug = m["virtual_slug"]
        if slug in cutoffs:
            out[slug] = cutoffs[slug]
            continue
        base = slug.split("::")[0]
        if base in cutoffs:
            out[slug] = cutoffs[base]
            continue
        panel_cutoff = m.get("training_cutoff")
        if panel_cutoff:
            out[slug] = panel_cutoff
            continue
        raise KeyError(
            f"training cutoff not found for {slug!r} in run "
            f"{m['source_run_id']!r} and panel manifest has no fallback"
        )
    return out


def _collect_source_run_hashes(panel: dict[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    """For each unique source_run_id, record (run_id, source_db_hash, dataset_hash, prompt_templates_hash)."""
    seen: dict[str, dict[str, Any]] = {}
    for m in panel["models"]:
        run_id = m["source_run_id"]
        if run_id in seen:
            continue
        run_mf = _load_source_manifest(repo_root / "runs" / run_id)
        hashes = run_mf.get("hashes", {})
        seen[run_id] = {
            "run_id": run_id,
            "source_db_hash": hashes.get("source_db"),
            "dataset_hash": hashes.get("metadata"),
            "prompt_templates_hash": hashes.get("prompt_templates"),
        }
    return list(seen.values())


def build_panel(panel_path: Path, out_dir: Path, mode: str = "copy", repo_root: Path | None = None) -> Path:
    """Build the virtual panel directory.

    Args:
        panel_path: panel manifest JSON (panels/<panel_id>.json).
        out_dir: target directory (created if missing).
        mode: "copy" (physical copies) or "symlink".
        repo_root: location of runs/. Defaults to current working directory,
            which is the repo root when invoked via the CLI from the project
            root.

    Returns:
        out_dir (so callers can chain).
    """
    panel = json.loads(panel_path.read_text())
    if repo_root is None:
        repo_root = Path.cwd()

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "db").mkdir(exist_ok=True)
    (out_dir / "analysis").mkdir(exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    model_files: dict[str, str] = {}
    db_hashes: dict[str, str] = {}
    for m in panel["models"]:
        src = repo_root / "runs" / m["source_run_id"] / "db" / m["db_filename"]
        if not src.exists():
            raise FileNotFoundError(f"source DB missing: {src}")
        dst = out_dir / "db" / m["db_filename"]
        if mode == "copy":
            shutil.copy2(src, dst)
        elif mode == "symlink":
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())
        else:
            raise ValueError(f"unknown mode: {mode!r}")
        model_files[m["virtual_slug"]] = m["db_filename"]
        db_hashes[m["virtual_slug"]] = _sha256_of_file(src)

    cutoffs = _collect_training_cutoffs(panel, repo_root)
    panel_source_runs = _collect_source_run_hashes(panel, repo_root)

    manifest = {
        "panel_id": panel["panel_id"],
        "schema_version": 1,
        "is_virtual_panel": True,
        "models": [m["virtual_slug"] for m in panel["models"]],
        "model_files": model_files,
        "model_training_cutoffs": cutoffs,
        "panel_source_runs": panel_source_runs,
        "panel_db_hashes": db_hashes,
        "composite_weights": panel["composite_weights"],
        "bootstrap_seed": panel["bootstrap_seed"],
        "bootstrap_iterations": panel["bootstrap_iterations"],
        "ci_alpha": panel["ci_alpha"],
        "families": panel.get("families", {}),
        "model_family_map": {m["virtual_slug"]: m["family"] for m in panel["models"]},
        "sampling_n": 3,
        "analysis_schema": "panel-v1",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return out_dir


def _cli() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--mode", choices=("copy", "symlink"), default="copy")
    args = p.parse_args()
    build_panel(args.panel, args.out, args.mode)
    print(f"panel built: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
