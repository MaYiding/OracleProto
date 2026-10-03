"""Freeze five private collection assignments after the coordinator releases them."""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger
from forecast_eval.config import Settings
from forecast_eval.runner import generate_run_id
from scripts.prepare_collection import digest, local
from scripts.run_handoff import MODELS, PACKAGE, verify


def save(path: Path, value: dict) -> None:
    local(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", action="store_true", help="Seal the coordinator's release into an existing package")
    args = parser.parse_args()
    source_dir = local(ROOT / "runs/collection_300")
    release_path = local(source_dir / "handoff_release.json")
    release = json.loads(release_path.read_text())
    if set(release.get("models", [])) != set(MODELS):
        raise ValueError("coordinator assignment must name all five models")
    if args.release:
        if release.get("status") != "released":
            raise ValueError("wait for coordinator release before allowing execution")
        manifest = verify()
        shutil.copyfile(release_path, local(PACKAGE / "release.json"))
        manifest["files"]["release.json"] = digest(local(PACKAGE / "release.json"))
        save(PACKAGE / "manifest.json", manifest)
        verify(require_release=True)
        logger.info("Coordinator release verified; fixed run IDs preserved")
        return
    if local(PACKAGE).exists():
        raise ValueError("handoff already exists; preserve its run IDs and immutable inputs")
    master = json.loads(local(source_dir / "plan.json").read_text())
    continuation = json.loads(local(source_dir / "continuation.json").read_text())
    inventory = json.loads(local(source_dir / "inventory.json").read_text())
    probes = json.loads(local(ROOT / "logs/collection_300/profile_probes.json").read_text())
    PACKAGE.mkdir()
    local(PACKAGE / "reference_db").mkdir()
    local(PACKAGE / "jobs").mkdir()
    shutil.copyfile(local(source_dir / "source.db"), local(PACKAGE / "source.db"))
    shutil.copyfile(release_path, local(PACKAGE / "release.json"))
    if digest(local(PACKAGE / "source.db")) != inventory["source_sha256"]:
        raise ValueError("frozen source checksum mismatch")
    manifest = {"anchor_ids": inventory["anchor_ids"], "additional_ids": inventory["additional_ids"],
                "jobs": {}, "files": {}, "code_files": {}, "reference_audit": {}}
    for model in MODELS:
        name = model + "--provider-default"
        active_db = local(ROOT / "runs" / continuation["run_id"] / "db" / (name + "__r5__c4.db"))
        if active_db.exists():
            with sqlite3.connect(active_db.as_uri() + "?mode=ro", uri=True) as conn:
                if conn.execute("SELECT count(*) FROM request_events").fetchone()[0]:
                    raise ValueError("assigned model already has request evidence; export its progress before delegation")
        profile = master["runtime"]["MODEL_PROFILES"][name]
        if not probes[name]["ok"] or probes[name]["profile"] != profile:
            raise ValueError("profile capability evidence mismatch: " + model)
        lineage = dict(master["reference_lineage"][name])
        source = local(ROOT / lineage["reference_db"])
        target = local(PACKAGE / "reference_db" / source.name)
        shutil.copyfile(source, target)
        if digest(target) != lineage["sha256"]:
            raise ValueError("reference checksum mismatch")
        with sqlite3.connect(target.as_uri() + "?mode=ro", uri=True) as conn:
            ids = {row[0] for row in conn.execute("SELECT id FROM questions")}
            completed = sum(conn.execute(f"SELECT count(*) FROM run_results WHERE s{i}_created_at IS NOT NULL AND s{i}_error IS NULL").fetchone()[0] for i in range(3))
        if ids != set(inventory["anchor_ids"]) or completed != 240:
            raise ValueError("reference does not contain all 80 x 3 completed observations")
        manifest["reference_audit"][model] = {"questions": len(ids), "completed_samples": completed}
        lineage["reference_db"] = str(target.relative_to(ROOT))
        runtime = dict(continuation["runtime"])
        runtime.update(MODELS=[name], MODEL_PROFILES={name: profile},
                       MODEL_QUESTION_IDS={name: inventory["additional_ids"]},
                       MODEL_TRAINING_CUTOFFS={name: master["runtime"]["MODEL_TRAINING_CUTOFFS"][name]},
                       COLLECTION_REFERENCE_DBS={}, MODEL_SAMPLE_INDICES={}, COLLECTION_MODEL="",
                       COLLECTION_SAMPLE_LIMIT=0, RUN_ID=generate_run_id(),
                       SOURCE_DB=str((PACKAGE / "source.db").relative_to(ROOT)),
                       LLM_BASE_URL="https://aihubmix.com/v1", LLM_MAX_CONCURRENCY=1,
                       PROMPT_TEMPLATE_STYLE="shared")
        # Explicit defaults prevent the recipient's .env from changing the protocol.
        frozen = Settings(**runtime, LEAK_DETECTOR_API_KEY="handoff-offline-validation").model_dump(mode="json")
        for key in list(frozen):
            if key.endswith(("_API_KEY", "_BASE_URL")):
                frozen.pop(key)
        frozen["MODEL_PROFILES"] = {name: profile}
        plan = {"phase": "continuation", "run_id": frozen["RUN_ID"], "runtime": frozen,
                "source_sha256": inventory["source_sha256"], "new_sample_count": 660,
                "search_call_ceiling": 2640, "reference_lineage": {name: lineage},
                "reference_db_hashes": {lineage["reference_db"]: lineage["sha256"]}, "status": "prepared"}
        relative = "jobs/" + model + ".json"
        save(PACKAGE / relative, plan)
        manifest["jobs"][model] = relative
    run_ids = [json.loads(local(PACKAGE / p).read_text())["run_id"] for p in manifest["jobs"].values()]
    for relative in manifest["jobs"].values():
        path = local(PACKAGE / relative)
        plan = json.loads(path.read_text())
        plan["prior_run_ids"] = run_ids
        save(path, plan)
    save(PACKAGE / "profile_probes.json", {m + "--provider-default": probes[m + "--provider-default"] for m in MODELS})
    paths = [local(ROOT / name) for name in ("evaluation.py", "pyproject.toml", "environment.yml",
             "scripts/run_handoff.py", "scripts/prepare_handoff.py", "scripts/collect_forecast_panel.py",
             "scripts/prepare_collection.py", "scripts/probe_collection.py", "scripts/catalog_collection.py")]
    paths += [local(p) for p in local(ROOT / "forecast_eval").glob("*.py")]
    manifest["code_files"] = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    inputs = [PACKAGE / "source.db", PACKAGE / "release.json", PACKAGE / "profile_probes.json"]
    inputs += list((PACKAGE / "jobs").glob("*.json")) + list((PACKAGE / "reference_db").glob("*.db"))
    manifest["files"] = {str(p.relative_to(PACKAGE)): digest(local(p)) for p in inputs}
    save(PACKAGE / "manifest.json", manifest)
    verify(require_release=release.get("status") == "released")
    logger.info("Prepared {} fixed assignments with 1200 reference and 3300 pending observations", len(MODELS))


if __name__ == "__main__":
    main()
