"""Index raw collection strata and normalize sample fields without scoring."""
from __future__ import annotations

import hashlib
import gzip
import json
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger

from forecast_eval import db
from scripts.prepare_collection import FIELDS, digest, local, read_db


def sample_coverage(plan: dict, question_ids: set[str], successes: dict, failures: dict) -> dict:
    runtime = plan["runtime"]
    profiles = {}
    for name in runtime["MODELS"]:
        scope = set(runtime.get("MODEL_QUESTION_IDS", {}).get(name, question_ids))
        if not scope <= question_ids:
            raise ValueError("collection plan references questions outside the fixed corpus")
        selection = runtime.get("MODEL_SAMPLE_INDICES", {}).get(name)
        expected = {(name, qid, index) for qid in scope
                    for index in (selection.get(qid, []) if selection is not None else range(runtime["SAMPLING_N"]))}
        pending = []
        for identity in sorted(expected - successes.keys()):
            errors = sorted(failures.get(identity, set()))
            if errors == ["content_policy"] and runtime.get("COLLECTION_RETAIN_MODEL_REFUSALS", False):
                state = "refused"
            else:
                state = "excluded" if errors == ["skipped_training_cutoff"] else "failed" if errors else "missing"
            pending.append({"question_id": identity[1], "sample_idx": identity[2], "state": state, "errors": errors})
        profiles[name] = {"expected": len(expected), "collected": len(expected & successes.keys()),
                          **{state: sum(item["state"] == state for item in pending)
                             for state in ("failed", "missing", "excluded", "refused")},
                          "pending_slots": [item for item in pending if item["state"] != "refused"],
                          "refused_slots": [item for item in pending if item["state"] == "refused"]}
    totals = {key: sum(profile[key] for profile in profiles.values())
              for key in ("expected", "collected", "failed", "missing", "excluded", "refused")}
    if totals["expected"] != plan["new_sample_count"]:
        raise ValueError("collection plan sample count differs from its selected slots")
    return {"phase": plan["phase"], **totals, "complete": totals["expected"] == totals["collected"],
            "attempts_complete": totals["expected"] == totals["collected"] + totals["refused"], "profiles": profiles}


def observed_samples(conn: sqlite3.Connection, source: dict, questions: dict):
    """Expose one source row per recorded sample, retaining every raw field."""
    for row in conn.execute(f"SELECT {FIELDS} FROM questions"):
        actual = dict(row)
        if actual != questions.get(row["id"]):
            raise ValueError(f"source question differs from fixed corpus: {source['path']} / {row['id']}")
    for row in conn.execute("SELECT * FROM run_results ORDER BY question_id"):
        for sample_idx in range(source["metadata"]["sampling_n"]):
            prefix = f"s{sample_idx}_"
            if not row[prefix + "created_at"]:
                continue
            result = {key[len(prefix):]: row[key] for key in row.keys() if key.startswith(prefix)}
            if source["cohort"] != "reference" and result["error"] is None:
                if result["correct"] is not None:
                    raise ValueError("collection unexpectedly contains scored answers")
                if not result["messages_trace"]:
                    raise ValueError("completed collection sample has no message trace")
                searches = json.loads(result["search_calls"] or "[]")
                config = json.loads(source["metadata"]["config_snapshot"])
                cutoff = (date.fromisoformat(questions[row["question_id"]]["end_time"]) +
                          timedelta(days=config["TAVILY_END_DATE_OFFSET_DAYS"])).isoformat()
                if len(searches) > config["REACT_MAX_SEARCH_CALLS"] or result["react_steps"] > config["REACT_MAX_STEPS"]:
                    raise ValueError("sample exceeds the declared collection budget")
                for search in searches:
                    if not all(key in search for key in ("raw_response", "results_raw", "detector_reasons", "visible_payload")):
                        raise ValueError("completed collection sample has incomplete search evidence")
                    if len(search["detector_reasons"]) != len(search["results_raw"]):
                        raise ValueError("detector reasons do not align with raw result items")
                    if search["end_date"] != cutoff:
                        raise ValueError("search cutoff differs from the declared time boundary")
                    kept = []
                    for item, verdict in zip(search["results_raw"], search["detector_verdicts"], strict=True):
                        refusal_dropped = config.get("LEAK_DETECTOR_DROP_CONTENT_POLICY", False) and verdict.split(":", 2)[:2] == ["failed", "content_policy"]
                        if verdict not in ("keep", "drop") and not refusal_dropped:
                            raise ValueError("completed sample includes a failed detector verdict")
                        if verdict == "keep":
                            kept.append({key: value for key, value in item.items() if value is not None})
                    if kept != search["visible_payload"]["results"]:
                        raise ValueError("model-visible payload differs from the kept items")
                actual_tools = [json.loads(message["content"]) for message in json.loads(result["messages_trace"])
                                if message["role"] == "tool"]
                actual_searches = [payload for payload in actual_tools if "error" not in payload]
                if actual_searches != [search["visible_payload"] for search in searches]:
                    raise ValueError("model context differs from the declared filtered search payloads")
            yield {"source_id": source["source_id"], "profile_id": source["profile_id"],
                   "question_id": row["question_id"], "sample_idx": sample_idx,
                   "cohort": source["cohort"], "user_prompt": row["user_prompt"], "result": result}


def build_catalog() -> dict:
    collection = local(ROOT / "runs/collection_300")
    inventory = json.loads(local(collection / "inventory.json").read_text())
    conn = read_db(local(ROOT / inventory["source_db"]))
    questions = {row["id"]: dict(row) for row in conn.execute(f"SELECT {FIELDS} FROM test_cases")}
    conn.close()
    if digest(local(ROOT / inventory["source_db"])) != inventory["source_sha256"]:
        raise ValueError("fixed source hash mismatch")
    candidates = [{"path": item["copy_path"], "expected_sha256": item["copy_sha256"],
                   "profile_id": item["model"].split("::")[0] + "--provider-default", "cohort": "reference"}
                  for item in inventory["reference_inventory"]]
    phases = []
    plans = []
    repairs = []
    for phase in ("continuation", "reasoning", "repair"):
        path = local(collection / (phase + ".json"))
        if not path.exists():
            continue
        plan = json.loads(path.read_text())
        plans.append(plan)
        if phase == "repair":
            repairs = [{**item, "repair_run_id": plan["run_id"]} for item in plan.get("repair_sources", [])]
        phases.append({"phase": phase, "run_id": plan["run_id"], "status": plan["status"],
                       "target_samples": plan["new_sample_count"], "plan_path": str(path.relative_to(ROOT))})
        for name in plan["runtime"]["MODELS"]:
            path = local(ROOT / "runs" / plan["run_id"] / "db" / (db.model_slug_safe(db.compose_virtual_slug(name, 5, 4)) + ".db"))
            paths = [path] + [local(ROOT / ref) for ref in plan["runtime"].get("COLLECTION_REFERENCE_DBS", {}).get(name, [])]
            for path in paths:
                if path.exists():
                    relative = str(path.relative_to(ROOT))
                    candidates.append({"path": relative, "profile_id": name, "cohort": phase,
                                       "expected_sha256": plan.get("reference_db_hashes", {}).get(relative)})
    sources = []
    seen = {}
    failures = {}
    recorded = 0
    destination = local(collection / "observations.jsonl.gz")
    staging = local(destination.with_suffix(".gz.tmp"))
    with gzip.open(staging, "wt", encoding="utf-8", compresslevel=6) as output:
        for source in candidates:
            path = local(ROOT / source["path"])
            if source.get("expected_sha256") and digest(path) != source["expected_sha256"]:
                raise ValueError("reference archive hash mismatch: " + source["path"])
            source["source_id"] = hashlib.sha256(source["path"].encode()).hexdigest()[:20]
            conn = read_db(path)
            conn.execute("BEGIN")
            try:
                source["metadata"] = dict(conn.execute("SELECT * FROM run_meta").fetchone())
                source["collection_contract_hash"] = db.compute_collection_contract_hash(json.loads(source["metadata"]["config_snapshot"]))
                source["templates"] = dict(conn.execute("SELECT key,value FROM prompt_templates"))
                has_journal = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='request_events'").fetchone()
                source["journal"] = {"table": "request_events", "event_id_through": conn.execute("SELECT coalesce(max(event_id),0) FROM request_events").fetchone()[0]} if has_journal else None
                source["evidence_gaps"] = ["unfiltered search bodies", "dropped item text", "detector reasons", "raw HTTP attempts"] if source["cohort"] == "reference" else []
                source["observed_samples"] = 0
                for record in observed_samples(conn, source, questions):
                    identity = (record["profile_id"], record["question_id"], record["sample_idx"])
                    if record["result"]["error"] is None and identity in seen:
                        raise ValueError(f"duplicate sample identity: {identity}")
                    if record["result"]["error"] is None:
                        seen[identity] = source["source_id"]
                    else:
                        failures.setdefault(identity, set()).add(record["result"]["error"])
                    recorded += 1
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    source["observed_samples"] += 1
            finally:
                conn.close()
            sources.append(source)
    coverage = [sample_coverage(plan, set(questions), seen, failures) for plan in plans]
    staging.replace(destination)
    catalog = {"collection_id": "collection_300", "built_at": db.utcnow_iso(),
               "source_db": inventory["source_db"], "source_sha256": inventory["source_sha256"],
               "anchor_question_ids": inventory["anchor_ids"], "additional_question_ids": inventory["additional_ids"],
                "phases": phases, "coverage": coverage,
                "sources": sources, "observations_path": str(destination.relative_to(ROOT)),
                "observations_encoding": "gzip",
               "observations_sha256": digest(destination),
               "field_contract": "result keys are source s{sample_idx}_ fields with that prefix removed; JSON strings remain verbatim; existing correctness is preserved, never recomputed",
               "journal_contract": "request_events remains in each source DB; source_id resolves its path; event_id_through bounds the indexed snapshot",
               "reference_failure_records": [{"model": item["model"], **error} for item in inventory["reference_inventory"] for error in item["errors"]],
               "repair_links": [{**item, "successful_source_id": seen.get((item["profile_id"], item["question_id"], item["sample_idx"]))} for item in repairs]}
    target = local(collection / "catalog.json")
    staging = local(target.with_suffix(".json.tmp"))
    staging.write_text(json.dumps(catalog, ensure_ascii=False, indent=2))
    staging.replace(target)
    logger.info("Indexed {} recorded samples from {} sources; no scoring or analysis", recorded, len(sources))
    return catalog


if __name__ == "__main__":
    build_catalog()
