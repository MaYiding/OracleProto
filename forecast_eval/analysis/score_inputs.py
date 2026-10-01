"""Read immutable scoring observations from a run and its declared reference DBs."""
from __future__ import annotations

import hashlib
import gzip
import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

from ..db import collection_contract, parse_virtual_slug
from ..errors import AnalysisContractError
from ..parser import parse_answer, parse_gt
from ..prompts import index_to_letter
from ..types import Question
from .scoring import bucket_of


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def local_path(path: Path) -> Path:
    resolved = path.resolve()
    protected = ("/Users/mayiding/Library/Mobile Documents", "/Users/mayiding/Library/CloudStorage",
                 "/Users/mayiding/Library/Application Support/CloudDocs")
    if any(resolved.is_relative_to(Path(p)) for p in protected):
        raise AnalysisContractError("Protected cloud paths cannot be analysis inputs or outputs")
    return resolved


def _trial(model: str, q: dict, i: int, raw: dict, source: str | None) -> dict:
    options = json.loads(q["options"])
    bucket = bucket_of(q["question_type"], q["choice_type"])
    gold = parse_gt(q["answer"])
    allowed = {index_to_letter(j) for j in range(len(options))}
    if (not options or not gold <= allowed or (bucket != "mc_multi" and len(gold) != 1)
            or (bucket in ("yes_no", "binary_named") and len(options) != 2)):
        raise AnalysisContractError(f"Invalid options or gold answer: {q['id']}")
    error = raw.get("error")
    state, pred = "missing", None
    if raw:
        if error == "skipped_training_cutoff":
            state = "cutoff"
        elif error == "content_policy":
            state = "refusal"
        elif error is not None:
            state = "error"
        else:
            if raw.get("parse_ok") == 1:
                try:
                    labels = json.loads(raw.get("final_answer_letters") or "null")
                    if isinstance(labels, list) and all(isinstance(x, str) for x in labels):
                        pred = frozenset(labels)
                except (TypeError, ValueError):
                    pass
            elif raw.get("parse_ok") is None:
                pred = parse_answer(raw.get("final_answer_raw") or "", Question(**q))
            state = "valid"
            if not pred or not pred <= allowed or (bucket != "mc_multi" and len(pred) != 1):
                state, pred = "invalid", None
    return {
        "model": model, "question_id": q["id"], "sample_idx": i, "bucket": bucket,
        "state": state, "error": error, "prediction": sorted(pred) if pred is not None else None,
        "gold": sorted(gold), "option_count": len(options), "source_db": source,
        **{key: raw.get(key) for key in ("tool_calls_count", "react_steps", "latency_ms",
            "prompt_tokens", "completion_tokens", "reasoning_tokens")},
    }


def _read_db(path: Path) -> tuple[dict, dict, dict]:
    path = local_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Declared model database is absent: {path}")
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        metas = conn.execute("SELECT * FROM run_meta").fetchall()
        if len(metas) != 1:
            raise AnalysisContractError(f"Expected exactly one run_meta row: {path}")
        meta = dict(metas[0])
        k = meta["sampling_n"]
        if not isinstance(k, int) or k < 1:
            raise AnalysisContractError(f"Invalid sampling_n in {path}")
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(run_results)")}
        fields = ("created_at", "error", "parse_ok", "final_answer_letters", "final_answer_raw",
                  "tool_calls_count", "react_steps", "latency_ms", "prompt_tokens",
                  "completion_tokens", "reasoning_tokens")
        selected = [f"s{i}_{f}" for i in range(k) for f in fields if f"s{i}_{f}" in columns]
        for i in range(k):
            if f"s{i}_created_at" not in columns:
                raise AnalysisContractError(f"Missing declared sample slot s{i}: {path}")
        records = {r["question_id"]: dict(r) for r in conn.execute(
            "SELECT question_id," + ",".join(selected) + " FROM run_results ORDER BY question_id")}
        questions = {r["id"]: dict(r) for r in conn.execute(
            "SELECT id,choice_type,question_type,event,options,answer,end_time FROM questions ORDER BY id")}
        if records.keys() - questions.keys():
            raise AnalysisContractError(f"Result rows have no question metadata: {path}")
        return meta, questions, records
    finally:
        conn.close()


def _check_reference(primary: dict, reference: dict) -> dict[str, Any]:
    for key in ("model", "sampling_n", "source_db_hash", "metadata_hash", "prompt_templates_hash",
                "reflection_protocol_hash", "belief_protocol_hash", "filters_snapshot", "training_cutoff"):
        if primary.get(key) != reference.get(key):
            raise AnalysisContractError(f"Reference database differs in {key}")
    a = collection_contract(json.loads(primary["config_snapshot"]))
    b = collection_contract(json.loads(reference["config_snapshot"]))
    strata = {}
    # These detector execution settings are explicitly retained as source strata.
    for key in ("LEAK_DETECTOR_RESPONSE_FORMAT", "LEAK_DETECTOR_MAX_TOKENS",
                "LEAK_DETECTOR_DROP_CONTENT_POLICY"):
        if a.get(key) != b.get(key):
            strata[key] = {"primary": a.get(key), "reference": b.get(key)}
        a.pop(key, None)
        b.pop(key, None)
    if a != b:
        raise AnalysisContractError("Reference database inference contract differs")
    return strata


def load_model(run_dir: Path, manifest: dict, model: str) -> dict[str, Any]:
    primary_path = local_path(run_dir / "db" / manifest["model_files"][model])
    primary, questions, rows = _read_db(primary_path)
    if primary["model"] != model:
        raise AnalysisContractError(f"Model identity does not match manifest: {model}")
    k = primary["sampling_n"]
    if "sampling_n" in manifest and manifest["sampling_n"] != k:
        raise AnalysisContractError("Manifest and database sampling_n differ")
    base = (parse_virtual_slug(model) or (model,))[0]
    snapshot = json.loads(primary["config_snapshot"])
    refs = snapshot.get("COLLECTION_REFERENCE_DBS", {}).get(base, [])
    sources = [(primary_path, primary, questions, rows, {})]
    questions = dict(questions)
    seen = {primary_path}
    for entry in refs:
        path = local_path(run_dir.parent.parent / entry)
        if not path.is_relative_to(local_path(run_dir.parent)):
            raise AnalysisContractError("Reference DB must remain inside the run root")
        if path in seen:
            continue
        seen.add(path)
        meta, qs, rs = _read_db(path)
        strata = _check_reference(primary, meta)
        for qid, q in qs.items():
            if qid in questions and questions[qid] != q:
                raise AnalysisContractError(f"Question metadata differs across sources: {qid}")
            questions[qid] = q
        sources.append((path, meta, qs, rs, strata))
    ids = manifest.get("filters", {}).get("question_ids")
    ids = sorted(questions) if ids is None else sorted(ids)
    if len(ids) != len(set(ids)) or set(ids) - questions.keys():
        raise AnalysisContractError("Declared questions are duplicated or absent from the database")
    cutoff = primary.get("training_cutoff")
    declared_cutoffs = manifest.get("model_training_cutoffs", {})
    declared = declared_cutoffs.get(model, declared_cutoffs.get(base))
    if cutoff and declared and cutoff != declared:
        raise AnalysisContractError("Manifest and database training cutoffs differ")
    cutoff = cutoff or declared
    normalized, observations, source_audit = {}, {}, []
    for path, meta, qs, rs, strata in sources:
        selected_qs = {qid: qs[qid] for qid in ids if qid in qs}
        selected_rows = {qid: rs[qid] for qid in ids if qid in rs}
        source_audit.append({
            "path": str(path), "run_id": meta["run_id"],
            "config_snapshot_sha256": canonical_hash(json.loads(meta["config_snapshot"])),
            "source_db_hash": meta["source_db_hash"], "metadata_hash": meta["metadata_hash"],
            "prompt_templates_hash": meta["prompt_templates_hash"],
            "reflection_protocol_hash": meta.get("reflection_protocol_hash"),
            "belief_protocol_hash": meta.get("belief_protocol_hash"),
            "observations_sha256": canonical_hash({"questions": selected_qs, "rows": selected_rows}),
            "detector_strata": strata,
        })
    for qid in ids:
        q = questions[qid]
        options = json.loads(q["options"])
        bucket = bucket_of(q["question_type"], q["choice_type"])
        gold = parse_gt(q["answer"])
        allowed = {index_to_letter(i) for i in range(len(options))}
        if (not options or not gold <= allowed or (bucket != "mc_multi" and len(gold) != 1)
                or (bucket in ("yes_no", "binary_named") and len(options) != 2)):
            raise AnalysisContractError(f"Invalid options or gold answer: {qid}")
        slots = []
        for i in range(k):
            candidates = []
            for path, _, _, rs, _ in sources:
                row = rs.get(qid, {})
                if row.get(f"s{i}_created_at") is not None:
                    value = {key[len(f's{i}_'):]: val for key, val in row.items() if key.startswith(f"s{i}_")}
                    candidates.append((path, value))
            completed = [(p, v) for p, v in candidates if v.get("error") in (None, "content_policy")]
            chosen = completed or candidates
            if len({canonical_hash(v) for _, v in chosen}) > 1:
                raise AnalysisContractError(f"Conflicting duplicate observations: {model}/{qid}/s{i}")
            path, raw = chosen[0] if chosen else (None, {})
            slots.append(_trial(model, q, i, raw, str(path) if path else None))
        excluded = bool(cutoff and date.fromisoformat(q["end_time"]) <= date.fromisoformat(cutoff))
        markers = [s["state"] == "cutoff" for s in slots]
        if any(markers) and not all(markers) and not excluded:
            raise AnalysisContractError(f"Mixed cutoff and admitted slots: {qid}")
        if excluded and any(s["state"] not in ("cutoff", "missing") for s in slots):
            raise AnalysisContractError(f"Observed result violates the declared training cutoff: {qid}")
        if excluded or all(markers):
            continue
        normalized[qid] = {"id": qid, "bucket": bucket, "options": options,
                           "gold": sorted(gold), "event": q["event"], "end_time": q["end_time"]}
        observations[qid] = slots
    return {"model": model, "sampling_n": k, "questions": normalized, "trials": observations,
            "sources": source_audit, "declared_questions": len(ids),
            "cutoff_questions": len(ids) - len(normalized)}


def load_catalog(catalog_path: Path, catalog: dict, profiles: list[str] | None = None) -> dict[str, dict]:
    """Consume the hashed, audited catalog export, including explicit profile aliases and repairs."""
    catalog_path = local_path(catalog_path)
    root = catalog_path.parent.parent.parent
    declared = set(s["profile_id"] for s in catalog["sources"])
    for phase in catalog["coverage"]:
        declared.update(phase["profiles"])
    models = sorted(declared if profiles is None else profiles)
    if not models or len(models) != len(set(models)) or set(models) - declared:
        raise AnalysisContractError("Select distinct profile IDs declared by the catalog")
    ids = catalog["anchor_question_ids"] + catalog["additional_question_ids"]
    if len(ids) != len(set(ids)):
        raise AnalysisContractError("Catalog question IDs overlap")
    ids = sorted(ids)
    corpus = local_path(root / catalog["source_db"])
    if not corpus.is_relative_to(root):
        raise AnalysisContractError("Catalog corpus must remain inside the project")
    with corpus.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != catalog["source_sha256"]:
            raise AnalysisContractError("Catalog corpus hash differs")
    conn = sqlite3.connect(corpus.as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        questions = {row["id"]: dict(row) for row in conn.execute(
            "SELECT id,choice_type,question_type,event,options,answer,end_time FROM test_cases") if row["id"] in ids}
    finally:
        conn.close()
    if set(ids) != questions.keys():
        raise AnalysisContractError("Catalog corpus lacks declared questions")
    sources = {}
    for source in catalog["sources"]:
        identity = source["source_id"]
        if identity in sources and sources[identity] != source:
            raise AnalysisContractError("A catalog source ID has conflicting metadata")
        meta = source["metadata"]
        base = (parse_virtual_slug(meta["model"]) or (meta["model"],))[0]
        if source["profile_id"] != base and not (
            source["cohort"] == "reference" and source["profile_id"] == base + "--provider-default"
        ):
            raise AnalysisContractError("Catalog profile alias does not match its source model")
        if canonical_hash(collection_contract(json.loads(meta["config_snapshot"]))) != source["collection_contract_hash"]:
            raise AnalysisContractError("Catalog collection contract hash differs")
        sources[identity] = source
    ks = {s["metadata"]["sampling_n"] for s in sources.values() if s["profile_id"] in models}
    if len(ks) != 1 or not isinstance(next(iter(ks)), int) or next(iter(ks)) < 1:
        raise AnalysisContractError("Catalog profiles require one declared repetition count")
    k = ks.pop()
    export = local_path(root / catalog["observations_path"])
    if not export.is_relative_to(root):
        raise AnalysisContractError("Catalog observations must remain inside the project")
    with export.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != catalog["observations_sha256"]:
            raise AnalysisContractError("Catalog observations hash differs")
    if catalog.get("observations_encoding") != "gzip":
        raise AnalysisContractError("Unsupported catalog observation encoding")
    candidates: dict[tuple[str, str, int], list[tuple[str, dict]]] = {}
    counts = {sid: 0 for sid in sources}
    retained_fields = ("created_at", "error", "parse_ok", "final_answer_letters", "final_answer_raw",
                       "tool_calls_count", "react_steps", "latency_ms", "prompt_tokens",
                       "completion_tokens", "reasoning_tokens")
    with gzip.open(export, "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            sid = record["source_id"]
            if sid not in sources or record["profile_id"] != sources[sid]["profile_id"]:
                raise AnalysisContractError("Observation source identity differs from catalog")
            if record["cohort"] != sources[sid]["cohort"] or record["question_id"] not in questions:
                raise AnalysisContractError("Observation cohort or question differs from catalog")
            index = record["sample_idx"]
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < sources[sid]["metadata"]["sampling_n"]:
                raise AnalysisContractError("Observation trial index is outside the declared range")
            counts[sid] += 1
            if record["profile_id"] not in models:
                continue
            if not record["result"].get("created_at"):
                raise AnalysisContractError("An exported observation has no completion timestamp")
            key = (record["profile_id"], record["question_id"], index)
            raw = {field: record["result"].get(field) for field in retained_fields}
            candidates.setdefault(key, []).append((sid, raw))
    if any(counts[sid] != source["observed_samples"] for sid, source in sources.items()):
        raise AnalysisContractError("Catalog observation counts differ")
    result = {}
    for model in models:
        model_sources = [s for s in sources.values() if s["profile_id"] == model]
        cutoffs = {s["metadata"]["training_cutoff"] for s in model_sources if s["metadata"].get("training_cutoff")}
        if len(cutoffs) > 1:
            raise AnalysisContractError("Profile sources disagree on the training cutoff")
        cutoff = next(iter(cutoffs), None)
        normalized, observations, audit = {}, {}, []
        for source in model_sources:
            meta = source["metadata"]
            audit.append({"path": source["path"], "run_id": meta["run_id"], "source_id": source["source_id"],
                          "cohort": source["cohort"], "collection_contract_hash": source["collection_contract_hash"],
                          "config_snapshot_sha256": canonical_hash(json.loads(meta["config_snapshot"])),
                          "source_db_hash": meta["source_db_hash"], "metadata_hash": meta["metadata_hash"],
                          "prompt_templates_hash": meta["prompt_templates_hash"],
                          "evidence_gaps": source.get("evidence_gaps", []), "observed_samples": source["observed_samples"]})
        for qid in ids:
            q = questions[qid]
            slots = []
            for i in range(k):
                available = candidates.get((model, qid, i), [])
                # A catalog repair explicitly resolves a failed source; successful answers must be unique.
                success = [(sid, raw) for sid, raw in available if raw["error"] is None]
                refused = [(sid, raw) for sid, raw in available if raw["error"] == "content_policy"]
                chosen = success or refused or available
                if len({canonical_hash(raw) for _, raw in chosen}) > 1 and (success or refused):
                    raise AnalysisContractError(f"Conflicting completed catalog observations: {model}/{qid}/s{i}")
                sid, raw = chosen[0] if chosen else (None, {})
                if success and any(v["error"] == "content_policy" for _, v in available):
                    raise AnalysisContractError("A content refusal cannot be replaced by a successful retry")
                t = _trial(model, q, i, raw, sources[sid]["path"] if sid else None)
                t["source_id"] = sid
                t["source_cohort"] = sources[sid]["cohort"] if sid else None
                slots.append(t)
            excluded = bool(cutoff and date.fromisoformat(q["end_time"]) <= date.fromisoformat(cutoff))
            markers = [t["state"] == "cutoff" for t in slots]
            if any(markers) and not all(markers) and not excluded:
                raise AnalysisContractError(f"Mixed cutoff and admitted catalog slots: {qid}")
            if excluded and any(t["state"] not in ("cutoff", "missing") for t in slots):
                raise AnalysisContractError(f"Catalog result violates the training cutoff: {qid}")
            if excluded or all(markers):
                continue
            normalized[qid] = {"id": qid, "bucket": slots[0]["bucket"], "options": json.loads(q["options"]),
                               "gold": slots[0]["gold"], "event": q["event"], "end_time": q["end_time"]}
            observations[qid] = slots
        result[model] = {"model": model, "sampling_n": k, "questions": normalized, "trials": observations,
                         "sources": audit, "declared_questions": len(ids), "cutoff_questions": len(ids) - len(normalized)}
    return result
