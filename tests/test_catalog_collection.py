"""Offline intake checks for independently collected raw observations."""
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from forecast_eval import db
from forecast_eval.config import ModelProfile
from scripts import catalog_collection as catalog, prepare_collection


@pytest.fixture
def returned(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "ROOT", tmp_path)
    monkeypatch.setattr(prepare_collection, "ROOT", tmp_path)
    def local(path):
        path = path.resolve()
        if not path.is_relative_to(tmp_path):
            raise ValueError("outside test project")
        return path
    monkeypatch.setattr(catalog, "local", local)
    collection = tmp_path / "collection"
    collection.mkdir()
    package = tmp_path / "package"
    (package / "jobs").mkdir(parents=True)
    profile, run_id = "test-arm", "returned-run"
    runtime = {"MODELS": [profile], "MODEL_QUESTION_IDS": {profile: ["added"]},
               "SAMPLING_N": 3, "SCORE_ANSWERS": False, "REACT_MAX_STEPS": 6,
               "MODEL_PROFILES": {profile: {"model": "test-provider"}},
               "TAVILY_MAX_RESULTS": [5], "REACT_MAX_SEARCH_CALLS": [4]}
    job = {"run_id": run_id, "runtime": runtime, "source_sha256": "source-hash"}
    job_path = package / "jobs" / "test.json"
    job_path.write_text(json.dumps(job))
    code = b"raw collection fixture\n"
    code_hash = hashlib.sha256(code).hexdigest()
    frozen = {"code_files": {"forecast_eval/react.py": code_hash, "scripts/run_handoff.py": "launcher"}}
    manifest_path = package / "manifest.json"
    manifest_path.write_text(json.dumps(frozen))
    run = tmp_path / "runs" / run_id
    (run / "db").mkdir(parents=True)
    result = run / "db" / "test-arm__r5__c4.db"
    result.write_bytes(b"placeholder; metadata is validated separately")
    assigned_runtime = {**runtime, "LLM_BASE_URL": "https://provider.example/v1",
                        "LEAK_DETECTOR_BASE_URL": "https://provider.example/v1"}
    (run / "handoff_assignment.json").write_text(json.dumps({
        "manifest_sha256": catalog.digest(manifest_path), "launcher_sha256": "launcher",
        "plan": {**job, "runtime": assigned_runtime}}))
    hashes = {"forecast_eval/react.py": code_hash}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    directory = run / "code" / fingerprint
    (directory / "forecast_eval").mkdir(parents=True)
    (directory / "forecast_eval/react.py").write_bytes(code)
    (run / "executions.jsonl").write_text(json.dumps({"directory": str(directory.relative_to(tmp_path)),
        "code_sha256": fingerprint, "file_hashes": hashes}) + "\n")
    (run / "dispatches.jsonl").write_text('{"event":"start"}\n')
    (run / "manifest.json").write_text(json.dumps({"run_id": run_id, "sampling_n": 3,
        "models": ["test-arm::r5::c4"], "hashes": {"source_db": "source-hash"}}))
    paths = [result, run / "handoff_assignment.json", run / "executions.jsonl",
             run / "dispatches.jsonl", run / "manifest.json", directory / "forecast_eval/react.py"]
    templates = {"prompt_template": "fixture"}
    item = {"profile_id": profile, "run_id": run_id, "expected_samples": 3, "status": "verified_return",
            "job_path": str(job_path.relative_to(tmp_path)), "job_sha256": catalog.digest(job_path),
            "source_sha256": "source-hash", "expected_result_db": str(result.relative_to(tmp_path)),
            "returned_artifacts": {str(p.relative_to(tmp_path)): catalog.digest(p) for p in paths}}
    registry = {"profiles": [item], "handoff_manifest_sha256": catalog.digest(manifest_path),
                "expected_prompt_templates_hash": db.compute_prompt_templates_hash(templates),
                "expected_detector_prompt_hash": "detector-hash"}
    registry_path = collection / "delegated_results.json"
    registry_path.write_text(json.dumps(registry))
    inventory = {"source_sha256": "source-hash", "additional_ids": ["added"]}
    return collection, registry_path, registry, inventory, assigned_runtime, templates


def test_unverified_return_is_not_counted_even_if_files_exist(returned):
    collection, path, registry, inventory, _, _ = returned
    registry["profiles"][0]["status"] = "awaiting_external_return"
    path.write_text(json.dumps(registry))
    sources, states = catalog.delegated_candidates(collection, inventory)
    assert sources == [] and states[0]["status"] == "awaiting_external_return"


def test_verified_return_preserves_source_identity_and_remaining_coverage(returned):
    collection, _, _, inventory, runtime, _ = returned
    sources, states = catalog.delegated_candidates(collection, inventory)
    assert len(sources) == 1 and sources[0]["cohort"] == "continuation"
    assert sources[0]["delegated_run_id"] == "returned-run"
    assert len(sources[0]["return_artifacts"]) == 6
    plan = {"phase": "continuation", "runtime": runtime, "new_sample_count": 3}
    coverage = catalog.sample_coverage(plan, {"added"}, {("test-arm", "added", 0): "returned"}, {})
    assert coverage["collected"] == 1 and coverage["missing"] == 2 and not coverage["complete"]
    assert states[0]["expected_samples"] == 3


@pytest.mark.parametrize("change,match", [
    ("body", "artifact checksum"), ("code", "artifact checksum"),
    ("missing_checksum", "required artifact checksums"), ("wal", "required artifact checksums"),
    ("scope", "scope mismatch"), ("job", "job checksum"),
    ("assignment", "frozen assignment"), ("manifest", "manifest identity"),
    ("code_identity", "snapshot identity"), ("frozen_code", "frozen code"),
])
def test_corrupt_or_out_of_scope_delegated_return_is_rejected(returned, change, match):
    collection, registry_path, registry, inventory, _, _ = returned
    item = registry["profiles"][0]
    root = collection.parent
    run = root / "runs/returned-run"
    altered = None
    if change == "body":
        (root / item["expected_result_db"]).write_bytes(b"changed")
    elif change == "code":
        code_path = next(p for p in item["returned_artifacts"] if p.endswith("react.py"))
        (root / code_path).write_bytes(b"changed")
    elif change == "missing_checksum":
        del item["returned_artifacts"][item["expected_result_db"]]
    elif change == "wal":
        Path(str(root / item["expected_result_db"]) + "-wal").write_bytes(b"unlisted WAL")
    elif change == "scope":
        inventory["additional_ids"] = ["different"]
    elif change == "job":
        (root / item["job_path"]).write_text("{}")
    else:
        name = {"assignment": "handoff_assignment.json", "manifest": "manifest.json",
                "code_identity": "executions.jsonl", "frozen_code": "executions.jsonl"}[change]
        altered = run / name
        value = json.loads(altered.read_text())
        if change == "assignment":
            value["plan"]["runtime"]["REACT_MAX_STEPS"] = 7
        elif change == "manifest":
            value["run_id"] = "different"
        elif change == "code_identity":
            value["code_sha256"] = "different"
        else:
            value["file_hashes"]["forecast_eval/react.py"] = "different"
            value["code_sha256"] = hashlib.sha256(json.dumps(value["file_hashes"], sort_keys=True).encode()).hexdigest()
        altered.write_text(json.dumps(value))
        item["returned_artifacts"][str(altered.relative_to(root))] = catalog.digest(altered)
    registry_path.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match=match):
        catalog.delegated_candidates(collection, inventory)


@pytest.mark.parametrize("change", [None, "budget", "cutoff", "model", "source", "prompt", "detector", "sampling"])
def test_delegated_metadata_matches_frozen_inference_contract(returned, change):
    collection, _, registry, inventory, runtime, templates = returned
    source = catalog.delegated_candidates(collection, inventory)[0][0]
    snapshot = {**runtime, "TAVILY_MAX_RESULTS": 5, "REACT_MAX_SEARCH_CALLS": 4,
                "leak_detector_prompt_hash": "detector-hash", "LLM_MAX_CONCURRENCY": 19}
    snapshot["MODEL_PROFILES"] = {name: ModelProfile.model_validate(profile).model_dump(mode="json")
                                  for name, profile in runtime["MODEL_PROFILES"].items()}
    metadata = {"run_id": "returned-run", "model": "test-arm::r5::c4", "sampling_n": 3,
                "source_db_hash": "source-hash", "prompt_templates_hash": registry["expected_prompt_templates_hash"]}
    if change == "budget": snapshot["REACT_MAX_STEPS"] = 7
    if change == "cutoff": snapshot["MODEL_TRAINING_CUTOFFS"] = {"test-arm": "2026-01-01"}
    if change == "model": metadata["model"] = "different"
    if change == "source": metadata["source_db_hash"] = "different"
    if change == "prompt": templates["prompt_template"] = "different"
    if change == "detector": snapshot["leak_detector_prompt_hash"] = "different"
    if change == "sampling": metadata["sampling_n"] = 2
    source.update(metadata={**metadata, "config_snapshot": json.dumps(snapshot)}, templates=templates)
    if change:
        with pytest.raises(ValueError, match="contract mismatch"):
            catalog.validate_delegated_metadata(source, inventory)
    else:
        catalog.validate_delegated_metadata(source, inventory)
        assert source["metadata"]["config_snapshot"] == json.dumps(snapshot)


@pytest.mark.parametrize("change", [None, "missing", "run", "model", "question", "sample"])
def test_delegated_journal_is_bound_to_assigned_slots(change):
    conn = sqlite3.connect(":memory:")
    source = {"delegated_run_id": "run", "metadata": {"model": "arm::r5::c4"},
              "allowed_question_ids": ["added"]}
    try:
        if change != "missing":
            conn.execute("CREATE TABLE request_events (run_id, model, question_id, sample_idx)")
            context = ["run", "arm::r5::c4", "added", 0]
            if change in ("run", "model", "question", "sample"):
                context[("run", "model", "question", "sample").index(change)] = "unassigned"
            conn.execute("INSERT INTO request_events VALUES (?,?,?,?)", context)
        if change:
            with pytest.raises(ValueError, match="request journal"):
                catalog.validate_delegated_journal(conn, source)
        else:
            catalog.validate_delegated_journal(conn, source)
    finally:
        conn.close()


@pytest.mark.parametrize("question,kinds,valid", [("added", ["llm.request", "sample.result"], True),
    ("added", ["sample.result"], False), ("added", ["llm.request"], False),
    ("anchor", ["llm.request", "sample.result"], False)])
def test_delegated_observation_requires_raw_attempt_and_result_evidence(question, kinds, valid):
    conn = db.connect(":memory:")
    db.init_schema(conn, 3)
    row = {"id": question, "choice_type": "single", "question_type": "yes_no", "event": "fixture",
           "options": "Yes, No", "answer": "A", "end_time": "2026-01-01"}
    conn.execute("INSERT INTO questions (id,choice_type,question_type,event,options,answer,end_time,imported_at) VALUES (?,?,?,?,?,?,?,?)",
                 (*row.values(), "fixture"))
    conn.execute("INSERT INTO run_results (question_id,user_prompt,s0_created_at,s0_error) VALUES (?,?,?,?)",
                 (question, "fixture", "fixture", "content_policy"))
    for kind in kinds:
        conn.execute("INSERT INTO request_events (run_id,model,question_id,sample_idx,attempt_id,request_id,kind,payload,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                     ("run", "arm::r5::c4", question, 0, "attempt", kind, kind, "{}", "fixture"))
    source = {"delegated_run_id": "run", "allowed_question_ids": ["added"], "metadata": {"model": "arm::r5::c4", "sampling_n": 3},
              "source_id": "source", "profile_id": "arm", "cohort": "continuation", "path": "fixture"}
    try:
        if valid:
            records = list(catalog.observed_samples(conn, source, {question: row}))
            assert len(records) == 1 and records[0]["result"]["error"] == "content_policy"
        else:
            with pytest.raises(ValueError, match="delegated"):
                list(catalog.observed_samples(conn, source, {question: row}))
    finally:
        conn.close()
