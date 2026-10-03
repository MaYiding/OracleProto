"""Offline guards for private fixed-scope collection assignments."""
import hashlib
import json

import pytest

from scripts import run_handoff


@pytest.fixture
def package(tmp_path, monkeypatch):
    monkeypatch.setattr(run_handoff, "local", lambda p: p.resolve())
    monkeypatch.setattr(run_handoff, "digest", lambda p: hashlib.sha256(p.read_bytes()).hexdigest())
    ids = [f"additional-{i}" for i in range(220)]
    manifest = {"files": {}, "code_files": {}, "jobs": {}, "anchor_ids": [f"anchor-{i}" for i in range(80)], "additional_ids": ids}
    for model in run_handoff.MODELS:
        name = model + "--provider-default"
        relative = model + ".json"
        plan = {"run_id": model, "runtime": {"MODELS": [name], "MODEL_PROFILES": {name: {"model": model}},
                "MODEL_QUESTION_IDS": {name: ids}, "SAMPLING_N": 3, "SCORE_ANSWERS": False, "RUN_ID": model}}
        (tmp_path / relative).write_text(json.dumps(plan))
        manifest["jobs"][model] = relative
        manifest["files"][relative] = run_handoff.digest(tmp_path / relative)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path, manifest


def test_release_is_required_before_paid_execution(package):
    path, manifest = package
    assert run_handoff.verify(path)["jobs"] == manifest["jobs"]
    with pytest.raises(ValueError, match="release is missing"):
        run_handoff.verify(path, require_release=True)


def test_tampered_job_cannot_resume(package):
    path, _ = package
    (path / "gpt-5.4.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        run_handoff.verify(path)


def test_anchor_questions_cannot_enter_added_cohort(package):
    path, manifest = package
    manifest["additional_ids"][0] = manifest["anchor_ids"][0]
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="disjoint"):
        run_handoff.verify(path)


def test_partial_release_is_rejected(package):
    path, manifest = package
    (path / "release.json").write_text(json.dumps({"status": "released", "models": ["gpt-5.4"]}))
    manifest["files"]["release.json"] = run_handoff.digest(path / "release.json")
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="all five"):
        run_handoff.verify(path, require_release=True)


def test_complete_release_passes_without_network(package):
    path, manifest = package
    (path / "release.json").write_text(json.dumps({"status": "released", "models": run_handoff.MODELS}))
    manifest["files"]["release.json"] = run_handoff.digest(path / "release.json")
    (path / "manifest.json").write_text(json.dumps(manifest))
    assert run_handoff.verify(path, require_release=True)["jobs"] == manifest["jobs"]
