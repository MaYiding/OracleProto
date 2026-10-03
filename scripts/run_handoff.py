"""Verify and resume a privately transferred, fixed collection assignment."""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import dotenv_values
from loguru import logger

import evaluation
from scripts import collect_forecast_panel as collector
from scripts.prepare_collection import digest, local

PACKAGE = ROOT / "runs/handoff_gpt_claude"
MODELS = ("gpt-5.4", "gpt-5.4-high", "gpt-5.3-codex", "claude-sonnet-4-6", "claude-opus-4-6-think")


def verify(package: Path = PACKAGE, require_release: bool = False) -> dict:
    package = local(package)
    manifest = json.loads(local(package / "manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        path = local(package / relative)
        if not path.is_relative_to(package) or digest(path) != expected:
            raise ValueError("handoff file checksum mismatch: " + relative)
    for relative, expected in manifest["code_files"].items():
        if digest(local(ROOT / relative)) != expected:
            raise ValueError("handoff execution code mismatch: " + relative)
    anchors, additional = manifest["anchor_ids"], manifest["additional_ids"]
    if len(set(anchors)) != 80 or len(set(additional)) != 220 or set(anchors) & set(additional):
        raise ValueError("handoff question cohorts must be disjoint 80/220 sets")
    if set(manifest["jobs"]) != set(MODELS):
        raise ValueError("handoff must contain exactly five assigned models")
    for model, relative in manifest["jobs"].items():
        if relative not in manifest["files"]:
            raise ValueError("job is not covered by the manifest")
        plan = json.loads(local(package / relative).read_text())
        runtime = plan["runtime"]
        name = model + "--provider-default"
        if (runtime["MODELS"] != [name] or runtime["MODEL_PROFILES"][name]["model"] != model
                or runtime["MODEL_QUESTION_IDS"] != {name: additional}
                or runtime["SAMPLING_N"] != 3 or runtime["SCORE_ANSWERS"]
                or runtime["RUN_ID"] != plan["run_id"]):
            raise ValueError("job identity or collection scope mismatch")
    if require_release:
        if "release.json" not in manifest["files"]:
            raise ValueError("coordinator release is missing; do not start paid collection")
        release = json.loads(local(package / "release.json").read_text())
        if release.get("status") != "released" or set(release.get("models", [])) != set(MODELS):
            raise ValueError("coordinator has not released all five assignments")
    return manifest


def execute(model: str) -> int:
    manifest = verify(require_release=True)
    plan = json.loads(local(PACKAGE / manifest["jobs"][model]).read_text())
    # One recipient process at a time bounds shared quota and unfinished paid work.
    with local(PACKAGE / "recipient.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        job_dir = local(PACKAGE / "progress" / model)
        job_dir.mkdir(parents=True, exist_ok=True)
        collector.COLLECTION = job_dir
        collector.write_json(job_dir / "continuation.json", plan)
        local(ROOT / "logs/collection_300").mkdir(parents=True, exist_ok=True)
        env = dotenv_values(local(ROOT / ".env"))
        if not all(env.get(key) for key in ("LLM_BASE_URL", "LLM_API_KEY", "TAVILY_API_KEY")):
            raise ValueError("recipient .env must supply LLM_BASE_URL, LLM_API_KEY and TAVILY_API_KEY")
        plan["runtime"]["LLM_BASE_URL"] = env["LLM_BASE_URL"]
        plan["runtime"]["LEAK_DETECTOR_BASE_URL"] = env["LLM_BASE_URL"]
        os.environ["LLM_API_KEY"] = env["LLM_API_KEY"]
        os.environ["TAVILY_API_KEY"] = env["TAVILY_API_KEY"]
        directory = local(ROOT / "runs" / plan["run_id"])
        directory.mkdir(parents=True, exist_ok=True)
        evaluation._configure_logging(directory / "logs/collection.log", "INFO")
        collector.capture_execution(directory)
        collector.write_json(directory / "handoff_assignment.json", {
            "manifest_sha256": digest(local(PACKAGE / "manifest.json")),
            "launcher_sha256": digest(local(Path(__file__))), "plan": plan,
        })

        async def collect() -> int:
            stop = asyncio.Event()
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGTERM, stop.set)
            try:
                return await collector.dispatch_batches(plan, env, stop)
            finally:
                loop.remove_signal_handler(signal.SIGTERM)

        try:
            return asyncio.run(collect())
        except Exception as exc:
            plan.update(status="blocked", error_type=type(exc).__name__)
            collector.write_json(job_dir / "continuation.json", plan)
            logger.error("Collection stopped ({}); inspect retained request audit and run logs before resuming", type(exc).__name__)
            return 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["verify", "status", "run"])
    parser.add_argument("model", nargs="?", choices=MODELS)
    args = parser.parse_args()
    os.chdir(local(ROOT))
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "localhost,127.0.0.1"
    manifest = verify()
    if args.action == "verify":
        verify(require_release=True)
        logger.info("Verified five assignments: 220 questions x 3 samples each; no API calls")
    elif args.action == "status":
        for model in ([args.model] if args.model else MODELS):
            plan = json.loads(local(PACKAGE / manifest["jobs"][model]).read_text())
            collector.COLLECTION = local(PACKAGE / "progress" / model)
            collector.COLLECTION.mkdir(parents=True, exist_ok=True)
            logger.info("{}: {}", model, collector.status(plan))
    else:
        if args.model is None:
            parser.error("run requires exactly one model")
        return execute(args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
