"""Prepare, run, and inspect raw forecast collection without scoring or analysis."""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import signal
import sqlite3
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import dotenv_values
import httpx
from loguru import logger
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt

import evaluation
from forecast_eval import db, runner
from forecast_eval.config import Settings
from forecast_eval.errors import CollectionBlockedError, parse_retry_after
from forecast_eval.types import QFilter
from scripts.prepare_collection import digest, local
from scripts.probe_collection import profile_candidates


COLLECTION = local(ROOT / "runs/collection_300")
PLAN = local(COLLECTION / "plan.json")
PRIORITY = ["qwen3.5-flash", "doubao-seed-2-0-lite", "DeepSeek-V3.2-Exp",
            "qwen3.5-35b-a3b", "qwen3.5-plus", "glm-5", "alicloud-kimi-k2.5",
            "minimax-m2.5", "gpt-oss-120b", "gpt-5.4", "gpt-5.4-high",
            "gpt-5.3-codex", "claude-sonnet-4-6", "claude-opus-4-6-think"]


def write_json(path: Path, value: dict) -> None:
    staging = local(path.with_name(f".{path.name}.{os.getpid()}.tmp"))
    staging.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    staging.replace(path)


def capture_execution(run_dir: Path) -> dict:
    paths = [local(ROOT / name) for name in ("evaluation.py", "pyproject.toml", "environment.yml",
             "scripts/collect_forecast_panel.py", "scripts/prepare_collection.py", "scripts/probe_collection.py",
             "scripts/catalog_collection.py")]
    paths += [local(path) for path in local(ROOT / "forecast_eval").glob("*.py")]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    directory = local(run_dir / "code" / fingerprint)
    directory.mkdir(parents=True, exist_ok=True)
    for source in paths:
        target = local(directory / source.relative_to(ROOT))
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        if digest(target) != hashes[str(source.relative_to(ROOT))]:
            raise ValueError("execution snapshot hash mismatch")
    record = {"captured_at": db.utcnow_iso(), "pid": os.getpid(), "code_sha256": fingerprint,
              "directory": str(directory.relative_to(ROOT)), "file_hashes": hashes}
    with local(run_dir / "executions.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    return record


def expensive(model: str) -> bool:
    return model.startswith(("gpt-5", "claude-", "gemini-"))


def sample_target(runtime: dict, name: str) -> int:
    slots = runtime.get("MODEL_SAMPLE_INDICES", {}).get(name, {})
    return sum(len(slots.get(question_id, range(runtime["SAMPLING_N"])))
               for question_id in runtime["MODEL_QUESTION_IDS"].get(name, range(300)))


def prepare_repairs() -> dict:
    path = local(COLLECTION / "repair.json")
    if path.exists():
        return json.loads(path.read_text())
    inventory = json.loads(local(COLLECTION / "inventory.json").read_text())
    continuation = json.loads(local(COLLECTION / "continuation.json").read_text())
    slots = {}
    sources = []
    for item in inventory["reference_inventory"]:
        name = item["model"].split("::")[0] + "--provider-default"
        if not item["errors"] or name not in continuation["runtime"]["MODELS"]:
            continue
        for error in item["errors"]:
            slots.setdefault(name, {}).setdefault(error["question_id"], []).append(error["sample_idx"])
            sources.append({"profile_id": name, "source_db": item["copy_path"], "source_sha256": item["copy_sha256"], **error})
    runtime = dict(continuation["runtime"])
    runtime.update(MODELS=list(slots), MODEL_SAMPLE_INDICES=slots,
                   MODEL_PROFILES={name: runtime["MODEL_PROFILES"][name] for name in slots},
                   MODEL_TRAINING_CUTOFFS={name: runtime["MODEL_TRAINING_CUTOFFS"][name] for name in slots},
                   MODEL_QUESTION_IDS={name: list(questions) for name, questions in slots.items()},
                   COLLECTION_REFERENCE_DBS={}, RUN_ID=runner.generate_run_id())
    count = sum(sample_target(runtime, name) for name in slots)
    plan = {"phase": "repair", "run_id": runtime["RUN_ID"], "runtime": runtime,
            "source_sha256": inventory["source_sha256"], "new_sample_count": count,
            "search_call_ceiling": count * 4, "repair_sources": sources,
            "status": "prepared" if count else "complete"}
    write_json(path, plan)
    return plan


def prepare_batches(plan: dict) -> dict:
    if "batches" in plan:
        return plan
    profiles = plan["runtime"]["MODEL_PROFILES"]
    first = [model + "--provider-default" for model in PRIORITY]
    second = list(profile_candidates())
    batches = []
    for phase, names in [("continuation", first), ("reasoning", second)]:
        runtime = dict(plan["runtime"])
        runtime.update(MODELS=names, MODEL_PROFILES={n: profiles[n] for n in names},
                       MODEL_TRAINING_CUTOFFS={n: runtime["MODEL_TRAINING_CUTOFFS"][n] for n in names},
                       MODEL_QUESTION_IDS={n: ids for n, ids in runtime["MODEL_QUESTION_IDS"].items() if n in names},
                       PROMPT_TEMPLATE_STYLE="shared", RUN_ID=runner.generate_run_id())
        count = sum(len(runtime["MODEL_QUESTION_IDS"].get(n, range(300))) * 3 for n in names)
        batch = {"phase": phase, "run_id": runtime["RUN_ID"], "runtime": runtime,
                 "source_sha256": plan["source_sha256"], "new_sample_count": count,
                 "search_call_ceiling": count * 4,
                 "reference_lineage": {n: value for n, value in plan["reference_lineage"].items() if n in names},
                 "status": "prepared" if phase == "continuation" else "awaiting_additional_quota"}
        write_json(local(COLLECTION / (phase + ".json")), batch)
        batches.append({k: batch[k] for k in ("phase", "run_id", "new_sample_count", "search_call_ceiling", "status")})
    plan.update(batches=batches, status="staged", active_phase="continuation")
    write_json(PLAN, plan)
    lines = ["# 两批原始采集计划", "", "第一批沿用可继续调用的 14 个原配置，仅补 220 题。先跑 Qwen、Doubao、DeepSeek、GLM、Kimi、MiniMax 和 gpt-oss，再跑 GPT-5 与 Claude。",
             f"第二批为 {len(second)} 个显式思考配置，每配置 300 题；等待补充额度后启动。顺序为 GPT-5.4、Claude Opus 4.6、GPT-5.3 Codex、Claude Sonnet 4.6、Gemini 3.1 Pro、GLM-5、Kimi K2.5，各模型内按配置清单依次执行。",
             "", "统一预算：每题 3 次；6 轮；4 次搜索；每次 5 条；输出上限 12,000 tokens；时间偏移 -1 天。",
             "默认模式沿用原模型 ID、采样参数及省略 effort 的方式，不把未指定的供应商默认设置标注为关闭思考。",
             "过滤模型：Qwen3.8-Flash，none，2048 tokens。原 80 题与新增题的过滤模型分层记录。",
             "", "每个配置先采最多 3 次，昂贵配置每小批最多 15 次采样。普通批次大小与预测、过滤、搜索、昂贵配置并发由运行设置及 dispatch_control.json 决定，在批次边界生效；每批实际值保存在运行目录的 dispatches.jsonl。断点续跑按未完成采样槽计数。",
             "每批开始查询 Tavily 余量，预留 100 次；不足以支撑完整小批时缩小批次或暂停。LLM 余额管理接口需要独立 Manage Key，当前没有该凭证，不能声称已检查模型账户余额；预测供应商内容拒绝单独留存，其他终止性错误即停。",
             "已完成样本不重跑。中断样本的每次请求保留独立 attempt_id，恢复时仅重试未完成样本。",
             "", "| 批次 | 运行 ID | 新增样本 | 搜索次数上限 |", "|---|---|---:|---:|"]
    for batch in batches:
        lines.append(f"| {batch['phase']} | `{batch['run_id']}` | {batch['new_sample_count']:,} | {batch['search_call_ceiling']:,} |")
    lines += ["", "## 配置清单", "", "| 批次 | 配置 ID | 请求模型 | effort | 题数 |", "|---|---|---|---|---:|"]
    for phase, names in [("continuation", first), ("reasoning", second)]:
        for name in names:
            p = profiles[name]
            lines.append(f"| {phase} | `{name}` | `{p['model']}` | {p.get('reasoning_effort', '未指定')} | {220 if phase == 'continuation' else 300} |")
    local(COLLECTION / "PLAN.md").write_text("\n".join(lines) + "\n")
    return plan


def create_plan() -> dict:
    if PLAN.exists():
        return json.loads(PLAN.read_text())
    inventory = json.loads(local(COLLECTION / "inventory.json").read_text())
    profiles = profile_candidates()
    references = {item["model"].split("::")[0]: item for item in inventory["reference_inventory"]}
    unavailable = {"DeepSeek-V3.2-Exp-Think", "grok-4-1-fast-reasoning",
                   "gemini-3.1-pro-preview", "gemini-3.1-flash-lite-preview"}
    scopes = {}
    lineage = {}
    cutoffs = {}
    for model, item in references.items():
        if model in unavailable:
            continue
        name = model + "--provider-default"
        profiles[name] = {"model": model}
        if model == "gpt-oss-120b":
            profiles[name]["replay_reasoning"] = False
        scopes[name] = inventory["additional_ids"]
        lineage[name] = {"reference_db": item["copy_path"], "sha256": item["copy_sha256"],
                         "source_run_id": item["run_id"], "reference_question_ids": inventory["anchor_ids"],
                         "reasoning_provenance": "provider_default_unspecified",
                         "reference_detector": "qwen3.6-flash-2026-04-16",
                         "collection_detector": "qwen3.8-flash", "combine_as_separate_strata": True}
    for name, profile in profiles.items():
        source_model = profile["model"]
        if source_model == "claude-opus-4-6":
            source_model = "claude-opus-4-6-think"
        elif source_model == "gemini-3.1-pro-preview-customtools":
            source_model = "gemini-3.1-pro-preview"
        elif source_model == "gemini-3.1-flash-lite":
            source_model = "gemini-3.1-flash-lite-preview"
        cutoffs[name] = references[source_model]["cutoff"]
        source_db = local(ROOT / references[source_model]["copy_path"])
        conn = sqlite3.connect(source_db.as_uri() + "?mode=ro", uri=True)
        source_config = json.loads(conn.execute("SELECT config_snapshot FROM run_meta").fetchone()[0])
        conn.close()
        profile.update(temperature=source_config["LLM_TEMPERATURE"], top_p=source_config["LLM_TOP_P"],
                       max_tokens_param=(source_config.get("MODEL_MAX_TOKENS_PARAM") or {}).get(source_model, "max_tokens"),
                       omit_sampling_fields=(source_config.get("MODEL_OMIT_SAMPLING_FIELDS") or {}).get(source_model, []))
    run_id = runner.generate_run_id()
    runtime = {
        "MODELS": list(profiles), "MODEL_PROFILES": profiles, "MODEL_QUESTION_IDS": scopes,
        "MODEL_TRAINING_CUTOFFS": cutoffs, "SOURCE_DB": inventory["source_db"],
        "SOURCE_TABLE": "test_cases", "RUN_ID": run_id, "RUNS_ROOT": "./runs",
        "SAMPLING_N": 3, "REACT_MAX_STEPS": 6, "REACT_MAX_SEARCH_CALLS": [4],
        "TAVILY_MAX_RESULTS": [5], "LLM_MAX_TOKENS": 12000, "LLM_TEMPERATURE": 0.7,
        "LLM_TOP_P": 1.0, "LLM_TIMEOUT_S": 240, "LLM_MAX_CONCURRENCY": 5,
        "SEARCH_MAX_CONCURRENCY": 5, "LEAK_DETECTOR_CONCURRENCY": 5,
        "TAVILY_END_DATE_OFFSET_DAYS": -1, "TAVILY_SEARCH_DEPTH": "basic",
        "TAVILY_INCLUDE_RAW_CONTENT": "markdown", "TAVILY_RAW_CONTENT_MAX_CHARS": 8000,
        "TAVILY_INCLUDE_ANSWER": "false", "ENABLE_WEB_SEARCH": True,
        "ENABLE_SEARCH_LEAK_FILTER": True, "LEAK_DETECTOR_MODEL": "qwen3.8-flash",
        "LEAK_DETECTOR_RESPONSE_FORMAT": "json_object",
        "LEAK_DETECTOR_DROP_CONTENT_POLICY": True,
        "LEAK_DETECTOR_BASE_URL": "https://aihubmix.com/v1", "LEAK_DETECTOR_REASONING_EFFORT": "none",
        "LEAK_DETECTOR_MAX_TOKENS": 2048, "LEAK_DETECTOR_TEMPERATURE": 0.0,
        "LEAK_DETECTOR_FAIL_ACTION": "drop", "LEAK_DETECTOR_TIMEOUT_S": 90,
        "REACT_REFLECTION_PROTOCOL": True, "REACT_MIN_SEARCH_CALLS": 0, "REACT_MAX_NUDGES": 2,
        "REACT_FINAL_ANSWER_RETRY": False, "REACT_BUDGET_EXCEEDED_DROP_TOOLS": True,
        "REACT_BUDGET_AWARENESS_PROTOCOL": True, "REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT": True,
        "REACT_FORCE_FINAL_ANSWER_LOOKAHEAD": 2, "BELIEF_PROTOCOL": False,
        "SCORE_ANSWERS": False, "WRITE_MESSAGES_TRACE": True, "WRITE_REQUEST_AUDIT": True,
        "REQUIRE_HEALTHY_RETRIEVAL": True, "DB_COMMIT_BATCH": 1, "RESUME": True,
        "COLLECTION_RETAIN_MODEL_REFUSALS": True,
    }
    plan = {"run_id": run_id, "source_sha256": inventory["source_sha256"],
            "runtime": runtime, "reference_lineage": lineage,
            "archived_unavailable_models": sorted(unavailable),
            "cutoff_provenance": "inherited conservative bounds from panel manifest; not newly verified provider training dates",
            "new_sample_count": sum(len(scopes.get(name, range(300))) * 3 for name in profiles),
            "search_call_ceiling": sum(len(scopes.get(name, range(300))) * 3 * 4 for name in profiles),
            "status": "prepared"}
    PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    lines = ["# 模型原始采集计划", "", f"运行 ID：`{run_id}`。仅采集，不计算正确性或汇总指标。",
             "", "每题 3 次采样；6 轮；4 次搜索；每次 5 条结果；模型输出上限 12,000 tokens。",
             "过滤模型：Qwen3.8-Flash，none，2048 tokens。原始 HTTP、未截断检索返回、过滤原文和理由均写入数据库。",
             "", "| 配置 ID | 请求模型 | reasoning_effort | 待采集题数 |", "|---|---|---|---:|"]
    for name, profile in profiles.items():
        lines.append(f"| `{name}` | `{profile['model']}` | {profile.get('reasoning_effort', '未指定')} | {len(scopes.get(name, range(300)))} |")
    lines.extend(["", f"待采集 {plan['new_sample_count']:,} 个样本，搜索调用上限 {plan['search_call_ceiling']:,} 次。",
                  "默认模式组的原 80 题与新增 220 题保留独立来源、请求契约和过滤模型标识；不能作为同一过滤条件的无差别样本池。",
                  "GPT-5.4 与 Claude Opus 4.6 的 high 保持 high；Claude Sonnet 4.6 的 xhigh 通过 AIHubMix 映射为 max。GLM、Kimi 的 high 表示思考开启。",
                  "原库已有错误样本保持失败状态，不能算作有效预测；详见 AUDIT.md。"])
    local(COLLECTION / "PLAN.md").write_text("\n".join(lines) + "\n")
    return plan


def status(plan: dict) -> dict:
    root = local(ROOT / "runs" / plan["run_id"])
    result = {"run_id": plan["run_id"], "target_samples": plan["new_sample_count"], "completed": 0,
              "errors": 0, "refusals": 0, "requests": 0, "profiles": {}, "observed_at": db.utcnow_iso()}
    for name in plan["runtime"]["MODELS"]:
        path = local(root / "db" / (db.model_slug_safe(db.compose_virtual_slug(name, 5, 4)) + ".db"))
        completed = errors = 0
        successful_slots = set()
        refused_slots = set()
        events = 0
        paths = [path] + [local(ROOT / ref) for ref in plan["runtime"].get("COLLECTION_REFERENCE_DBS", {}).get(name, [])]
        for path in paths:
            if not path.exists():
                continue
            conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
            for i in range(plan["runtime"]["SAMPLING_N"]):
                completed += conn.execute(f"SELECT count(*) FROM run_results WHERE s{i}_created_at IS NOT NULL AND s{i}_error IS NULL").fetchone()[0]
                errors += conn.execute(f"SELECT count(*) FROM run_results WHERE s{i}_error IS NOT NULL AND s{i}_error != 'skipped_training_cutoff'").fetchone()[0]
                successful_slots.update((row[0], i) for row in conn.execute(
                    f"SELECT question_id FROM run_results WHERE s{i}_created_at IS NOT NULL AND s{i}_error IS NULL"))
                refused_slots.update((row[0], i) for row in conn.execute(
                    f"SELECT question_id FROM run_results WHERE s{i}_created_at IS NOT NULL AND s{i}_error = 'content_policy'"))
            events += conn.execute("SELECT count(*) FROM request_events WHERE kind LIKE '%.request'").fetchone()[0]
            conn.close()
        refusals = len(refused_slots - successful_slots)
        result["profiles"][name] = {"completed": completed, "errors": errors, "refusals": refusals, "requests": events}
        result["completed"] += completed
        result["errors"] += errors
        result["refusals"] += refusals
        result["requests"] += events
    write_json(local(COLLECTION / (plan.get("phase", "collection") + "_status.json")), result)
    logger.info("collection status {}", {k: v for k, v in result.items() if k != "profiles"})
    return result


def cached_search_quota(keys: list[str], plan: dict, *, max_age_seconds: int) -> int | None:
    path = local(ROOT / "logs/collection_300/quota_latest.json")
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(saved["checked_at"])).total_seconds()
    if age > max_age_seconds:
        return None
    fingerprints = {hashlib.sha256(key.encode()).hexdigest()[:12] for key in keys}
    if fingerprints != {entry["key_fingerprint"] for entry in saved["keys"]}:
        return None
    spent = 0
    run_ids = {plan["run_id"]}
    for phase in ("continuation", "reasoning", "repair"):
        phase_path = local(COLLECTION / (phase + ".json"))
        if phase_path.exists():
            phase_plan = json.loads(phase_path.read_text())
            run_ids.add(phase_plan["run_id"])
            run_ids.update(phase_plan.get("prior_run_ids", []))
    for run_id in run_ids:
        root = local(ROOT / "runs" / run_id / "db")
        if not root.exists():
            continue
        for path in root.glob("*.db"):
            path = local(path)
            conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
            spent += conn.execute("SELECT count(*) FROM request_events WHERE kind='search.request' AND created_at>=?",
                                  (saved["checked_at"],)).fetchone()[0]
            conn.close()
    return max(0, int(saved["remaining"]) - spent)


async def available_searches(keys: list[str], plan: dict) -> int:
    cached = cached_search_quota(keys, plan, max_age_seconds=3600)
    if cached is not None:
        return cached
    try:
        return await query_search_quota(keys)
    except (httpx.HTTPError, CollectionBlockedError):
        cached = cached_search_quota(keys, plan, max_age_seconds=86400)
        if cached is None:
            raise
        logger.warning("Usage endpoint unavailable; using the last confirmed quota minus all subsequent search attempts: {}", cached)
        return cached


async def query_search_quota(keys: list[str]) -> int:
    checked_at = db.utcnow_iso()
    records = []

    def retryable(exc: BaseException) -> bool:
        return isinstance(exc, httpx.TransportError) or (
            isinstance(exc, httpx.HTTPStatusError) and
            (exc.response.status_code == 429 or exc.response.status_code >= 500))

    def retry_wait(state) -> float:
        response = getattr(state.outcome.exception(), "response", None)
        delay = parse_retry_after(response.headers) if response is not None else None
        return max(0.0, delay) if delay is not None else min(60.0, 30.0 * state.attempt_number)

    remaining = 0
    async with httpx.AsyncClient(timeout=30) as client:
        for key in dict.fromkeys(keys):
            fingerprint = hashlib.sha256(key.encode()).hexdigest()[:12]
            async for attempt in AsyncRetrying(retry=retry_if_exception(retryable), wait=retry_wait,
                                               stop=stop_after_attempt(3), sleep=asyncio.sleep, reraise=True,
                                               before_sleep=lambda state: logger.warning(
                                                   "Tavily usage key={} attempt={} error={} retry_in={}s",
                                                   fingerprint, state.attempt_number,
                                                   type(state.outcome.exception()).__name__, state.next_action.sleep)):
                with attempt:
                    response = await client.get("https://api.tavily.com/usage", headers={"Authorization": "Bearer " + key})
                    if response.status_code == 429 or response.status_code >= 500:
                        response.raise_for_status()
            if response.status_code in (401, 403, 432, 433):
                records.append({"key_fingerprint": fingerprint, "http_status": response.status_code, "remaining": 0})
                continue
            response.raise_for_status()
            body = response.json()
            account = body.get("account", {})
            usage = body.get("key", body.get("usage", {}))
            if not isinstance(account.get("plan_limit"), (int, float)):
                raise CollectionBlockedError("Tavily usage response has no account plan limit")
            left = max(0, account["plan_limit"] - account.get("plan_usage", 0))
            if isinstance(usage, dict) and usage.get("limit") is not None:
                left = min(left, max(0, usage["limit"] - usage.get("usage", 0)))
            records.append({"key_fingerprint": fingerprint, "http_status": 200, "remaining": left,
                            "usage": usage, "account": account})
            remaining += int(left)
    record = {"checked_at": checked_at, "finished_at": db.utcnow_iso(), "remaining": remaining, "keys": records,
              "assumption": "user-supplied keys each have an independent 1000-credit allowance"}
    write_json(local(ROOT / "logs/collection_300/quota_latest.json"), record)
    return remaining


def dispatch_settings(plan: dict, env: dict) -> Settings:
    path = local(COLLECTION / "dispatch_control.json")
    overrides = json.loads(path.read_text()) if path.exists() else {}
    bounds = {name: (1, 20) for name in ("LLM_MAX_CONCURRENCY", "SEARCH_MAX_CONCURRENCY",
                                       "LEAK_DETECTOR_CONCURRENCY", "COLLECTION_EXPENSIVE_CONCURRENCY")}
    bounds["COLLECTION_BATCH_SAMPLES"] = (3, 300)
    if not isinstance(overrides, dict) or set(overrides) - bounds.keys():
        raise CollectionBlockedError("dispatch_control.json accepts only concurrency and batch settings")
    if any(type(value) is not int or not bounds[name][0] <= value <= bounds[name][1]
           for name, value in overrides.items()):
        raise CollectionBlockedError("dispatch concurrency or batch setting is outside its integer bounds")
    return Settings(**{**plan["runtime"], **overrides}, LEAK_DETECTOR_API_KEY=env["LLM_API_KEY"])


def record_dispatch(plan: dict, event: str, **fields) -> None:
    path = local(ROOT / "runs" / plan["run_id"] / "dispatches.jsonl")
    record = {"created_at": db.utcnow_iso(), "pid": os.getpid(), "event": event, **fields}
    with path.open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def excluded_profiles(plan: dict) -> set[str]:
    shared_path = local(COLLECTION / "local_queue_exclusions.json")
    run_id = plan.get("run_id")
    run_path = local(ROOT / "runs" / run_id / "local_queue_exclusions.json") if run_id else None
    excluded = set()
    for path in (shared_path, run_path):
        if path is None or not path.exists():
            continue
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise CollectionBlockedError("local queue exclusions require an object")
        if value.get("run_id") != run_id:
            if path == run_path:
                raise CollectionBlockedError("run-local queue exclusions have a mismatched run ID")
            continue
        profiles = value.get("profiles")
        if not isinstance(profiles, list) or any(not isinstance(name, str) for name in profiles):
            raise CollectionBlockedError("local queue exclusions require profile IDs")
        if set(profiles) - set(plan["runtime"]["MODELS"]):
            raise CollectionBlockedError("local queue exclusions contain an unknown profile")
        excluded.update(profiles)
    return excluded


async def dispatch_batches(plan: dict, env: dict, stop_requested: asyncio.Event | None = None) -> int:
    path = local(COLLECTION / (plan["phase"] + ".json"))
    retain_refusals = plan["runtime"].get("COLLECTION_RETAIN_MODEL_REFUSALS", False)
    for name in plan["runtime"]["MODELS"]:
        target = sample_target(plan["runtime"], name)
        while True:
            if stop_requested is not None and stop_requested.is_set():
                plan.update(status="stopped_at_batch_boundary", updated_at=db.utcnow_iso())
                write_json(path, plan)
                record_dispatch(plan, "stopped_at_batch_boundary")
                return 0
            if name in excluded_profiles(plan):
                break
            counts = status(plan)
            done = counts["profiles"].get(name, {}).get("completed", 0)
            if retain_refusals:
                done += counts["profiles"].get(name, {}).get("refusals", 0)
            if done >= target:
                break
            free_bytes = shutil.disk_usage(ROOT).free
            if free_bytes < 10 * 1024**3:
                raise CollectionBlockedError(f"Insufficient disk space for raw collection: {free_bytes} bytes free; reserve is 10 GiB")
            settings = dispatch_settings(plan, env)
            remaining = await available_searches(settings.TAVILY_API_KEY, plan)
            if stop_requested is not None and stop_requested.is_set():
                continue
            costly = expensive(plan["runtime"]["MODEL_PROFILES"][name]["model"])
            cap = 3 if done == 0 else (15 if costly else settings.COLLECTION_BATCH_SAMPLES)
            cap = min(cap, target - done, max(0, (remaining - 100) // 12 * 3))
            if cap < min(3, target - done):
                raise CollectionBlockedError(f"Tavily reserve reached: {remaining} estimated credits remain")
            settings = settings.model_copy(update={"COLLECTION_MODEL": name, "COLLECTION_SAMPLE_LIMIT": cap,
                                                   "LLM_MAX_CONCURRENCY": settings.COLLECTION_EXPENSIVE_CONCURRENCY if costly else settings.LLM_MAX_CONCURRENCY})
            plan.update(status="running", active_profile=name, batch_sample_limit=cap,
                        tavily_remaining=remaining, updated_at=db.utcnow_iso())
            plan.pop("block_reason", None)
            write_json(path, plan)
            snapshot = db.snapshot_settings(settings)
            record_dispatch(plan, "start", profile=name, completed_before=done, sample_limit=cap,
                            config_snapshot=snapshot,
                            settings_sha256=hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest())
            logger.info("Dispatch {} new_samples<={} concurrency={} detector_concurrency={} Tavily_remaining={}",
                        name, cap, settings.LLM_MAX_CONCURRENCY, settings.LEAK_DETECTOR_CONCURRENCY, remaining)
            code = await evaluation._run_async(settings, QFilter(), plan["run_id"], local(ROOT / "runs" / plan["run_id"]), skip_analysis=True)
            counts = status(plan)
            record_dispatch(plan, "finish", profile=name, exit_code=code, counts=counts["profiles"].get(name, {}))
            if code not in (0, 6):
                raise CollectionBlockedError(f"evaluation exit {code}; inspect run manifest and request_events")
            accounted = counts["profiles"].get(name, {}).get("completed", 0)
            if retain_refusals:
                accounted += counts["profiles"].get(name, {}).get("refusals", 0)
            if accounted <= done:
                raise CollectionBlockedError("dispatch made no progress")
    counts = status(plan)
    excluded = excluded_profiles(plan)
    final_status = "local_queue_complete_pending_delegation" if excluded else (
        "complete_with_refusals" if counts.get("refusals") else "complete")
    plan.update(status=final_status,
                finished_at=db.utcnow_iso(), model_refusals=counts.get("refusals", 0))
    write_json(path, plan)
    return 0


def run(plan: dict) -> int:
    source = local(ROOT / plan["runtime"]["SOURCE_DB"])
    if digest(source) != plan["source_sha256"]:
        raise ValueError("fixed source hash mismatch")
    for path, expected in plan.get("reference_db_hashes", {}).items():
        if digest(local(ROOT / path)) != expected:
            raise ValueError("reference stratum hash mismatch: " + path)
    probes = json.loads(local(ROOT / "logs/collection_300/profile_probes.json").read_text())
    required = plan["runtime"]["MODEL_PROFILES"]
    failed = [name for name in required if not probes.get(name, {}).get("ok")
              or probes[name].get("profile") != required[name]]
    if failed:
        raise ValueError("capability probes incomplete: " + ", ".join(failed))
    env = dotenv_values(local(ROOT / ".env"))
    run_dir = local(ROOT / "runs" / plan["run_id"])
    run_dir.mkdir(exist_ok=True)
    evaluation._configure_logging(run_dir / "logs/collection.log", "INFO")
    with local(COLLECTION / "collection.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()))
        lock.flush()
        capture_execution(run_dir)
        excluded = excluded_profiles(plan)
        queue = {"run_id": plan["run_id"], "pid": os.getpid(), "effective_at": db.utcnow_iso(),
                 "excluded_profiles": sorted(excluded),
                 "local_profiles": [name for name in plan["runtime"]["MODELS"] if name not in excluded]}
        write_json(local(run_dir / "local_queue_state.json"), queue)
        record_dispatch(plan, "queue", **queue)
        async def collect() -> int:
            stop_requested = asyncio.Event()
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGTERM, stop_requested.set)
            try:
                return await dispatch_batches(plan, env, stop_requested)
            finally:
                loop.remove_signal_handler(signal.SIGTERM)
        try:
            return asyncio.run(collect())
        except Exception as exc:
            message = str(exc)
            for key in (env.get("LLM_API_KEY"), env.get("LEAK_DETECTOR_API_KEY")):
                if key:
                    message = message.replace(key, "<redacted>")
            plan.update(status="blocked", block_reason=message, updated_at=db.utcnow_iso())
            write_json(local(COLLECTION / (plan["phase"] + ".json")), plan)
            logger.error("Collection stopped: {}", message)
            return 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "run", "status"])
    parser.add_argument("--phase", choices=["continuation", "reasoning", "repair"], default="continuation")
    args = parser.parse_args()
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "localhost,127.0.0.1"
    prepare_batches(create_plan())
    if args.phase == "repair":
        prepare_repairs()
    plan = json.loads(local(COLLECTION / (args.phase + ".json")).read_text())
    if args.action == "plan":
        logger.info("Prepared {} profiles and {} samples", len(plan["runtime"]["MODELS"]), plan["new_sample_count"])
    elif args.action == "status":
        status(plan)
    else:
        return run(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
