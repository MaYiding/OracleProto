"""Exercise detector verdicts and fixed reasoning profiles against real APIs."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx
from dotenv import dotenv_values
from loguru import logger
from openai import AsyncOpenAI

from forecast_eval import db, leak_filter, llm
from forecast_eval.config import ModelProfile, Settings
from forecast_eval.search import SearchResultItem
from forecast_eval.tools import WEB_SEARCH_SCHEMA


def profile_candidates() -> dict[str, dict]:
    profiles = {}
    families = {
        "gpt-5.4": ["none", "low", "medium", "high"],
        "claude-opus-4-6": ["low", "medium", "high"],
        "gpt-5.3-codex": ["xhigh"],
        "claude-sonnet-4-6": ["xhigh"],
        "gemini-3.1-pro-preview-customtools": ["high"],
        "glm-5": ["high"],
        "alicloud-kimi-k2.5": ["high"],
    }
    for model, efforts in families.items():
        for effort in efforts:
            profiles[f"{model}--reasoning-{effort}"] = {"model": model, "reasoning_effort": effort}
    return profiles


async def main() -> None:
    root = ROOT.resolve()
    if str(root) != "/Users/mayiding/Desktop/Git/Forecast":
        raise ValueError("run this probe in the designated local project")
    output = (root / "logs/collection_300").resolve()
    if not output.is_relative_to(root):
        raise ValueError("unsafe output")
    output.mkdir(exist_ok=True)
    env = dotenv_values(root / ".env")
    plan_path = root / "runs/collection_300/plan.json"
    profiles = json.loads(plan_path.read_text())["runtime"]["MODEL_PROFILES"] if plan_path.exists() else profile_candidates()
    settings = Settings(MODELS=list(profiles), MODEL_PROFILES=profiles,
                        ENABLE_SEARCH_LEAK_FILTER=True, LEAK_DETECTOR_MODEL="qwen3.8-flash",
                        LEAK_DETECTOR_API_KEY=env["LLM_API_KEY"], LEAK_DETECTOR_BASE_URL=env["LLM_BASE_URL"],
                        LEAK_DETECTOR_REASONING_EFFORT="none", LLM_RETRY_MAX=2,
                        LLM_MAX_TOKENS=12000, LLM_TIMEOUT_S=240,
                        LEAK_DETECTOR_RETRY_MAX=1, LEAK_DETECTOR_TIMEOUT_S=90)
    conn = db.connect(output / "probe.db")
    db.init_schema(conn, 1)
    detector = leak_filter.get_detector_client(settings)
    verdicts = []
    with db.capture_requests(conn, settings=settings, run_id="capability_probe", model="qwen3.8-flash",
                             question_id="synthetic-time-boundary", sample_idx=0):
        for expected, text in [("keep", "A factory opened on 2020-01-15."),
                               ("drop", "The factory opened on 2026-08-15.")]:
            verdict, reason = await leak_filter._detect_one(
                SearchResultItem(title="Factory report", url="https://example.org/report", content=text),
                "2025-01-01", settings, detector)
            verdicts.append({"expected": expected, "actual": verdict, "reason": reason})
    (output / "detector_probe.json").write_text(json.dumps(verdicts, ensure_ascii=False, indent=2))
    if any(item["actual"] != item["expected"] for item in verdicts):
        raise RuntimeError("detector did not pass the synthetic boundary checks")
    logger.info("Qwen3.8-Flash detector passed keep/drop boundary checks")
    saved_path = output / "profile_probes.json"
    saved = json.loads(saved_path.read_text()) if saved_path.exists() else {}
    sem = asyncio.Semaphore(1)

    async def probe(name: str) -> None:
        if saved.get(name, {}).get("ok") and saved[name].get("profile") == profiles[name]:
            return
        async with sem:
            record = {"profile": profiles[name]}
            with db.capture_requests(conn, settings=settings, run_id="capability_probe", model=name,
                                     question_id="synthetic-tool-roundtrip", sample_idx=0):
                try:
                    messages = [{"role": "user", "content": "Use web_search exactly once with query 'OracleProto probe 17 times 19'. Wait for the tool result, then return PROBE_OK and its integer value. Do not compute before the search."}]
                    first = await llm.chat(model=name, messages=messages, tools=[WEB_SEARCH_SCHEMA], settings=settings)
                    messages.append(first.message)
                    calls = first.message.get("tool_calls") or []
                    if not calls:
                        raise ValueError("no tool call emitted")
                    for call in calls:
                        if call["function"]["name"] != "web_search":
                            raise ValueError("unexpected tool")
                        messages.append({"role": "tool", "tool_call_id": call["id"],
                                         "content": json.dumps({"results": [{"title": "Synthetic probe fixture", "content": "The integer value is 323."}]})})
                    messages.append({"role": "user", "content": "Search budget exhausted. Return PROBE_OK 323 now without tools."})
                    final = await llm.chat(model=name, messages=messages, tools=[], settings=settings)
                    content = final.message.get("content") or ""
                    record.update(ok="PROBE_OK" in content and "323" in content and not final.message.get("tool_calls"),
                                  final=content, finish_reason=final.finish_reason,
                                  reasoning_tokens=first.usage.reasoning_tokens + final.usage.reasoning_tokens,
                                  reasoning_chars=len(str(first.message.get("reasoning_content") or "")) + len(str(final.message.get("reasoning_content") or "")),
                                  returned_model=getattr(final.raw, "model", None))
                    if profiles[name].get("reasoning_effort") == "none" and (record["reasoning_tokens"] or record["reasoning_chars"]):
                        record.update(ok=False, error="none still returned reasoning")
                except Exception as exc:
                    record.update(ok=False, error=str(exc).replace(settings.LLM_API_KEY, "<redacted>"))
            saved[name] = record
            saved_path.write_text(json.dumps(saved, ensure_ascii=False, indent=2))
            logger.info("{} ok={} reasoning_tokens={} error={}", name, record["ok"], record.get("reasoning_tokens"), record.get("error"))

    await asyncio.gather(*(probe(name) for name in profiles))
    conn.close()
    logger.info("Profiles accepted: {}/{}", sum(bool(saved.get(name, {}).get("ok")) for name in profiles), len(profiles))


if __name__ == "__main__":
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "localhost,127.0.0.1"
    asyncio.run(main())
