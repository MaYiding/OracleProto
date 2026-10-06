"""Stage-2 LLM-based leak filter (search-leak-filter-v1).

The harness already has five layers of information barrier (web_search schema,
end_date injection, Tavily API end_date filter, MODEL_TRAINING_CUTOFFS, and
:online slug ban). Tavily filters by *crawl/index* date but the page body may
still describe events that happened after `q.end_time`. This module adds a
sixth layer: every result item is sent through an independent detector LLM that
returns ``keep`` / ``drop`` per item. Items the detector flags ``drop`` are
removed before the main LLM ever sees the search payload.

Key invariants enforced here:

* The detector is invoked with a strict input field whitelist
  ``(title, url, published_date, content, raw_content, cutoff_date)``. The
  ``Question`` object MUST NOT be visible — the detector only judges *time*,
  never *answers*.
* Failures fail-closed by default: HTTP / timeout / 5xx / invalid JSON /
  invalid verdict are all retried up to ``LEAK_DETECTOR_RETRY_MAX`` times,
  AUTH errors are caught locally (never propagated) and treated as immediate
  failure. ``LEAK_DETECTOR_FAIL_ACTION`` then decides drop vs keep.
* Detector requests duplicate ``llm._assert_no_browsing`` semantics inline
  because we deliberately bypass ``llm.chat`` (different client singleton,
  different backoff config). ``model.endswith(":online")`` plus any of
  ``tools`` / ``plugins`` / ``tool_choice`` keyword arguments are rejected.
* The detector verdict and reason are surfaced ONLY through ``SearchResult.audit``
  (the audit dict consumed by ``react._record_search_call``); they MUST NOT
  appear in any LLM-visible message payload.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
import weakref
from dataclasses import asdict
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from .config import Settings
from .db import audit_event
from .errors import AuthError, CollectionBlockedError, ErrorKind, classify, should_retry
from .search import SearchResult, SearchResultItem


# ---- Prompt template (D5 six principles) ---------------------------------------------

# Six explicit constraints required by the spec (search-leak-filter-v1):
#   1. cutoff_date placeholder rendered as ISO 8601
#   2. specific / scheduled / speculative future events all → drop
#   3. "When in doubt, output drop, not keep."
#   4. "Judge solely from the provided text. Do NOT use your own knowledge…"
#   5. strict JSON output {"verdict": "keep|drop", "reason": "..."}
#   6. NO mention of "question", "answer", "options" (avoid framing as answer
#      auditor). Words below carefully avoid those tokens.
LEAK_DETECTOR_PROMPT_TEMPLATE: str = """\
You audit a single search result for temporal leakage against a hard cutoff date.

cutoff_date: {cutoff_date}

Definition of leakage: the result text mentions ANY event whose realisation,
scheduling or speculation references a moment STRICTLY AFTER cutoff_date.
This includes:

- specific observed event happening after cutoff_date (e.g. "the 2026-09 ruling
  upheld the prior decision");
- scheduled future event whose date is after cutoff_date (e.g. "the launch is
  scheduled for 2026-08-15");
- speculative future prediction explicitly tied to a post-cutoff window (e.g.
  "analysts expect the merger to close in late 2026").

Decision rules:

1. Judge solely from the provided text. Do NOT use your own knowledge of what
   events have or have not occurred. Even if you "know" something happened, only
   act on what the text explicitly says.
2. When in doubt, output drop, not keep.
3. References to events strictly on or before cutoff_date are fine.
4. Mere republication / aggregation pages whose body only describes pre-cutoff
   events should be kept.

Output exactly one JSON object on a single line, with no prose around it:

{{"verdict": "keep" | "drop", "reason": "<one short sentence>"}}

Result fields under audit:

title: {title}
url: {url}
published_date: {published_date}
content: {content}
raw_content: {raw_content}
"""


def _compute_prompt_hash() -> str:
    """sha256 of the prompt template, first 16 hex chars.

    Recorded in ``run_meta.config_snapshot.leak_detector_prompt_hash`` so two
    runs produced with different prompt revisions show different fingerprints
    even when the human-readable ``LEAK_DETECTOR_PROMPT_VERSION`` was forgotten.
    """
    return hashlib.sha256(
        LEAK_DETECTOR_PROMPT_TEMPLATE.encode("utf-8")
    ).hexdigest()[:16]


# ---- Detector client (process-level singleton, independent of llm._client) ---

_detector_client: AsyncOpenAI | None = None
_detector_semaphores: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _detector_semaphore(limit: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    semaphores = _detector_semaphores.setdefault(loop, {})
    return semaphores.setdefault(limit, asyncio.Semaphore(limit))


def get_detector_client(settings: Settings) -> AsyncOpenAI:
    """Lazy module-level ``AsyncOpenAI`` singleton for detector calls.

    The detector lives in a separate namespace from the main LLM so we MUST
    NOT reuse ``forecast_eval/llm.py:_client`` even when base_url / api_key
    coincide. Reasons: independent quota accounting, independent timeouts /
    backoff, clearer log triage.

    ``LEAK_DETECTOR_BASE_URL`` left empty falls back to ``LLM_BASE_URL`` so
    "use the same provider as the main LLM" stays a one-line .env override.
    Cell-local overrides of ``LEAK_DETECTOR_API_KEY`` / ``LEAK_DETECTOR_BASE_URL``
    are NOT supported — the singleton is captured at first access and reused
    across grid cells (Non-Goal: detector is not a grid axis).
    """
    global _detector_client
    if _detector_client is None:
        base_url = settings.LEAK_DETECTOR_BASE_URL or settings.LLM_BASE_URL
        _detector_client = AsyncOpenAI(
            api_key=settings.LEAK_DETECTOR_API_KEY,
            base_url=base_url,
            max_retries=0,
        )
    return _detector_client


# ---- Inline send-time assertions (mirror llm._assert_no_browsing) -----------


def _assert_detector_safe(model: str, kwargs: dict[str, Any]) -> None:
    """Send-time refusal mirroring ``llm._assert_no_browsing``.

    The startup-time ``Settings._post_validate`` already rejects ``:online``
    slugs, but a test fixture or partial config drift via
    ``model_copy(update={...})`` could bypass that. Re-check here so a
    poisoned cell-local view never reaches the SDK. Detector kwargs MUST NOT
    carry ``tools`` / ``plugins`` / ``tool_choice`` either: the detector is a
    pure text auditor and external tool injection would re-open the
    information barrier.
    """
    if model.endswith(":online"):
        raise ValueError(
            f"detector model {model!r} ends with ':online' — provider-native "
            "browsing is not allowed"
        )
    for forbidden in ("tools", "plugins", "tool_choice"):
        if forbidden in kwargs:
            raise ValueError(
                f"detector kwargs MUST NOT carry {forbidden!r} (provider-native "
                "browsing / extra tool surface is forbidden)"
            )


# ---- Verdict parsing --------------------------------------------------------


def _parse_verdict(content: str) -> tuple[str, str] | None:
    """Return ``(verdict, reason)`` or ``None`` on any structural defect.

    Accept the smallest superset of well-formed responses: the assistant
    message must contain a JSON object with ``verdict`` ∈ ``{keep, drop}`` and
    a string ``reason``. Anything else → None (caller treats as a retry-eligible
    failure). Robust against models that wrap the JSON in trailing prose by
    extracting the first balanced ``{...}`` substring.
    """
    if not content:
        return None
    text = content.strip()
    # Accept either pure JSON or a JSON object embedded in prose.
    candidates: list[str] = []
    if text.startswith("{") and text.endswith("}"):
        candidates.append(text)
    # Try to grab the first {...} via brace counting.
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break
    for raw in candidates:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        verdict = obj.get("verdict")
        reason = obj.get("reason", "")
        if verdict not in ("keep", "drop"):
            return None
        if not isinstance(reason, str):
            reason = str(reason)
        return verdict, reason
    return None


def _render_user_message(item: SearchResultItem, cutoff_date: str) -> str:
    """Render the per-item detector prompt with the input field whitelist.

    Strictly six fields: title / url / published_date / content / raw_content
    / cutoff_date. ``None`` values are projected to literal placeholder strings
    so the rendered prompt stays JSON-friendly and never includes Python's
    "None" repr (which the model might confuse with an absent date).
    """
    return LEAK_DETECTOR_PROMPT_TEMPLATE.format(
        cutoff_date=cutoff_date,
        title=item.title or "",
        url=item.url or "",
        published_date=item.published_date if item.published_date else "(unknown)",
        content=item.content or "",
        raw_content=item.raw_content if item.raw_content else "(empty)",
    )


# ---- Per-item detector call -------------------------------------------------


def _failure_reason(kind: ErrorKind, exc: BaseException | None) -> str:
    """Stable short string for audit reasons (won't drift if str(exc) jitters)."""
    if exc is None:
        return f"failed:{kind.value if hasattr(kind, 'value') else str(kind)}"
    return f"failed:{kind.value if hasattr(kind, 'value') else str(kind)}: {type(exc).__name__}"


async def _detect_one(
    item: SearchResultItem,
    cutoff_date: str,
    settings: Settings,
    client: AsyncOpenAI,
) -> tuple[str, str]:
    """Run the detector against a single result item with retry / fail-closed.

    Return values:
        ("keep", reason)    — detector judged the result safe
        ("drop", reason)    — detector flagged temporal leakage
        ("failed:<kind>", reason) — retries exhausted; caller applies FAIL_ACTION

    AUTH errors (401 / 403) are caught locally and converted to
    ``failed:auth`` — they MUST NOT propagate, otherwise
    ``runner._run_task_with_retry`` would treat them as a top-level run abort.
    """
    model = settings.LEAK_DETECTOR_MODEL
    user_message = _render_user_message(item, cutoff_date)
    base_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": user_message}],
        "max_tokens": settings.LEAK_DETECTOR_MAX_TOKENS,
        "temperature": settings.LEAK_DETECTOR_TEMPERATURE,
        "timeout": settings.LEAK_DETECTOR_TIMEOUT_S,
    }
    effort = getattr(settings, "LEAK_DETECTOR_REASONING_EFFORT", None)
    if effort is not None:
        base_kwargs["extra_body"] = {"reasoning_effort": effort}
    response_format = getattr(settings, "LEAK_DETECTOR_RESPONSE_FORMAT", None)
    if response_format is not None:
        base_kwargs["response_format"] = {"type": response_format}
    _assert_detector_safe(model, base_kwargs)

    backoff = list(settings.LEAK_DETECTOR_BACKOFF_S)
    max_attempts = max(1, int(settings.LEAK_DETECTOR_RETRY_MAX) + 1)
    last_kind: ErrorKind | None = None
    last_exc: BaseException | None = None
    last_content: str | None = None

    for attempt in range(1, max_attempts + 1):
        request_id = audit_event("detector.request", {
            "endpoint": (settings.LEAK_DETECTOR_BASE_URL or settings.LLM_BASE_URL).rstrip("/") + "/chat/completions",
            "body": {**{k: v for k, v in base_kwargs.items() if k not in ("timeout", "extra_body")},
                     **base_kwargs.get("extra_body", {})}, "attempt": attempt,
        })
        try:
            raw = await client.chat.completions.with_raw_response.create(**base_kwargs)
        except asyncio.CancelledError:
            audit_event("detector.cancelled", {}, request_id=request_id)
            raise
        except Exception as exc:
            audit_event("detector.error", {"type": type(exc).__name__, "message": str(exc),
                        "status": getattr(exc, "status_code", None),
                        "body": getattr(getattr(exc, "response", None), "text", None)}, request_id=request_id)
            kind = classify(exc)
            last_exc = exc
            last_kind = kind
            if kind is ErrorKind.AUTH:
                # Local catch: detector key misconfig MUST NOT abort the run.
                logger.warning(
                    "leak_filter detector AUTH failure (no retry); kind={} err={}",
                    kind,
                    exc,
                )
                return _failure_reason(kind, exc), "auth"
            if not should_retry(kind):
                logger.warning(
                    "leak_filter detector non-retryable error kind={} attempt={} err={}",
                    kind,
                    attempt,
                    exc,
                )
                return _failure_reason(kind, exc), str(exc)[:200]
            # retryable: fall through to backoff
        else:
            audit_event("detector.response", {"status": getattr(raw, "status_code", None),
                        "body": getattr(raw, "text", None)}, request_id=request_id)
            try:
                parsed = raw.parse()
                content = parsed.choices[0].message.content if parsed.choices else None
            except Exception as exc:  # noqa: BLE001 — bad envelope is a parse failure
                last_kind = ErrorKind.UNKNOWN
                last_exc = exc
                content = None
            last_content = content if isinstance(content, str) else None
            verdict_pair = _parse_verdict(last_content or "")
            if verdict_pair is not None:
                audit_event("detector.verdict", {"verdict": verdict_pair[0], "reason": verdict_pair[1]}, request_id=request_id)
                return verdict_pair
            # Treat parse failure as a logical failure: count it as a retry.
            last_kind = last_kind or ErrorKind.UNKNOWN

        # Decide whether to retry based on attempt budget.
        if attempt >= max_attempts:
            break
        idx = attempt - 1
        wait_s = float(backoff[idx]) if idx < len(backoff) else 0.0
        if wait_s > 0:
            await asyncio.sleep(wait_s)

    kind = last_kind or ErrorKind.UNKNOWN
    reason = (
        f"retries exhausted (kind={kind})"
        if last_exc is None
        else f"retries exhausted (kind={kind}, last={type(last_exc).__name__})"
    )
    return _failure_reason(kind, last_exc), reason


# ---- Top-level filter -------------------------------------------------------


def _dominant_error_kind(verdicts: list[str]) -> str | None:
    """First failed kind that appears, or None if no failures.

    Returning the *first occurrence* keeps the audit deterministic when
    detector errors are mixed (e.g. one network blip + one bad JSON would
    surface "network" since it appeared first).
    """
    for v in verdicts:
        if v.startswith("failed:"):
            payload = v[len("failed:") :]
            kind = payload.split(":", 1)[0]
            return kind
    return None


async def filter_search_result(
    result: SearchResult,
    *,
    end_date: str,
    settings: Settings,
    client: AsyncOpenAI | None = None,
) -> SearchResult:
    """Apply the detector to ``result.results`` and mutate-in-place.

    The function returns the same ``SearchResult`` instance with audit metadata
    populated under ``result.audit``. Callers (``search.tavily_search``)
    receive a result whose ``.results`` list has been pruned to verdict=keep
    items (plus failed items when ``LEAK_DETECTOR_FAIL_ACTION=keep``).

    When all items are dropped we also clear ``result.answer`` because the
    Tavily ``answer`` field is synthesised from the same set of pages and
    leaving it intact would let the main LLM read a summary derived from
    leaked content.
    """
    if not result.ok:
        # Defensive: callers should not invoke us on failed Tavily payloads,
        # but if they do, return the result unchanged (no audit, no detector
        # calls). This mirrors the Tavily-failed scenario in the spec.
        return result

    items = list(result.results)
    n_raw = len(items)
    # ``published_dates_raw`` holds raw-order published_date for every item the
    # detector audited. ``react._record_search_call`` re-uses this so
    # ``search_calls.published_dates`` length stays == n_results_raw (audit
    # invariant from results-persistence spec, even after items are pruned).
    published_dates_raw = [it.published_date for it in items]
    audit_skeleton: dict[str, Any] = {
        "n_results_raw": n_raw,
        "n_results_kept": n_raw,
        "detector_verdicts": [],
        "detector_latency_ms": 0,
        "detector_error_kind": None,
        "published_dates_raw": published_dates_raw,
        "results_raw": [asdict(item) for item in items],
        "detector_reasons": [],
    }
    if n_raw == 0:
        result.audit = audit_skeleton
        return result

    detector_client = client if client is not None else get_detector_client(settings)
    semaphore = _detector_semaphore(max(1, int(settings.LEAK_DETECTOR_CONCURRENCY)))

    async def _bounded(idx: int, it: SearchResultItem) -> tuple[int, str, str]:
        async with semaphore:
            verdict, reason = await _detect_one(it, end_date, settings, detector_client)
            return idx, verdict, reason

    t0 = time.monotonic()
    pairs = await asyncio.gather(
        *[_bounded(i, it) for i, it in enumerate(items)]
    )
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    # asyncio.gather preserves submission order, but be defensive: sort by idx
    # so verdicts[i] corresponds to original items[i] (audit indexing
    # invariant).
    pairs.sort(key=lambda p: p[0])
    verdicts = [v for _, v, _ in pairs]
    content_policy_drops = {
        verdict for verdict in verdicts
        if getattr(settings, "LEAK_DETECTOR_DROP_CONTENT_POLICY", False)
        and verdict.split(":", 2)[:2] == ["failed", "content_policy"]
    }
    fail_action = settings.LEAK_DETECTOR_FAIL_ACTION
    kept: list[SearchResultItem] = []
    for it, verdict in zip(items, verdicts):
        if verdict == "keep":
            kept.append(it)
        elif verdict == "drop" or verdict in content_policy_drops:
            continue
        else:  # failed:*
            if fail_action == "keep":
                kept.append(it)
            # else drop (default fail-closed)

    audit_skeleton["n_results_kept"] = len(kept)
    audit_skeleton["detector_verdicts"] = verdicts
    audit_skeleton["detector_reasons"] = [reason for _, _, reason in pairs]
    audit_skeleton["detector_latency_ms"] = elapsed_ms
    audit_skeleton["detector_error_kind"] = _dominant_error_kind(verdicts)

    result.results = kept
    if not kept:
        # Tavily's `answer` is synthesised from the same pages — clearing it
        # avoids leaking content via the summary even when every item is
        # dropped. Partial-drop case keeps the answer (known trade-off, see
        # search-tool spec).
        result.answer = None
    result.audit = audit_skeleton
    audit_event("search.filtered", {"audit": audit_skeleton, "visible_payload": result.to_llm_payload()})
    blocking_failures = [v for v in verdicts if v.startswith("failed:") and v not in content_policy_drops]
    if blocking_failures and getattr(settings, "REQUIRE_HEALTHY_RETRIEVAL", False):
        raise CollectionBlockedError("detector failed; collection paused with raw evidence preserved")
    return result
