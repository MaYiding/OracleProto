from __future__ import annotations

import asyncio
import json
import time
from datetime import date, timedelta
from typing import Any

import httpx
from loguru import logger

from .config import Settings
from .db import utcnow_iso
from .llm import ChatResponse, chat as llm_chat
from .parser import Belief, is_correct, parse_answer, parse_belief, parse_gt
from .prompts import (
    BELIEF_PROTOCOL,
    REFLECTION_PROTOCOL,
    _build_nudge_message,
    build_budget_awareness_protocol,
    build_continuation_after_unboxed_content,
    build_last_step_force_finalisation,
    build_penultimate_step_warning,
    build_search_budget_exhausted_commit,
    build_tool_error_status,
    render_user_prompt,
)
from .search import SearchResult, tavily_search
from .tools import (
    WEB_SEARCH_SCHEMA,
    extract_query,
    parse_tool_arguments,
    tool_error_message,
    tool_result_message,
)
from .types import Question, SampleResult


def _compute_end_date(end_time: str, offset_days: int) -> str:
    return (date.fromisoformat(end_time) + timedelta(days=offset_days)).isoformat()


def _belief_to_step_dict(step: int, b: Belief | None) -> dict[str, Any] | None:
    """Project a parsed `Belief` into the per-step trace dict, or None on
    parse failure. The `delta_reason` slot summarises *why this step changed*
    by taking the first key_evidence bullet (or the first open_questions
    bullet as a fallback) — this is what the behavior analysis layer
    consumes to plot belief evolution. `counterevidence` is the raw list (may
    be empty) so `counterevidence_engagement` can do letter-matching
    against the final choice without re-parsing the boxed answer.
    """
    if b is None:
        return None
    if b.key_evidence:
        delta_reason = b.key_evidence[0]
    elif b.open_questions:
        delta_reason = b.open_questions[0]
    else:
        delta_reason = ""
    return {
        "step": step,
        "p": b.probabilities,
        "confidence": b.confidence,
        "delta_reason": delta_reason,
        "counterevidence": list(b.counterevidence),
    }


def _record_step(
    step_metrics: list[dict[str, Any]],
    totals: dict[str, int],
    *,
    step: int,
    resp: ChatResponse,
    latency_ms: int,
    belief: Belief | None,
) -> None:
    """Append a per-step observability snapshot and accumulate token totals.

    `n_tool_calls` reads from the assistant message inside `resp.message` —
    that message is the one the loop is about to append, so the count reflects
    "tool calls emitted by THIS step" (not total). `latency_ms` is the wall
    clock for this single `llm.chat` invocation, computed by the caller via
    `time.monotonic()`. `belief` is the per-step `Belief` (or None when
    the protocol is disabled or the block failed to parse) — the entry's
    `belief` slot is always present (None when missing) so the JSON schema
    is uniform across protocol-on / protocol-off runs.
    """
    assistant_msg = resp.message
    n_tool_calls = len(assistant_msg.get("tool_calls") or [])
    step_metrics.append(
        {
            "step": step,
            "prompt": resp.usage.prompt_tokens,
            "completion": resp.usage.completion_tokens,
            "reasoning": resp.usage.reasoning_tokens,
            "latency_ms": latency_ms,
            "finish_reason": resp.finish_reason,
            "n_tool_calls": n_tool_calls,
            "belief": _belief_to_step_dict(step, belief),
        }
    )
    totals["prompt"] += resp.usage.prompt_tokens
    totals["completion"] += resp.usage.completion_tokens
    totals["reasoning"] += resp.usage.reasoning_tokens


def _record_search_call(
    search_calls: list[dict[str, Any]],
    *,
    query: str,
    end_date: str,
    result: SearchResult | None,
) -> None:
    if result is None or not result.ok:
        search_calls.append(
            {
                "query": query,
                "end_date": end_date,
                "n_results": 0,
                "published_dates": [],
                "error_kind": result.error_kind if result else "pre_call_error",
            }
        )
        return
    # search-leak-filter-v1: when the leak filter ran, surface its audit on the
    # search_calls entry. `n_results` means "results the main LLM actually
    # saw", so old analysis scripts reading entry["n_results"] still get a
    # sensible single number — they just see the post-detector count when
    # the filter was active.
    audit = result.audit
    if audit is not None:
        # `published_dates` keeps *raw* ordering (length == n_results_raw) so
        # detector_verdicts[i] aligns with published_dates[i] for forensic use.
        # `n_results` reports kept count (matches result.results length) so old
        # analysis scripts reading entry["n_results"] still get the post-filter
        # number the main LLM actually saw.
        search_calls.append(
            {
                "query": query,
                "end_date": end_date,
                "n_results": int(audit["n_results_kept"]),
                "published_dates": list(audit.get("published_dates_raw", [])),
                "n_results_raw": int(audit["n_results_raw"]),
                "n_results_kept": int(audit["n_results_kept"]),
                "detector_verdicts": list(audit["detector_verdicts"]),
                "detector_latency_ms": int(audit["detector_latency_ms"]),
                "detector_error_kind": audit["detector_error_kind"],
                "detector_reasons": list(audit.get("detector_reasons", [])),
                "results_raw": audit.get("results_raw", []),
                "raw_response": result.raw_response,
                "visible_payload": result.to_llm_payload(),
            }
        )
        return
    search_calls.append(
        {
            "query": query,
            "end_date": end_date,
            "n_results": len(result.results),
            "published_dates": [r.published_date for r in result.results],
        }
    )


async def run_react(
    q: Question,
    *,
    model: str,
    sample_idx: int,
    settings: Settings,
    templates: dict[str, str],
    run_id: str,
    search_semaphore: asyncio.Semaphore | None = None,
    httpx_client: httpx.AsyncClient | None = None,
) -> SampleResult:
    """One sample of the ReAct loop.

    The LLM receives a single `user` message with no system role. The only tool
    it sees is `web_search(query)` — `end_date` is computed from `q.end_time`
    by the tool layer and never exposed.

    This function NEVER writes `error` on success — the runner is responsible
    for wrapping it and recording retry-exhausted / content-policy errors.
    """
    end_date = _compute_end_date(q.end_time, settings.TAVILY_END_DATE_OFFSET_DAYS)
    belief_enabled = settings.BELIEF_PROTOCOL
    # force-final-answer-near-limit-v1 (A): static budget-awareness tail. The
    # cell-local int form of REACT_MAX_SEARCH_CALLS is what react sees here
    # (the dispatcher already downcast the grid list).
    if settings.REACT_BUDGET_AWARENESS_PROTOCOL:
        budget_awareness = build_budget_awareness_protocol(
            max_steps=settings.REACT_MAX_STEPS,
            max_search_calls=int(settings.REACT_MAX_SEARCH_CALLS)
            if settings.ENABLE_WEB_SEARCH
            else 0,
        )
    else:
        budget_awareness = None
    user_prompt = render_user_prompt(
        q,
        templates,
        template_style=settings.PROMPT_TEMPLATE_STYLE,
        budget_awareness_protocol=budget_awareness,
        reflection_protocol=REFLECTION_PROTOCOL if settings.REACT_REFLECTION_PROTOCOL else None,
        belief_protocol=BELIEF_PROTOCOL if belief_enabled else None,
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]

    search_calls: list[dict[str, Any]] = []
    tokens = {"prompt": 0, "completion": 0, "reasoning": 0}
    step_metrics: list[dict[str, Any]] = []
    # Aligned with each LLM step: index `i` holds the parsed Belief for step
    # `i`, or None when parsing failed (or when the protocol is disabled).
    beliefs_per_step: list[Belief | None] = []
    t0 = time.monotonic()
    final_raw = ""
    steps_executed = 0
    nudges_used = 0
    budget_exhausted_notified = False
    # When the previous step returned content but no parseable `\boxed{...}`,
    # we want to tell the model "your last reply had no boxed answer; here is
    # your live status; continue or commit". The historical implementation
    # injected that message immediately after the assistant turn — that could
    # collide with the next step's penultimate / last-step / budget-exhausted
    # injection (two consecutive user messages with overlapping content). We
    # now defer it: set the flag here, inject the unified continuation message
    # at the TOP of the next iteration UNLESS a higher-priority injection
    # (last-step cutoff, penultimate warning, budget-exhausted notice) already
    # carries the same status header for that turn.
    pending_continuation = False
    last_resp: ChatResponse | None = None
    # `effective_max_search_calls` is what the LLM-facing prompts should
    # advertise as the search budget: 0 when web_search is disabled (so the
    # status header reads "web_search disabled" instead of "0/0 used"), the
    # cell-local int otherwise. Used by every harness injection below so the
    # number the model sees is always consistent with the real budget.
    effective_max_search_calls = (
        int(settings.REACT_MAX_SEARCH_CALLS) if settings.ENABLE_WEB_SEARCH else 0
    )
    # ENABLE_WEB_SEARCH=false -> the LLM sees no tool schema at all; the loop
    # returns content on the first turn and breaks, and Tavily is never called.
    # Nudges are also disabled in this case — there are no more searches to make.
    tool_schemas: list[dict[str, Any]] = (
        [WEB_SEARCH_SCHEMA] if settings.ENABLE_WEB_SEARCH else []
    )
    nudge_enabled = (
        settings.ENABLE_WEB_SEARCH
        and settings.REACT_MIN_SEARCH_CALLS > 0
        and settings.REACT_MAX_NUDGES > 0
    )

    for step in range(settings.REACT_MAX_STEPS):
        steps_executed = step + 1
        searches_done_now = len(search_calls)
        # ------------------------------------------------------------------
        # Pre-step injection decision.
        #
        # All four runtime injection paths share the same status header (see
        # `prompts._build_status_header`) so the model gets one predictable
        # shape per turn. We pick AT MOST ONE of:
        #
        #   (1) last-step hard cutoff — `tools=[]` + force-finalise text;
        #   (2) penultimate soft warning — tools still exposed inside the
        #       LOOKAHEAD window;
        #   (3) budget-exhausted commit notice — search budget consumed and
        #       `REACT_BUDGET_EXCEEDED_DROP_TOOLS=True`, fired ONCE;
        #   (4) continuation after a content-no-boxed turn — tells the model
        #       "your last reply had no \\boxed{...}, here is the live status".
        #
        # Priority is (1) > (2) > (3) > (4); higher-priority injections
        # already carry the live status header so we never double-inject.
        # `pending_continuation` is consumed unconditionally each turn — once
        # we've entered any branch the model sees a status header, so the
        # "no boxed last turn" reminder has been delivered (or superseded).
        # ------------------------------------------------------------------
        force_final_active = (
            settings.REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT
            and (settings.REACT_MAX_STEPS - step) <= settings.REACT_FORCE_FINAL_ANSWER_LOOKAHEAD
        )
        budget_dropped = (
            settings.REACT_BUDGET_EXCEEDED_DROP_TOOLS
            and searches_done_now >= settings.REACT_MAX_SEARCH_CALLS
        )
        force_final_hard_cutoff = False
        injection: str | None = None
        if force_final_active:
            remaining = settings.REACT_MAX_STEPS - step
            if remaining == 1:
                # Last step — hard cutoff: status header + force-finalise +
                # tools=[] so the model can ONLY emit content. Supersedes
                # both the budget-exhausted branch and the continuation
                # reminder (whichever else would have fired, this one wins
                # because it carries the strictest contract).
                injection = build_last_step_force_finalisation(
                    current_step=steps_executed,
                    max_steps=settings.REACT_MAX_STEPS,
                    searches_done=searches_done_now,
                    max_search_calls=effective_max_search_calls,
                )
                force_final_hard_cutoff = True
            else:
                # Remaining ∈ [2, LOOKAHEAD] — soft warning. The penultimate
                # builder branches internally on whether the search budget is
                # already exhausted (in that case it tells the model the
                # tools are gone THIS turn). Tools list decision is made
                # below: we still hand `tool_schemas` to the LLM unless the
                # search budget is actually spent.
                injection = build_penultimate_step_warning(
                    current_step=steps_executed,
                    max_steps=settings.REACT_MAX_STEPS,
                    searches_done=searches_done_now,
                    max_search_calls=effective_max_search_calls,
                )
        elif budget_dropped and not budget_exhausted_notified:
            # Search budget just hit (or hit on a previous turn but we never
            # got around to telling the model — only fires once per run).
            injection = build_search_budget_exhausted_commit(
                current_step=steps_executed,
                max_steps=settings.REACT_MAX_STEPS,
                searches_done=searches_done_now,
                max_search_calls=effective_max_search_calls,
            )
            budget_exhausted_notified = True
        elif pending_continuation:
            # Lowest priority: previous turn was content w/o `\boxed{...}` and
            # nothing else needs to fire. Inject at the TOP of the next
            # iteration so it cannot collide with penultimate / last-step
            # injections fired in the same round.
            injection = build_continuation_after_unboxed_content(
                current_step=steps_executed,
                max_steps=settings.REACT_MAX_STEPS,
                searches_done=searches_done_now,
                max_search_calls=effective_max_search_calls,
            )

        if injection is not None:
            messages.append({"role": "user", "content": injection})
        # `pending_continuation` is reset whether or not we injected — if a
        # higher-priority branch fired, its status header has already conveyed
        # the live counters, so the continuation reminder would be redundant.
        pending_continuation = False

        # ------------------------------------------------------------------
        # Tool schema decision.
        #
        # `tools_for_this_step` is what the LLM sees on this turn:
        # - hard cutoff (last step): always `[]`
        # - search budget consumed AND DROP_TOOLS=True: always `[]`
        # - else: full tool schema list (empty when web_search disabled)
        # ------------------------------------------------------------------
        if force_final_hard_cutoff or budget_dropped:
            tools_for_this_step: list[dict[str, Any]] = []
        else:
            tools_for_this_step = tool_schemas

        t_step_start = time.monotonic()
        resp = await llm_chat(
            model=model,
            messages=messages,
            settings=settings,
            tools=tools_for_this_step,
        )
        t_step_ms = int((time.monotonic() - t_step_start) * 1000)
        assistant_msg = resp.message
        # Belief parsing is independent of the boxed-answer path: a failed
        # parse here MUST NOT pollute parse_ok / correct, MUST NOT trigger a
        # retry, MUST NOT write `error`. When the protocol is disabled we
        # skip parse_belief entirely and store None for this step.
        if belief_enabled:
            step_belief = parse_belief(assistant_msg.get("content") or "", q)
        else:
            step_belief = None
        beliefs_per_step.append(step_belief)
        _record_step(
            step_metrics,
            tokens,
            step=step,
            resp=resp,
            latency_ms=t_step_ms,
            belief=step_belief,
        )
        last_resp = resp
        messages.append(assistant_msg)

        tool_calls = assistant_msg.get("tool_calls") or []
        if not tool_calls:
            content = assistant_msg.get("content") or ""
            # Soft search-floor: nudge the model once to keep researching when
            # it commits below `REACT_MIN_SEARCH_CALLS`. Floor is opt-in
            # (default 0) and bounded by `REACT_MAX_NUDGES` so we cannot
            # nudge-loop forever. Nudges DO consume a step against the hard
            # `REACT_MAX_STEPS` ceiling. The injected message carries the
            # same unified status header as every other harness injection.
            if (
                nudge_enabled
                and searches_done_now < settings.REACT_MIN_SEARCH_CALLS
                and nudges_used < settings.REACT_MAX_NUDGES
                and step < settings.REACT_MAX_STEPS - 1
            ):
                messages.append(
                    {
                        "role": "user",
                        "content": _build_nudge_message(
                            current_step=steps_executed,
                            max_steps=settings.REACT_MAX_STEPS,
                            searches_done=searches_done_now,
                            max_search_calls=effective_max_search_calls,
                            min_required=settings.REACT_MIN_SEARCH_CALLS,
                        ),
                    }
                )
                nudges_used += 1
                continue
            # Exit signal is a parseable `\boxed{...}`, NOT "no tool_calls".
            # `final_raw` is always updated so the natural-end fallback can
            # surface the most recent content turn even when no `\boxed{...}`
            # ever appeared. The hard step ceiling guarantees termination
            # even on chains of unboxed content turns.
            final_raw = content
            if parse_answer(content, q) is not None:
                break
            # No parseable boxed → defer the continuation message to the next
            # iteration's pre-step injection so it cannot collide with a
            # higher-priority status injection (last-step / penultimate /
            # budget-exhausted). On the FINAL iteration there is no next
            # turn, so the flag would just be discarded (the loop ends and
            # `final_raw` already holds whatever content we got).
            pending_continuation = True

        for tc in tool_calls:
            tc_id = tc.get("id") or ""
            fn = tc.get("function") or {}
            fn_name = fn.get("name")
            raw_args = fn.get("arguments")
            # Live budget snapshot attached to every tool_error this turn so
            # the model sees its remaining budget INSIDE the assistant→tool
            # cycle (where we cannot inject a user message). `searches_done`
            # uses the post-loop count so each successive error in the same
            # turn reflects any search that just ran.
            error_status = build_tool_error_status(
                current_step=steps_executed,
                max_steps=settings.REACT_MAX_STEPS,
                searches_done=len(search_calls),
                max_search_calls=effective_max_search_calls,
                # If THIS step is the last one (hard cutoff) we already have
                # tools=[] and the model isn't doing more tool calls. For all
                # earlier steps the next step strips tools iff the budget is
                # already exhausted (or about to be after this turn). We use
                # the simple "step+1 is the LAST step" heuristic plus the
                # current-budget check; either way the model is told its
                # search runway is gone.
                next_step_strips_tools=(
                    step + 1 >= settings.REACT_MAX_STEPS
                    or (
                        settings.REACT_BUDGET_EXCEEDED_DROP_TOOLS
                        and len(search_calls) >= settings.REACT_MAX_SEARCH_CALLS
                    )
                ),
            )

            if fn_name != "web_search":
                messages.append(
                    tool_error_message(
                        tc_id, f"unknown tool: {fn_name}", status=error_status
                    )
                )
                continue

            if len(search_calls) >= settings.REACT_MAX_SEARCH_CALLS:
                messages.append(
                    tool_error_message(
                        tc_id, "search budget exceeded", status=error_status
                    )
                )
                continue

            args, err = parse_tool_arguments(raw_args if isinstance(raw_args, str) else json.dumps(raw_args or {}))
            if err is not None or args is None:
                messages.append(
                    tool_error_message(
                        tc_id, err or "invalid arguments", status=error_status
                    )
                )
                continue

            query, qerr = extract_query(args)
            if qerr is not None or query is None:
                messages.append(
                    tool_error_message(
                        tc_id, qerr or "missing query", status=error_status
                    )
                )
                continue

            if search_semaphore is not None:
                await search_semaphore.acquire()
            try:
                result = await tavily_search(query, end_date, settings, client=httpx_client)
            finally:
                if search_semaphore is not None:
                    search_semaphore.release()

            _record_search_call(search_calls, query=query, end_date=end_date, result=result)
            messages.append(tool_result_message(tc_id, result.to_llm_payload()))

    # Loop exited either by break or by exhausting REACT_MAX_STEPS.
    # Harness-resilience D2: if the loop exited cleanly with an empty
    # `final_raw` (e.g. step budget consumed by tool_calls and the model never
    # produced a content turn), nudge once with `tools=[]` to force a
    # `\boxed{...}` answer. We do NOT increment `nudges_used` (semantically
    # different from "nudge to keep searching"); we DO add the call to
    # `react_steps` and `step_metrics` so the trace stays auditable.
    final_answer_retry_used = 0
    if final_raw == "" and settings.REACT_FINAL_ANSWER_RETRY:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Time to commit. Output your final \\boxed{...} answer now "
                    "without further searches or tool calls."
                ),
            }
        )
        t_retry_start = time.monotonic()
        retry_resp = await llm_chat(
            model=model,
            messages=messages,
            settings=settings,
            tools=[],
        )
        t_retry_ms = int((time.monotonic() - t_retry_start) * 1000)
        retry_msg = retry_resp.message
        # Skip belief parsing on the bail-out turn — we are explicitly asking
        # for a boxed answer, not a structured belief block.
        beliefs_per_step.append(None)
        _record_step(
            step_metrics,
            tokens,
            step=steps_executed,
            resp=retry_resp,
            latency_ms=t_retry_ms,
            belief=None,
        )
        last_resp = retry_resp
        messages.append(retry_msg)
        steps_executed += 1
        final_raw = retry_msg.get("content") or ""
        final_answer_retry_used = 1

    parsed = parse_answer(final_raw, q)
    correct = None
    if getattr(settings, "SCORE_ANSWERS", True):
        try:
            gt = parse_gt(q.answer)
        except ValueError:
            logger.error("question {} has invalid answer field: {!r}", q.id, q.answer)
            gt = frozenset()
        correct = is_correct(parsed, gt) if parsed is not None else None

    belief_final, belief_trace, belief_parse_ok = _finalize_belief_fields(
        belief_enabled=belief_enabled, beliefs_per_step=beliefs_per_step
    )

    return SampleResult(
        run_id=run_id,
        question_id=q.id,
        model=model,
        sample_idx=sample_idx,
        final_answer_letters=json.dumps(sorted(parsed)) if parsed is not None else None,
        final_answer_raw=final_raw,
        correct=int(correct) if isinstance(correct, bool) else None,
        parse_ok=1 if parsed is not None else 0,
        tool_calls_count=len(search_calls),
        react_steps=steps_executed,
        prompt_tokens=tokens["prompt"],
        completion_tokens=tokens["completion"],
        reasoning_tokens=tokens["reasoning"],
        latency_ms=int((time.monotonic() - t0) * 1000),
        user_prompt=user_prompt,
        messages_trace=json.dumps(messages, ensure_ascii=False) if settings.WRITE_MESSAGES_TRACE else None,
        search_calls=json.dumps(search_calls, ensure_ascii=False),
        error=None,
        created_at=utcnow_iso(),
        # Final-state envelope fields are taken from the LAST llm.chat response
        # so the recorded `finish_reason` reflects how the loop actually
        # terminated (stop / length / tool_calls). They are None when the loop
        # never ran (REACT_MAX_STEPS=0 — only reachable defensively).
        finish_reason=last_resp.finish_reason if last_resp is not None else None,
        nudges_used=nudges_used,
        step_metrics=json.dumps(step_metrics, ensure_ascii=False),
        response_id=last_resp.response_id if last_resp is not None else None,
        system_fingerprint=last_resp.system_fingerprint if last_resp is not None else None,
        service_tier=last_resp.service_tier if last_resp is not None else None,
        belief_final=belief_final,
        belief_trace=belief_trace,
        belief_parse_ok=belief_parse_ok,
        final_answer_retry_used=final_answer_retry_used,
    )


def _finalize_belief_fields(
    *,
    belief_enabled: bool,
    beliefs_per_step: list[Belief | None],
) -> tuple[str | None, str | None, int]:
    """Aggregate per-step Beliefs into the three persisted fields.

    Spec invariants (from the react-loop / answer-scoring spec):
    - Protocol disabled → all three are None / None / 0.
    - Last step's belief drives `belief_final` and `belief_parse_ok` (no
      "borrowing" from earlier successful parses if the final step failed).
    - `belief_trace` includes EVERY step (with None entries for failed ones)
      whenever at least one step parsed successfully; if every step failed
      it is None.
    """
    if not belief_enabled or not beliefs_per_step:
        return None, None, 0

    last_belief = beliefs_per_step[-1]
    if last_belief is not None:
        belief_parse_ok = 1
        belief_final = json.dumps(last_belief.probabilities, ensure_ascii=False)
    else:
        belief_parse_ok = 0
        belief_final = None

    if any(b is not None for b in beliefs_per_step):
        trace: list[dict[str, Any] | None] = [
            _belief_to_step_dict(idx, b) for idx, b in enumerate(beliefs_per_step)
        ]
        belief_trace = json.dumps(trace, ensure_ascii=False)
    else:
        belief_trace = None

    return belief_final, belief_trace, belief_parse_ok
