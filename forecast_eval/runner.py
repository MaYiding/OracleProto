"""Top-level orchestration for one evaluation run.

Contracts:
- `evaluation.py` prepares per-model SQLite connections under
  `RUNS_ROOT/{run_id}/db/` and hands them to `run()`.
- This module owns the task queue, the async writers (one per model), the
  cutoff-filter + resume logic, and the progress log.
- The DB layer stores raw observations only; statistics are computed post-hoc
  by `forecast_eval.analysis`.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable

from loguru import logger

from . import db as dbmod
from .config import Settings
from .db import AsyncWriter, utcnow_iso
from .errors import AuthError, CollectionBlockedError, ErrorKind, classify
from .llm import AuthError as _LLMAuthError  # noqa: F401 — re-exported for callers
from .react import run_react
from .types import QFilter, Question, SampleResult


@dataclass
class Task:
    question: Question
    model: str  # virtual slug `{real}::r{R}::c{C}` in grid runs, plain real slug otherwise
    sample_idx: int
    # Cell-local Settings sub-view: TAVILY_MAX_RESULTS / REACT_MAX_SEARCH_CALLS
    # are downcast to single ints by the dispatcher so `run_react` and
    # `tavily_search` see the right (R, C) per task without contextvars or
    # closures. See `evaluation.py::_make_settings_factory`.
    settings: Settings


@dataclass
class RunStats:
    total: int = 0
    completed_preexisting: int = 0
    skipped_cutoff: int = 0
    planned: int = 0
    done: int = 0
    errors: dict[str, int] = field(default_factory=dict)
    aborted: bool = False
    abort_reason: str | None = None
    deferred: int = 0


def generate_run_id(now: datetime | None = None) -> str:
    ts = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:4]}"


def _skipped_cutoff_row(q: Question, sample_idx: int) -> dict[str, Any]:
    # `nudges_used=0` is meaningful (no LLM call happened), the other
    # observability fields are None. Belief fields are None / None / 0
    # because no assistant message exists to parse a belief block from.
    return SampleResult(
        run_id="",
        question_id=q.id,
        model="",
        sample_idx=sample_idx,
        final_answer_letters=None,
        final_answer_raw=None,
        correct=None,
        parse_ok=0,
        tool_calls_count=0,
        react_steps=0,
        prompt_tokens=0,
        completion_tokens=0,
        reasoning_tokens=0,
        latency_ms=0,
        user_prompt=None,
        messages_trace=None,
        search_calls=None,
        error="skipped_training_cutoff",
        created_at=utcnow_iso(),
        finish_reason=None,
        nudges_used=0,
        step_metrics=None,
        response_id=None,
        system_fingerprint=None,
        service_tier=None,
        belief_final=None,
        belief_trace=None,
        belief_parse_ok=0,
        final_answer_retry_used=0,
    ).to_row()


def _error_row(q: Question, sample_idx: int, error: str) -> dict[str, Any]:
    # Same shape as the cutoff row: nudges_used=0 because no react-loop counter
    # ever ticked, and the observability + belief fields are None / 0 — no LLM
    # response means no belief block to parse either.
    return SampleResult(
        run_id="",
        question_id=q.id,
        model="",
        sample_idx=sample_idx,
        final_answer_letters=None,
        final_answer_raw=None,
        correct=None,
        parse_ok=0,
        tool_calls_count=0,
        react_steps=0,
        prompt_tokens=0,
        completion_tokens=0,
        reasoning_tokens=0,
        latency_ms=0,
        user_prompt=None,
        messages_trace=None,
        search_calls=None,
        error=error,
        created_at=utcnow_iso(),
        finish_reason=None,
        nudges_used=0,
        step_metrics=None,
        response_id=None,
        system_fingerprint=None,
        service_tier=None,
        belief_final=None,
        belief_trace=None,
        belief_parse_ok=0,
        final_answer_retry_used=0,
    ).to_row()


def build_task_plan(
    *,
    questions: list[Question],
    settings: Settings,
    completed: dict[str, set[tuple[str, int]]],
    run_id: str,
    settings_factory: Callable[[str, int, int], Settings] | None = None,
) -> tuple[list[Task], dict[str, list[dict[str, Any]]], RunStats]:
    """Expand (questions × models × samples), drop resumed cells, then split:
       - `todo`: LLM work to dispatch (per-model writers consume this)
       - `cutoff_rows`: model -> list of pre-seeded rows marked
         `error="skipped_training_cutoff"` (not counted as LLM work)
       - `stats`: counters for progress logging

    Resume takes precedence over cutoff filtering — a cell already completed
    must never be re-emitted as skipped_training_cutoff.

    `settings_factory(virtual_slug, R, C) -> Settings` returns the cell-local
    Settings sub-view; the dispatcher in evaluation.py supplies it. When None
    (legacy callers / tests), each task inherits the global `settings`
    unchanged. Cutoff lookups always go through the real model name (parsed
    out of virtual slug when applicable) so `MODEL_TRAINING_CUTOFFS` keeps the
    human-friendly real slug as key.
    """
    stats = RunStats()
    todo: list[Task] = []
    cutoff_rows: dict[str, list[dict[str, Any]]] = {m: [] for m in settings.MODELS}

    def _resolve_settings(slug: str) -> Settings:
        if settings_factory is None:
            return settings
        parsed = dbmod.parse_virtual_slug(slug)
        if parsed is None:
            # Non-virtual slug: fall back to globals' first-element R/C so
            # legacy single-cell callers still get a coherent sub-view.
            R = int(settings.TAVILY_MAX_RESULTS[0]) if settings.TAVILY_MAX_RESULTS else 0
            C = int(settings.REACT_MAX_SEARCH_CALLS[0]) if settings.REACT_MAX_SEARCH_CALLS else 0
            return settings_factory(slug, R, C)
        _real, R, C = parsed
        return settings_factory(slug, R, C)

    scoped_ids = {qid for ids in (getattr(settings, "MODEL_QUESTION_IDS", {}) or {}).values() for qid in ids}
    for q in sorted(questions, key=lambda question: question.id not in scoped_ids):
        q_end = date.fromisoformat(q.end_time)
        for model in settings.MODELS:
            # `model` may be a virtual slug `{real}::r{R}::c{C}` — peel the real
            # part off for cutoff lookup; falls back to `model` when not virtual.
            parsed = dbmod.parse_virtual_slug(model)
            real_model = parsed[0] if parsed is not None else model
            allowed = (getattr(settings, "MODEL_QUESTION_IDS", {}) or {}).get(real_model)
            if allowed is not None and q.id not in allowed:
                continue
            cutoff = settings.MODEL_TRAINING_CUTOFFS.get(real_model)
            is_cutoff_hit = cutoff is not None and q_end <= cutoff
            done_for_model = completed.get(model, set())
            cell_settings = _resolve_settings(model)
            sample_indices = getattr(settings, "MODEL_SAMPLE_INDICES", {}).get(real_model, {}).get(q.id)
            for s in range(settings.SAMPLING_N):
                if sample_indices is not None and s not in sample_indices:
                    continue
                stats.total += 1
                key = (q.id, s)
                if key in done_for_model:
                    stats.completed_preexisting += 1
                    continue
                if is_cutoff_hit:
                    cutoff_rows[model].append(_skipped_cutoff_row(q, s))
                    stats.skipped_cutoff += 1
                    continue
                todo.append(
                    Task(question=q, model=model, sample_idx=s, settings=cell_settings)
                )
                stats.planned += 1

    return todo, cutoff_rows, stats


async def _run_task_with_retry(
    task: Task,
    *,
    _global_settings: Settings,  # kept for type / signal use; per-task config is read from task.settings
    templates: dict[str, str],
    run_id: str,
    llm_semaphore: asyncio.Semaphore,
    search_semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Execute one sample; classify any terminal exception into an error row.

    AUTH errors re-raise so the caller aborts the whole run; everything else
    produces a row with `error=...` the writer can still land.

    All cell-local config (TAVILY_MAX_RESULTS / REACT_MAX_SEARCH_CALLS / etc.)
    is read from `task.settings` — the dispatcher's per-cell sub-view. The
    `_global_settings` parameter is unused for per-task logic; we keep it as a
    typed channel for future global-only signals (semaphore sizes are already
    materialised by the caller).
    """
    # task.model carries the virtual slug `{real}::r{R}::c{C}` for grid runs;
    # the upstream LLM provider only knows the real model name, so we strip
    # the (R, C) suffix here before dispatching the API call. Non-virtual slugs
    # (legacy single-cell callers / tests) pass through unchanged.
    parsed_slug = dbmod.parse_virtual_slug(task.model)
    real_model = parsed_slug[0] if parsed_slug is not None else task.model
    try:
        async with llm_semaphore:
            result = await run_react(
                task.question,
                model=real_model,
                sample_idx=task.sample_idx,
                settings=task.settings,
                templates=templates,
                run_id=run_id,
                search_semaphore=search_semaphore,
            )
        return result.to_row()
    except AuthError:
        raise
    except _LLMAuthError:
        raise
    except (asyncio.CancelledError, CollectionBlockedError):
        raise
    except Exception as exc:
        kind = classify(exc)
        if kind is ErrorKind.AUTH:
            raise AuthError(str(exc)) from exc
        error_str = str(kind) if kind is not ErrorKind.UNKNOWN else "unknown"
        logger.exception(
            "sample failed q={} model={} sample={} kind={}",
            task.question.id,
            task.model,
            task.sample_idx,
            kind,
        )
        return _error_row(task.question, task.sample_idx, error_str)


def _log_progress(
    *,
    run_id: str,
    done: int,
    total: int,
    sampling_n: int,
    task: Task,
    row: dict[str, Any],
) -> None:
    error = row.get("error")
    q = task.question
    if error:
        logger.error(
            "[run={}] [{}/{}] q={} qt={} ct={} model={} sample={}/{} error={} retry_exhausted",
            run_id,
            done,
            total,
            q.id,
            q.question_type,
            q.choice_type,
            task.model,
            task.sample_idx + 1,
            sampling_n,
            error,
        )
    else:
        logger.info(
            "[run={}] [{}/{}] q={} qt={} ct={} model={} sample={}/{} correct={} parse_ok={} steps={} tool_calls={} latency={}ms retry={}",
            run_id,
            done,
            total,
            q.id,
            q.question_type,
            q.choice_type,
            task.model,
            task.sample_idx + 1,
            sampling_n,
            row.get("correct"),
            row.get("parse_ok"),
            row.get("react_steps"),
            row.get("tool_calls_count"),
            row.get("latency_ms"),
            row.get("final_answer_retry_used"),
        )


async def run(
    *,
    settings: Settings,
    filters: QFilter,
    questions: list[Question],
    templates: dict[str, str],
    run_id: str,
    conns: dict[str, sqlite3.Connection],
    settings_factory: Callable[[str, int, int], Settings] | None = None,
) -> RunStats:
    """Orchestrate one run across all configured models.

    Caller responsibility (see `evaluation.py`):
      * create `RUNS_ROOT/{run_id}/{db,analysis,logs}/`
      * open one sqlite connection per model under db/
      * run `db.init_schema(conn, SAMPLING_N)`
      * run `loader.sync_questions(...)` + `loader.sync_prompt_templates(...)` per DB
      * run `db.register_run_meta(...)` per DB

    This function then:
      * loads per-model resume sets
      * plans the task list + cutoff rows (one list per model), each Task
        carrying a cell-local settings sub-view from `settings_factory`
      * spawns one `AsyncWriter` per model
      * drives the ReAct loop across `LLM_MAX_CONCURRENCY` workers
      * finishes each model's `run_meta` row on clean exit (or leaves it open
        if aborted by AUTH)

    `settings_factory(slug) -> Settings` lets the dispatcher inject per-cell
    R / C as cell-local sub-views. When omitted (legacy callers), each Task
    inherits the global `settings` unchanged, preserving v4 byte-level behavior.
    """
    sampling_n = settings.SAMPLING_N
    models = list(conns.keys())

    # Per-model resume set
    completed: dict[str, set[tuple[str, int]]] = {
        m: dbmod.load_completed_samples(conns[m], sampling_n,
            retain_model_refusals=settings.COLLECTION_RETAIN_MODEL_REFUSALS) for m in models
    }
    for model, conn in conns.items():
        name = (dbmod.parse_virtual_slug(model) or (model,))[0]
        target_meta = conn.execute("SELECT * FROM run_meta").fetchone()
        for reference_path in settings.COLLECTION_REFERENCE_DBS.get(name, []):
            path = Path(reference_path).resolve()
            if not path.is_relative_to(Path(settings.RUNS_ROOT).resolve()):
                raise ValueError("reference DB must remain inside RUNS_ROOT")
            source = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
            source.row_factory = sqlite3.Row
            try:
                meta = source.execute("SELECT * FROM run_meta").fetchone()
                for key in ("model", "sampling_n", "source_db_hash", "metadata_hash", "prompt_templates_hash", "reflection_protocol_hash", "belief_protocol_hash", "filters_snapshot"):
                    if meta[key] != target_meta[key]:
                        raise ValueError(f"reference DB differs in {key}")
                before = dbmod.collection_contract(json.loads(meta["config_snapshot"]))
                after = dbmod.collection_contract(json.loads(target_meta["config_snapshot"]))
                # Detector request controls retain separate source strata.
                for contract in (before, after):
                    contract.pop("LEAK_DETECTOR_RESPONSE_FORMAT", None)
                source_cap = before.get("LEAK_DETECTOR_MAX_TOKENS")
                target_cap = after.get("LEAK_DETECTOR_MAX_TOKENS")
                if source_cap is not None and target_cap is not None and source_cap <= target_cap:
                    for contract in (before, after):
                        contract.pop("LEAK_DETECTOR_MAX_TOKENS", None)
                if after.get("LEAK_DETECTOR_DROP_CONTENT_POLICY"):
                    for contract in (before, after):
                        contract.pop("LEAK_DETECTOR_DROP_CONTENT_POLICY", None)
                if before != after:
                    raise ValueError("reference DB inference contract differs")
                completed[model].update(dbmod.load_completed_samples(source, sampling_n,
                    retain_model_refusals=settings.COLLECTION_RETAIN_MODEL_REFUSALS))
            finally:
                source.close()
    todo, cutoff_rows, stats = build_task_plan(
        questions=questions,
        settings=settings,
        completed=completed,
        run_id=run_id,
        settings_factory=settings_factory,
    )
    total_pending = len(todo)
    if settings.COLLECTION_MODEL:
        todo = [task for task in todo if
                (dbmod.parse_virtual_slug(task.model) or (task.model,))[0] == settings.COLLECTION_MODEL]
    if settings.COLLECTION_SAMPLE_LIMIT:
        todo = todo[:settings.COLLECTION_SAMPLE_LIMIT]
    stats.deferred = total_pending - len(todo)
    stats.planned = len(todo)

    logger.info(
        "[run={}] plan: total={} already_done={} skipped_cutoff={} to_run={}",
        run_id,
        stats.total,
        stats.completed_preexisting,
        stats.skipped_cutoff,
        stats.planned,
    )

    writers: dict[str, AsyncWriter] = {
        m: AsyncWriter(conns[m], sampling_n=sampling_n, batch=settings.DB_COMMIT_BATCH)
        for m in models
    }
    for w in writers.values():
        await w.start()

    llm_sem = asyncio.Semaphore(settings.LLM_MAX_CONCURRENCY)
    search_sem = asyncio.Semaphore(settings.SEARCH_MAX_CONCURRENCY)
    audit_conns = {}
    for model, conn in conns.items():
        path = conn.execute("PRAGMA database_list").fetchone()[2]
        audit_conns[model] = dbmod.connect(path) if path else conn

    # Cutoff rows are enqueued first. They flow through each model's writer and
    # inflate the [done/total] denominator in a predictable way.
    for model, rows in cutoff_rows.items():
        w = writers[model]
        for row in rows:
            await w.enqueue_result(row)

    done_counter = stats.completed_preexisting + stats.skipped_cutoff
    aborted = False

    async def _worker(task: Task) -> None:
        nonlocal done_counter
        with dbmod.capture_requests(
            audit_conns[task.model], settings=task.settings, run_id=run_id,
            model=task.model, question_id=task.question.id, sample_idx=task.sample_idx,
        ):
            row = await _run_task_with_retry(
                task, _global_settings=settings, templates=templates, run_id=run_id,
                llm_semaphore=llm_sem, search_semaphore=search_sem,
            )
            dbmod.audit_event("sample.result", row)
        await writers[task.model].enqueue_result(row)
        done_counter += 1
        kind = row.get("error")
        if kind:
            stats.errors[kind] = stats.errors.get(kind, 0) + 1
            retained_refusal = settings.COLLECTION_RETAIN_MODEL_REFUSALS and kind == ErrorKind.CONTENT_POLICY
            if settings.REQUIRE_HEALTHY_RETRIEVAL and not retained_refusal:
                raise CollectionBlockedError(f"sample failed: {kind}; inspect saved request events before resuming")
        _log_progress(
            run_id=run_id,
            done=done_counter,
            total=stats.total,
            sampling_n=sampling_n,
            task=task,
            row=row,
        )

    worker_tasks: list[asyncio.Task] = []
    try:
        for t in todo:
            worker_tasks.append(asyncio.create_task(_worker(t)))
        for fut in asyncio.as_completed(worker_tasks):
            try:
                await fut
            except (AuthError, CollectionBlockedError) as exc:
                logger.error("[run={}] collection blocked: {}", run_id, exc)
                aborted = True
                stats.abort_reason = str(exc)
                for t in worker_tasks:
                    t.cancel()
                break
    finally:
        for task in worker_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*worker_tasks, return_exceptions=True)
        for w in writers.values():
            await w.drain()
        for w in writers.values():
            await w.close()
        for model, conn in audit_conns.items():
            if conn is not conns[model]:
                conn.close()
        if not aborted and not stats.deferred:
            for m, conn in conns.items():
                dbmod.finish_run_meta(conn, run_id)

    stats.done = done_counter
    stats.aborted = aborted
    return stats
