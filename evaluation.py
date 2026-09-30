#!/usr/bin/env python3
"""Forecast evaluation entry point.

Usage:
    python evaluation.py
    python evaluation.py --question-type yes_no
    python evaluation.py --question-type multiple_choice --choice-type multi
    python evaluation.py --skip-analysis                  # run only, no post-hoc stats

All runtime knobs live in `.env`; the CLI only filters questions.

Layout produced per run:
    {RUNS_ROOT}/{run_id}/
        manifest.json          # run-level metadata + per-model file map
        db/
            <model_slug>.db    # one SQLite per model (wide run_results table)
        analysis/              # CSV/MD/JSON written after the run finishes
        logs/
            {run_id}.log
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import signal
import sqlite3
import sys
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from forecast_eval import analysis
from forecast_eval import db as dbmod
from forecast_eval import leak_filter, loader, runner
from forecast_eval.config import Settings
from forecast_eval.errors import AuthError
from forecast_eval.llm import AuthError as LLMAuthError
from forecast_eval.prompts import BELIEF_PROTOCOL, REFLECTION_PROTOCOL
from forecast_eval.types import QFilter


def _compute_reflection_protocol(settings: Settings) -> tuple[str | None, str | None]:
    """Return `(text, hash16)` when reflection is enabled, else `(None, None)`.

    The hash is sha256 truncated to the first 16 hex chars — wide enough to
    distinguish meaningful protocol revisions, narrow enough to read in logs.
    Independent of `prompt_templates_hash` (DESIGN.md decision 2): the
    reflection protocol is appended outside the canonical template body, so
    runs with and without it stay comparable on `prompt_templates_hash` alone.
    """
    if not settings.REACT_REFLECTION_PROTOCOL:
        return None, None
    text = REFLECTION_PROTOCOL
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return text, digest


def _compute_belief_protocol(settings: Settings) -> tuple[str | None, str | None]:
    """Return `(text, hash16)` when belief protocol is enabled, else (None, None).

    Mirrors `_compute_reflection_protocol`: the belief protocol fingerprint is
    independent of `prompt_templates_hash` AND of `reflection_protocol_hash`,
    so two runs that differ only in `Settings.BELIEF_PROTOCOL` keep the same
    template hash but diverge here. Recorded both in `run_meta.belief_protocol_*`
    columns and at the manifest top level for grep-without-DB convenience.
    """
    if not settings.BELIEF_PROTOCOL:
        return None, None
    text = BELIEF_PROTOCOL
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return text, digest


QUESTION_TYPE_CHOICES = ("yes_no", "binary_named", "multiple_choice")
CHOICE_TYPE_CHOICES = ("single", "multi")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evaluation",
        description="Run LLM forecast evaluation against forecast_eval_set_example.db.",
    )
    parser.add_argument(
        "--question-type",
        action="append",
        choices=QUESTION_TYPE_CHOICES,
        help="Filter by question_type; pass multiple times to allow several.",
    )
    parser.add_argument(
        "--choice-type",
        action="append",
        choices=CHOICE_TYPE_CHOICES,
        help="Filter by choice_type; pass multiple times to allow several.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip the post-run statistics pass; raw DBs still land in db/.",
    )
    return parser.parse_args(argv)


def _configure_logging(log_file: Path, level: str) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_file,
        level="DEBUG",
        rotation="100 MB",
        retention=5,
    )


def _write_manifest(
    manifest_path: Path,
    *,
    run_id: str,
    settings: Settings,
    filters: QFilter,
    question_count: int,
    question_ids: list[str],
    virtual_models: list[str],
    real_models: list[str],
    model_files: dict[str, str],
    source_db_hash: str,
    metadata_hash: str,
    prompt_templates_hash: str,
    reflection_protocol_hash: str | None,
    belief_protocol_hash: str | None,
    started_at: str,
) -> None:
    # `reflection_protocol_hash` and `belief_protocol_hash` live at the top
    # level so users can grep manifest.json without opening any DB. The full
    # texts stay inside `run_meta.{reflection,belief}_protocol_text`.
    # `analysis_schema` lets the analysis layer dispatch on which metric
    # families to compute (probabilistic family vs accuracy-only fallback).
    #
    # `models` and `model_files` carry the *virtual* slug list so the
    # analysis main path (which iterates `manifest.models`) naturally walks
    # every grid cell. The top-level `grid` section captures the cartesian
    # skeleton (R / C lists + default anchors + real_models) for analysis/grid
    # to consume without opening per-cell .db files.
    r_list = list(settings.TAVILY_MAX_RESULTS)
    c_list = list(settings.REACT_MAX_SEARCH_CALLS)
    default_r = settings.GRID_DEFAULT_R if settings.GRID_DEFAULT_R is not None else r_list[0]
    default_c = settings.GRID_DEFAULT_C if settings.GRID_DEFAULT_C is not None else c_list[0]
    payload: dict[str, Any] = {
        "run_id": run_id,
        "schema_version": dbmod.SCHEMA_VERSION,
        "analysis_schema": "v4",
        "sampling_n": settings.SAMPLING_N,
        "models": virtual_models,
        "model_files": model_files,
        "model_profiles": {name: profile.model_dump(mode="json") for name, profile in settings.MODEL_PROFILES.items()},
        "inference_profiles_hash": dbmod.snapshot_settings(settings).get("inference_profiles_hash"),
        "model_training_cutoffs": {
            m: d.isoformat() for m, d in settings.MODEL_TRAINING_CUTOFFS.items()
        },
        "filters": {
            **filters.snapshot(),
            "question_count": question_count,
            "question_ids": question_ids,
        },
        "hashes": {
            "source_db": source_db_hash,
            "metadata": metadata_hash,
            "prompt_templates": prompt_templates_hash,
        },
        "reflection_protocol_hash": reflection_protocol_hash,
        "belief_protocol_hash": belief_protocol_hash,
        "grid": {
            "r_list": r_list,
            "c_list": c_list,
            "default_r": default_r,
            "default_c": default_c,
            "real_models": real_models,
            "n_cells": len(real_models) * len(r_list) * len(c_list),
        },
        "started_at": started_at,
        "finished_at": None,
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _finalise_manifest(manifest_path: Path, finished_at: str) -> None:
    if not manifest_path.exists():
        return
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    data["finished_at"] = finished_at
    manifest_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _init_model_db(
    *,
    db_path: Path,
    cell_settings: Settings,
    run_id: str,
    virtual_model: str,
    real_model: str,
    R: int,
    C: int,
    effective_min_search_calls: int,
    source_path: Path,
    filters: QFilter,
    filters_snapshot: dict[str, Any],
    source_db_hash: str,
    metadata_hash: str,
    prompt_templates_hash: str,
    reflection_protocol_text: str | None,
    reflection_protocol_hash: str | None,
    belief_protocol_text: str | None,
    belief_protocol_hash: str | None,
) -> tuple[sqlite3.Connection, dict[str, str], list]:
    """Initialise one .db file for a single grid cell.

    `cell_settings` is the dispatcher-derived per-cell sub-view: its
    TAVILY_MAX_RESULTS / REACT_MAX_SEARCH_CALLS are single ints (not lists).
    `snapshot_settings(cell_settings)` therefore writes single-value R/C into
    `config_snapshot`, so opening any one .db reveals exactly which grid cell
    it represents (DESIGN.md decision 5).

    `grid_origin` is recorded at the top level of the persisted config snapshot
    via `register_run_meta` — auditors can read it without re-deriving from
    the virtual slug.

    `training_cutoff` is keyed by the *real* model slug (not the virtual one);
    callers parse the slug and pass it explicitly.
    """
    conn = dbmod.connect(db_path)
    dbmod.init_schema(conn, cell_settings.SAMPLING_N)
    cutoff = cell_settings.MODEL_TRAINING_CUTOFFS.get(real_model)
    config_snapshot = dbmod.snapshot_settings(cell_settings)
    # search-leak-filter-v1: detector fingerprint triplet (enabled / model /
    # prompt_hash) into config_snapshot. Same injection pattern as
    # `grid_origin` below — keeps `db.register_run_meta` signature unchanged
    # and `db.snapshot_settings` pure (Settings → dict). Disabled runs still
    # write the three keys with default values so the JSON shape stays
    # consistent across runs.
    leak_enabled = bool(cell_settings.ENABLE_SEARCH_LEAK_FILTER)
    config_snapshot = {
        **config_snapshot,
        "leak_detector_enabled": leak_enabled,
        "leak_detector_model": cell_settings.LEAK_DETECTOR_MODEL if leak_enabled else "",
        "leak_detector_prompt_hash": (
            leak_filter._compute_prompt_hash() if leak_enabled else ""
        ),
    }
    existing = conn.execute("SELECT * FROM run_meta").fetchone()
    if existing is not None:
        expected = {
            "run_id": run_id, "model": virtual_model, "sampling_n": cell_settings.SAMPLING_N,
            "source_db_hash": source_db_hash, "metadata_hash": metadata_hash,
            "prompt_templates_hash": prompt_templates_hash,
            "reflection_protocol_hash": reflection_protocol_hash,
            "belief_protocol_hash": belief_protocol_hash,
        }
        mismatch = [key for key, value in expected.items() if existing[key] != value]
        if dbmod.collection_contract(json.loads(existing["config_snapshot"])) != dbmod.collection_contract(config_snapshot):
            mismatch.append("collection_contract")
        if json.loads(existing["filters_snapshot"]) != filters_snapshot:
            mismatch.append("filters_snapshot")
        if mismatch:
            conn.close()
            raise ValueError(f"resume would mix different contracts: {', '.join(mismatch)}")
    templates = loader.sync_prompt_templates(source_path, conn, template_style=cell_settings.PROMPT_TEMPLATE_STYLE)
    questions = loader.sync_questions(source_path, conn, filters, table=cell_settings.SOURCE_TABLE)
    grid_origin = {
        "real_model": real_model,
        "R": R,
        "C": C,
        "effective_min_search_calls": effective_min_search_calls,
    }
    dbmod.register_run_meta(
        conn,
        run_id=run_id,
        model=virtual_model,
        sampling_n=cell_settings.SAMPLING_N,
        filters_snapshot=filters_snapshot,
        config_snapshot=config_snapshot,
        source_db_hash=source_db_hash,
        metadata_hash=metadata_hash,
        prompt_templates_hash=prompt_templates_hash,
        training_cutoff=cutoff.isoformat() if cutoff else None,
        reflection_protocol_text=reflection_protocol_text,
        reflection_protocol_hash=reflection_protocol_hash,
        belief_protocol_text=belief_protocol_text,
        belief_protocol_hash=belief_protocol_hash,
        grid_origin=grid_origin,
    )
    return conn, templates, questions


def _make_settings_factory(
    global_settings: Settings,
) -> Callable[[str, int, int], Settings]:
    """Return a factory mapping `(virtual_slug, R, C)` -> cell-local sub-view.

    Each sub-view downcasts TAVILY_MAX_RESULTS / REACT_MAX_SEARCH_CALLS to the
    cell's single int and silently clamps REACT_MIN_SEARCH_CALLS to
    `min(global_min, C)` (DESIGN.md decision 4). The original `global_settings`
    instance is not mutated; pydantic-settings `model_copy(update=...)` returns
    a fresh immutable copy each time. Results are memoised on `(slug, R, C)`
    so repeated calls in the same run share a single Settings object.
    """
    cache: dict[tuple[str, int, int], Settings] = {}
    global_min = global_settings.REACT_MIN_SEARCH_CALLS

    def factory(slug: str, R: int, C: int) -> Settings:
        key = (slug, int(R), int(C))
        if key in cache:
            return cache[key]
        effective_min = min(global_min, C)
        view = global_settings.model_copy(
            update={
                "TAVILY_MAX_RESULTS": R,
                "REACT_MAX_SEARCH_CALLS": C,
                "REACT_MIN_SEARCH_CALLS": effective_min,
            }
        )
        cache[key] = view
        return view

    return factory


async def _run_async(
    settings: Settings,
    filters: QFilter,
    run_id: str,
    run_dir: Path,
    skip_analysis: bool,
) -> int:
    source_path = settings.source_db_path()
    if not source_path.exists():
        logger.error("source DB not found at {}", source_path)
        return 2

    db_dir = run_dir / "db"
    analysis_dir = run_dir / "analysis"
    logs_dir = run_dir / "logs"
    for d in (db_dir, analysis_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Peek at questions once to compute filters_snapshot + question_count. We
    # use a scratch in-memory DB so the loader logic stays unchanged; each
    # per-model DB below will re-sync the questions into its own file.
    scratch = dbmod.connect(":memory:")
    dbmod.init_schema(scratch, settings.SAMPLING_N)
    templates_preview = loader.sync_prompt_templates(source_path, scratch, template_style=settings.PROMPT_TEMPLATE_STYLE)
    questions_preview = loader.sync_questions(source_path, scratch, filters, table=settings.SOURCE_TABLE)
    scratch.close()

    if not questions_preview:
        logger.error("no questions matched the filter; aborting")
        return 3

    raw_features = loader.load_raw_features_json(source_path)
    source_hash = dbmod.compute_source_db_hash(source_path)
    metadata_hash = dbmod.compute_metadata_hash(raw_features)
    templates_hash = dbmod.compute_prompt_templates_hash(templates_preview)
    reflection_text, reflection_hash = _compute_reflection_protocol(settings)
    belief_text, belief_hash = _compute_belief_protocol(settings)

    filters_snapshot: dict[str, Any] = {
        **filters.snapshot(),
        "question_count": len(questions_preview),
        "question_ids": [q.id for q in questions_preview],
    }
    started_at = dbmod.utcnow_iso()

    # Cartesian-expand (real_model, R, C) -> virtual slug list. Single-value
    # R / C in .env (length-1 lists) reduce this to one virtual slug per model,
    # which is exactly the single-cell behavior with a longer .db filename.
    real_models = list(settings.MODELS)
    r_list = list(settings.TAVILY_MAX_RESULTS)
    c_list = list(settings.REACT_MAX_SEARCH_CALLS)
    global_min = settings.REACT_MIN_SEARCH_CALLS
    cell_index: dict[str, tuple[str, int, int, int]] = {}
    virtual_models: list[str] = []
    for real, R, C in itertools.product(real_models, r_list, c_list):
        slug = dbmod.compose_virtual_slug(real, R, C)
        effective_min = min(global_min, C)
        cell_index[slug] = (real, R, C, effective_min)
        virtual_models.append(slug)

    settings_factory = _make_settings_factory(settings)

    model_files: dict[str, str] = {}
    conns: dict[str, sqlite3.Connection] = {}
    templates_by_model: dict[str, dict[str, str]] = {}
    questions_by_model: dict[str, list] = {}

    try:
        for virtual_model in virtual_models:
            real_model, R, C, effective_min = cell_index[virtual_model]
            slug_safe = dbmod.model_slug_safe(virtual_model)
            db_path = db_dir / f"{slug_safe}.db"
            model_files[virtual_model] = db_path.name
            cell_settings = settings_factory(virtual_model, R, C)
            conn, templates, questions = _init_model_db(
                db_path=db_path,
                cell_settings=cell_settings,
                run_id=run_id,
                virtual_model=virtual_model,
                real_model=real_model,
                R=R,
                C=C,
                effective_min_search_calls=effective_min,
                source_path=source_path,
                filters=filters,
                filters_snapshot=filters_snapshot,
                source_db_hash=source_hash,
                metadata_hash=metadata_hash,
                prompt_templates_hash=templates_hash,
                reflection_protocol_text=reflection_text,
                reflection_protocol_hash=reflection_hash,
                belief_protocol_text=belief_text,
                belief_protocol_hash=belief_hash,
            )
            conns[virtual_model] = conn
            templates_by_model[virtual_model] = templates
            questions_by_model[virtual_model] = questions

        manifest_path = run_dir / "manifest.json"
        _write_manifest(
            manifest_path,
            run_id=run_id,
            settings=settings,
            filters=filters,
            question_count=len(questions_preview),
            question_ids=[q.id for q in questions_preview],
            virtual_models=virtual_models,
            real_models=real_models,
            model_files=model_files,
            source_db_hash=source_hash,
            metadata_hash=metadata_hash,
            prompt_templates_hash=templates_hash,
            reflection_protocol_hash=reflection_hash,
            belief_protocol_hash=belief_hash,
            started_at=started_at,
        )

        primary_model = virtual_models[0]
        templates = templates_by_model[primary_model]
        questions = questions_by_model[primary_model]

        # The runner iterates `settings.MODELS` to build the task plan. To
        # preserve that contract while expanding to virtual slugs, hand it a
        # cell-local view of settings whose MODELS field is the virtual list.
        runner_settings = settings.model_copy(update={"MODELS": virtual_models})

        try:
            stats = await runner.run(
                settings=runner_settings,
                filters=filters,
                questions=questions,
                templates=templates,
                run_id=run_id,
                conns=conns,
                settings_factory=settings_factory,
            )
        except (AuthError, LLMAuthError) as e:
            logger.error("AUTH error, aborting run: {}", e)
            return 4

        if stats.aborted:
            state = json.loads(manifest_path.read_text())
            state["status"] = "blocked"
            state["block_reason"] = stats.abort_reason
            manifest_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))
            logger.error("[run={}] incomplete; resume after resolving the blocker", run_id)
            return 4
        if stats.deferred:
            state = json.loads(manifest_path.read_text())
            state.update(status="partial", remaining_samples=stats.deferred)
            manifest_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))
            return 6
        finished_at = dbmod.utcnow_iso()
        _finalise_manifest(manifest_path, finished_at)

        logger.info(
            "[run={}] finished. total={} done={} skipped_cutoff={} errors={}",
            run_id,
            stats.total,
            stats.done,
            stats.skipped_cutoff,
            stats.errors,
        )
    finally:
        for conn in conns.values():
            try:
                conn.close()
            except Exception:  # noqa: BLE001 — best-effort close
                logger.exception("failed to close a model DB connection")

    if skip_analysis:
        logger.info("[run={}] --skip-analysis: analysis/ not generated", run_id)
    else:
        try:
            written = analysis.run_analysis(
                run_dir,
                # Pass through the difficulty-weighted composite settings from
                # .env to the analysis layer; the analysis module does not
                # read .env itself, evaluation supplies them.
                composite_weights=settings.COMPOSITE_WEIGHTS,
                composite_overrides=settings.COMPOSITE_WEIGHT_OVERRIDES,
            )
            logger.info(
                "[run={}] analysis written: {}",
                run_id,
                ", ".join(p.name for p in written),
            )
        except Exception:  # noqa: BLE001 — analysis must never fail the run
            logger.exception("analysis pass failed; raw DBs are intact")

    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        settings = Settings()
    except Exception as exc:
        print(f"[evaluation] failed to load .env: {exc}", file=sys.stderr)
        return 2

    filters = QFilter(
        question_types=frozenset(args.question_type) if args.question_type else None,
        choice_types=frozenset(args.choice_type) if args.choice_type else None,
    )

    run_id = settings.RUN_ID or runner.generate_run_id()
    run_dir = settings.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(run_dir / "logs" / f"{run_id}.log", settings.LOG_LEVEL)
    logger.info("run_id={} dir={} filters={}", run_id, run_dir, filters.snapshot())

    def _handle_sigint(signum: int, frame: Any) -> None:  # noqa: ANN401
        logger.warning("SIGINT received; attempting graceful shutdown...")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        return asyncio.run(
            _run_async(settings, filters, run_id, run_dir, args.skip_analysis)
        )
    except KeyboardInterrupt:
        logger.warning("interrupted; some rows may not have been persisted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
