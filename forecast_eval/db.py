"""Per-model SQLite persistence layer.

Each evaluation run owns a directory `RUNS_ROOT/{run_id}/`; inside `db/` we
write ONE SQLite file per model. Every model DB is self-contained: it has a
copy of the source questions, the prompt templates, a single `run_meta` row,
and a WIDE `run_results` table with one row per question and one column group
`s{i}_*` per sample index `0..SAMPLING_N-1`.

The DB stores raw observations only. Statistics (pass@1, pass_any@N, majority
vote, etc.) are computed post-hoc by `forecast_eval.analysis` and written to
`RUNS_ROOT/{run_id}/analysis/`.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import Settings
from .errors import CollectionBlockedError


SCHEMA_VERSION = 5


_audit_context: ContextVar[dict[str, Any] | None] = ContextVar("request_audit", default=None)


@contextmanager
def capture_requests(conn: sqlite3.Connection, *, settings: Settings, run_id: str,
                     model: str, question_id: str, sample_idx: int):
    """Bind concurrent HTTP calls to one durable sample attempt."""
    context = None
    if getattr(settings, "WRITE_REQUEST_AUDIT", True):
        context = dict(conn=conn, run_id=run_id, model=model, question_id=question_id,
                       sample_idx=sample_idx, attempt_id=uuid.uuid4().hex,
                       secrets=[settings.LLM_API_KEY, settings.LEAK_DETECTOR_API_KEY,
                                *settings.TAVILY_API_KEY])
    token = _audit_context.set(context)
    try:
        yield
    finally:
        _audit_context.reset(token)


def audit_event(kind: str, payload: Any, *, request_id: str | None = None) -> str:
    """Append an event immediately, including failed and interrupted attempts."""
    request_id = request_id or uuid.uuid4().hex
    context = _audit_context.get()
    if context is None:
        return request_id
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    encoded = json.dumps(payload, ensure_ascii=False, default=str)
    for secret in context["secrets"]:
        if secret:
            encoded = encoded.replace(secret, "<redacted>")
    try:
        context["conn"].execute(
            "INSERT INTO request_events (run_id, model, question_id, sample_idx, "
            "attempt_id, request_id, kind, payload, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (context["run_id"], context["model"], context["question_id"],
             context["sample_idx"], context["attempt_id"], request_id, kind, encoded, utcnow_iso()),
        )
    except sqlite3.Error as exc:
        raise CollectionBlockedError("request audit could not be persisted") from exc
    return request_id


# ---------- Per-sample column definitions ----------

# Each sample_idx contributes this column suite (prefixed with "s{i}_").
# v3 (2026-04) appended 6 observability columns at the tail; v4 (2026-04)
# appends 3 belief columns; v5 (2026-04, harness-resilience) appends 1
# `final_answer_retry_used` indicator. Keep the order stable: existing code
# assumes new fields land after `created_at`, and the migration path adds them
# with `ALTER TABLE ADD COLUMN` in this same order.
PER_SAMPLE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("final_answer_letters", "TEXT"),
    ("final_answer_raw", "TEXT"),
    ("correct", "INTEGER"),
    ("parse_ok", "INTEGER"),
    ("tool_calls_count", "INTEGER"),
    ("react_steps", "INTEGER"),
    ("prompt_tokens", "INTEGER"),
    ("completion_tokens", "INTEGER"),
    ("reasoning_tokens", "INTEGER"),
    ("latency_ms", "INTEGER"),
    ("messages_trace", "TEXT"),
    ("search_calls", "TEXT"),
    ("error", "TEXT"),
    ("created_at", "TEXT"),
    ("finish_reason", "TEXT"),
    ("nudges_used", "INTEGER"),
    ("step_metrics", "TEXT"),
    ("response_id", "TEXT"),
    ("system_fingerprint", "TEXT"),
    ("service_tier", "TEXT"),
    ("belief_final", "TEXT"),
    ("belief_trace", "TEXT"),
    ("belief_parse_ok", "INTEGER"),
    ("final_answer_retry_used", "INTEGER"),
)
PER_SAMPLE_FIELD_NAMES: tuple[str, ...] = tuple(name for name, _ in PER_SAMPLE_COLUMNS)

# Columns added in v3 (used by the migration path). Order matters because the
# ALTER statements run in this sequence on each sample's column group.
_V3_NEW_PER_SAMPLE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("finish_reason", "TEXT"),
    ("nudges_used", "INTEGER"),
    ("step_metrics", "TEXT"),
    ("response_id", "TEXT"),
    ("system_fingerprint", "TEXT"),
    ("service_tier", "TEXT"),
)
_V3_NEW_RUN_META_COLUMNS: tuple[tuple[str, str], ...] = (
    ("reflection_protocol_text", "TEXT"),
    ("reflection_protocol_hash", "TEXT"),
)

# Columns added in v4 (belief protocol observability). Same idempotent
# `ALTER TABLE ADD COLUMN` template as v3; the migration sees v3-stamped DBs
# and adds these three per-sample columns plus the two run_meta columns.
_V4_NEW_PER_SAMPLE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("belief_final", "TEXT"),
    ("belief_trace", "TEXT"),
    ("belief_parse_ok", "INTEGER"),
)
_V4_NEW_RUN_META_COLUMNS: tuple[tuple[str, str], ...] = (
    ("belief_protocol_text", "TEXT"),
    ("belief_protocol_hash", "TEXT"),
)

# Columns added in v5 (harness-resilience: final_answer_retry tracker). Same
# idempotent ALTER TABLE template as v3 / v4. Only one new per-sample column;
# no run_meta column added (the run-wide config_snapshot already records the
# REACT_FINAL_ANSWER_RETRY / REACT_BUDGET_EXCEEDED_DROP_TOOLS settings).
_V5_NEW_PER_SAMPLE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("final_answer_retry_used", "INTEGER"),
)


def sample_col(sample_idx: int, field: str) -> str:
    return f"s{sample_idx}_{field}"


# ---------- Schema (static parts + dynamic run_results) ----------

_STATIC_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS request_events (
    event_id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL,
    model TEXT NOT NULL,
    question_id TEXT NOT NULL,
    sample_idx INTEGER NOT NULL,
    attempt_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_request_events_sample
ON request_events(question_id, sample_idx, attempt_id);

CREATE TABLE IF NOT EXISTS questions (
    id            TEXT PRIMARY KEY,
    choice_type   TEXT NOT NULL CHECK (choice_type IN ('single','multi')),
    question_type TEXT NOT NULL CHECK (question_type IN ('yes_no','binary_named','multiple_choice')),
    event         TEXT NOT NULL,
    options       TEXT NOT NULL,
    answer        TEXT NOT NULL,
    end_time      TEXT NOT NULL,
    imported_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_questions_choice_type   ON questions(choice_type);
CREATE INDEX IF NOT EXISTS idx_questions_question_type ON questions(question_type);

CREATE TABLE IF NOT EXISTS prompt_templates (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    imported_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_meta (
    run_id                    TEXT PRIMARY KEY,
    model                     TEXT NOT NULL,
    sampling_n                INTEGER NOT NULL,
    config_snapshot           TEXT NOT NULL,
    filters_snapshot          TEXT NOT NULL,
    source_db_hash            TEXT NOT NULL,
    metadata_hash             TEXT NOT NULL,
    prompt_templates_hash     TEXT NOT NULL,
    training_cutoff           TEXT,
    started_at                TEXT NOT NULL,
    finished_at               TEXT,
    reflection_protocol_text  TEXT,
    reflection_protocol_hash  TEXT,
    belief_protocol_text      TEXT,
    belief_protocol_hash      TEXT
);
"""


def _run_results_ddl(sampling_n: int) -> str:
    if sampling_n < 1:
        raise ValueError(f"sampling_n must be >= 1, got {sampling_n}")
    cols: list[str] = [
        "question_id TEXT PRIMARY KEY",
        "user_prompt TEXT",
    ]
    for i in range(sampling_n):
        for name, sql_type in PER_SAMPLE_COLUMNS:
            cols.append(f"{sample_col(i, name)} {sql_type}")
    cols.append("FOREIGN KEY (question_id) REFERENCES questions(id)")
    joined = ",\n    ".join(cols)
    return (
        "CREATE TABLE IF NOT EXISTS run_results (\n    "
        + joined
        + "\n);\n"
        + "CREATE INDEX IF NOT EXISTS idx_run_results_question ON run_results(question_id);"
    )


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


def apply_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 5000")


def connect(path: str | Path) -> sqlite3.Connection:
    """Open a sqlite3 connection with standard PRAGMAs applied."""
    conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    apply_pragmas(conn)
    return conn


def init_schema(conn: sqlite3.Connection, sampling_n: int) -> None:
    """Create all tables for one model's DB with a run_results shape matching
    `sampling_n`. Idempotent as long as `sampling_n` stays the same across runs
    that reuse the same DB.

    Older DBs are migrated in-place via the chain v2→v3→v4. Each step inspects
    `PRAGMA table_info` and only ALTERs missing columns, so chained calls on
    an already-migrated DB are no-ops. Migrations run in `init_schema`'s own
    thread *before* `AsyncWriter` starts, so writers never see a half-migrated
    schema.
    """
    conn.executescript(_STATIC_SCHEMA_SQL)
    conn.executescript(_run_results_ddl(sampling_n))
    _migrate_v2_to_v3(conn, sampling_n)
    _migrate_v3_to_v4(conn, sampling_n)
    _migrate_v4_to_v5(conn, sampling_n)
    row = conn.execute(
        "SELECT version FROM schema_version WHERE version = ?",
        (SCHEMA_VERSION,),
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION, utcnow_iso()),
        )
    _assert_run_results_matches(conn, sampling_n)


def _migrate_v2_to_v3(conn: sqlite3.Connection, sampling_n: int) -> None:
    """Add the v3 columns in-place when an existing DB still reports v2.

    The migration is fully idempotent: it inspects `PRAGMA table_info` before
    each `ALTER TABLE ADD COLUMN` and skips columns that already exist (CREATE
    TABLE IF NOT EXISTS handles brand-new DBs by leaving everything in place).
    Only when at least one column was added do we record a `version=3` row in
    `schema_version`, preserving the historical `version=2` row for audit.
    """
    versions = {
        int(r["version"])
        for r in conn.execute("SELECT version FROM schema_version").fetchall()
    }
    # Already past v2: nothing to do (covers v3, v4 and any future stamp).
    if any(v >= 3 for v in versions):
        return

    existing_results = {
        r["name"] for r in conn.execute("PRAGMA table_info(run_results)").fetchall()
    }
    existing_meta = {
        r["name"] for r in conn.execute("PRAGMA table_info(run_meta)").fetchall()
    }

    altered = False
    for i in range(sampling_n):
        for name, sql_type in _V3_NEW_PER_SAMPLE_COLUMNS:
            col = sample_col(i, name)
            if col in existing_results:
                continue
            conn.execute(f"ALTER TABLE run_results ADD COLUMN {col} {sql_type}")
            altered = True
    for name, sql_type in _V3_NEW_RUN_META_COLUMNS:
        if name in existing_meta:
            continue
        conn.execute(f"ALTER TABLE run_meta ADD COLUMN {name} {sql_type}")
        altered = True

    if altered and 2 in versions:
        # Record the upgrade only when we actually transformed a v2 DB; brand
        # new v3+ DBs go through the regular `init_schema` insert path below.
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (3, utcnow_iso()),
        )


def _migrate_v3_to_v4(conn: sqlite3.Connection, sampling_n: int) -> None:
    """Add the v4 belief columns in-place when an existing DB is still v3.

    Same template as `_migrate_v2_to_v3`: inspect `PRAGMA table_info` first,
    only ALTER missing columns, stamp `version=4` only when an actual upgrade
    happened from a v3 stamp. Brand-new v4 DBs land at version=4 via
    `init_schema`'s tail INSERT, not via this migration's stamp.
    """
    versions = {
        int(r["version"])
        for r in conn.execute("SELECT version FROM schema_version").fetchall()
    }
    if 4 in versions:
        return

    existing_results = {
        r["name"] for r in conn.execute("PRAGMA table_info(run_results)").fetchall()
    }
    existing_meta = {
        r["name"] for r in conn.execute("PRAGMA table_info(run_meta)").fetchall()
    }

    altered = False
    for i in range(sampling_n):
        for name, sql_type in _V4_NEW_PER_SAMPLE_COLUMNS:
            col = sample_col(i, name)
            if col in existing_results:
                continue
            conn.execute(f"ALTER TABLE run_results ADD COLUMN {col} {sql_type}")
            altered = True
    for name, sql_type in _V4_NEW_RUN_META_COLUMNS:
        if name in existing_meta:
            continue
        conn.execute(f"ALTER TABLE run_meta ADD COLUMN {name} {sql_type}")
        altered = True

    if altered and 3 in versions:
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (4, utcnow_iso()),
        )


def _migrate_v4_to_v5(conn: sqlite3.Connection, sampling_n: int) -> None:
    """Add the v5 harness-resilience columns when an existing DB is still v4.

    Same idempotent template as `_migrate_v3_to_v4`: inspect `PRAGMA table_info`
    first, only ALTER missing columns, stamp `version=5` only when an actual
    upgrade happened from a v4 stamp. Brand-new v5 DBs land at version=5 via
    `init_schema`'s tail INSERT, not via this migration's stamp.
    """
    versions = {
        int(r["version"])
        for r in conn.execute("SELECT version FROM schema_version").fetchall()
    }
    if 5 in versions:
        return

    existing_results = {
        r["name"] for r in conn.execute("PRAGMA table_info(run_results)").fetchall()
    }

    altered = False
    for i in range(sampling_n):
        for name, sql_type in _V5_NEW_PER_SAMPLE_COLUMNS:
            col = sample_col(i, name)
            if col in existing_results:
                continue
            conn.execute(f"ALTER TABLE run_results ADD COLUMN {col} {sql_type}")
            altered = True

    if altered and 4 in versions:
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (5, utcnow_iso()),
        )


def _assert_run_results_matches(conn: sqlite3.Connection, sampling_n: int) -> None:
    """Guardrail: if a DB already exists with a different SAMPLING_N, fail fast."""
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(run_results)").fetchall()}
    expected = {"question_id", "user_prompt"}
    for i in range(sampling_n):
        for name, _ in PER_SAMPLE_COLUMNS:
            expected.add(sample_col(i, name))
    missing = expected - existing
    if missing:
        raise ValueError(
            "run_results table is missing columns for the configured SAMPLING_N; "
            "the DB was created with a different N. "
            f"missing={sorted(missing)}"
        )


# ---------- Hashing ----------

def compute_source_db_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_kv_string(mapping: dict[str, Any]) -> str:
    parts = []
    for k in sorted(mapping):
        v = mapping[k]
        if isinstance(v, str):
            enc = v
        else:
            enc = json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        parts.append(f"{k}={enc}")
    return "\n".join(parts)


def compute_metadata_hash(features_json: str | dict[str, Any]) -> str:
    if isinstance(features_json, str):
        try:
            data = json.loads(features_json)
        except json.JSONDecodeError:
            return hashlib.sha256(features_json.encode("utf-8")).hexdigest()
    else:
        data = features_json
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_prompt_templates_hash(templates: dict[str, str]) -> str:
    canonical = _canonical_kv_string(templates)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------- Redaction ----------

def redact_api_key(raw: str | None, provider: str) -> dict[str, Any]:
    if not raw:
        return {"provider": provider, "prefix": "", "length": 0, "sha256_12": ""}
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return {
        "provider": provider,
        "prefix": raw[:4],
        "length": len(raw),
        "sha256_12": digest,
    }


# Single-key fields (str) -> directly redact_api_key.
# List fields (after TAVILY_API_KEY upgrade): redact each one, persisted as
# [{prefix, sha256_12, ...}, ...]. This lets post-hoc audits see "which keys
# this run used" without exposing plaintext.
_API_KEY_FIELDS_STR = {
    "LLM_API_KEY": "llm",
    "LEAK_DETECTOR_API_KEY": "leak_detector",
}
_API_KEY_FIELDS_LIST = {
    "TAVILY_API_KEY": "tavily",
}


def snapshot_settings(settings: Settings) -> dict[str, Any]:
    """Return a JSON-ready dict with API keys redacted. Dates are isoformatted."""
    raw = settings.model_dump()
    redacted: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _API_KEY_FIELDS_STR:
            redacted[key] = redact_api_key(getattr(settings, key), _API_KEY_FIELDS_STR[key])
        elif key in _API_KEY_FIELDS_LIST:
            provider = _API_KEY_FIELDS_LIST[key]
            keys: list[str] = list(getattr(settings, key) or [])
            redacted[key] = [redact_api_key(k, provider) for k in keys]
        elif key == "MODEL_TRAINING_CUTOFFS":
            redacted[key] = {m: d.isoformat() for m, d in settings.MODEL_TRAINING_CUTOFFS.items()}
        else:
            redacted[key] = value
    profiles = raw.get("MODEL_PROFILES", {})
    if profiles:
        redacted["inference_profiles_hash"] = hashlib.sha256(
            json.dumps(profiles, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    return redacted


def collection_contract(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Fields that can change which evidence or answers a sample produces."""
    prefixes = ("REACT_", "MODEL_", "TAVILY_")
    names = {
        "MODELS", "LLM_BASE_URL", "LLM_MAX_TOKENS", "LLM_TEMPERATURE", "LLM_TOP_P",
        "LLM_REASONING_MODEL_PATTERNS", "SAMPLING_N", "BELIEF_PROTOCOL", "SCORE_ANSWERS",
        "ENABLE_WEB_SEARCH", "ENABLE_SEARCH_LEAK_FILTER", "LEAK_DETECTOR_BASE_URL",
        "WRITE_MESSAGES_TRACE", "WRITE_REQUEST_AUDIT", "REQUIRE_HEALTHY_RETRIEVAL",
        "LEAK_DETECTOR_MODEL", "LEAK_DETECTOR_REASONING_EFFORT", "LEAK_DETECTOR_TEMPERATURE",
        "LEAK_DETECTOR_MAX_TOKENS", "LEAK_DETECTOR_FAIL_ACTION", "leak_detector_prompt_hash",
        "PROMPT_TEMPLATE_STYLE",
        "LEAK_DETECTOR_RESPONSE_FORMAT",
        "LEAK_DETECTOR_DROP_CONTENT_POLICY",
    }
    excluded = {"TAVILY_API_KEY", "TAVILY_KEY_COOLDOWN_S"}
    if not snapshot.get("MODEL_SAMPLE_INDICES"):
        excluded.add("MODEL_SAMPLE_INDICES")
    if not snapshot.get("LEAK_DETECTOR_DROP_CONTENT_POLICY"):
        excluded.add("LEAK_DETECTOR_DROP_CONTENT_POLICY")
    return {key: value for key, value in snapshot.items()
            if (key in names or key.startswith(prefixes)) and key not in excluded}


def compute_collection_contract_hash(snapshot: dict[str, Any]) -> str:
    canonical = json.dumps(collection_contract(snapshot), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------- Model slug safety ----------

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def model_slug_safe(model: str) -> str:
    """Produce a filesystem-safe filename stem for a model slug.

    `/` is converted to `__` (to keep the provider/model visual split) and any
    other character outside `[A-Za-z0-9._-]` becomes `_`. Collisions are not
    expected in practice because the original slug space is small.

    Examples:
        >>> model_slug_safe("anthropic/claude-sonnet-4.5")
        'anthropic__claude-sonnet-4.5'
        >>> model_slug_safe("openai/gpt-5::r5::c3")
        'openai__gpt-5__r5__c3'
    """
    if not model:
        raise ValueError("model slug must not be empty")
    safe = model.replace("/", "__")
    safe = _UNSAFE_CHARS.sub("_", safe)
    return safe


# ---------- Virtual slug for (real_model, R, C) grid cells ----------

_VIRTUAL_SLUG_RE = re.compile(r"^(?P<real>.+?)::r(?P<R>\d+)::c(?P<C>\d+)$")


def compose_virtual_slug(real_model: str, R: int, C: int) -> str:
    """Encode a `(real_model, R, C)` grid cell into a single slug.

    `R` is the cell-local `TAVILY_MAX_RESULTS` and `C` is the cell-local
    `REACT_MAX_SEARCH_CALLS`. The returned slug is opaque to the runner /
    DB schema / analysis main path: those layers treat it as just another
    `model: str`. Only `evaluation.py` (dispatcher) calls this and only
    `analysis/grid.py` reverses it via `parse_virtual_slug`.

    The `::` delimiter is used because no LLM provider slug we encounter
    contains it (OpenRouter `openai/gpt-5`, Alibaba Bailian `qwen3-max`, etc.) and
    `model_slug_safe` already normalises `:` to `_` for filesystem use.

    Raises ValueError if `real_model` itself contains `::` (would break
    round-tripping via `parse_virtual_slug`).
    """
    if "::" in real_model:
        raise ValueError(
            f"real_model slug must not contain '::' (got {real_model!r})"
        )
    return f"{real_model}::r{int(R)}::c{int(C)}"


def parse_virtual_slug(slug: str) -> tuple[str, int, int] | None:
    """Inverse of `compose_virtual_slug`.

    Returns `(real_model, R, C)` when `slug` matches the virtual pattern,
    else returns `None` (caller treats it as a plain non-virtual slug).
    Never raises.

    The regex uses non-greedy `(.+?)` to handle real_models that contain
    `_` or other special characters; the `::r{R}::c{C}` tail is anchored
    by `$` so trailing junk also yields `None`.
    """
    if not isinstance(slug, str):
        return None
    m = _VIRTUAL_SLUG_RE.match(slug)
    if m is None:
        return None
    return m.group("real"), int(m.group("R")), int(m.group("C"))


# ---------- Run meta (per-model DB) ----------

def register_run_meta(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    model: str,
    sampling_n: int,
    filters_snapshot: dict[str, Any],
    config_snapshot: dict[str, Any],
    source_db_hash: str,
    metadata_hash: str,
    prompt_templates_hash: str,
    training_cutoff: str | None = None,
    started_at: str | None = None,
    reflection_protocol_text: str | None = None,
    reflection_protocol_hash: str | None = None,
    belief_protocol_text: str | None = None,
    belief_protocol_hash: str | None = None,
    grid_origin: dict[str, Any] | None = None,
) -> None:
    """Insert or refresh the single `run_meta` row for this model's DB.

    `finished_at` and `started_at` are preserved on resume so the original
    start timestamp survives. `reflection_protocol_text` / `..._hash` and
    `belief_protocol_text` / `..._hash` are independent of
    `prompt_templates_hash` (DESIGN.md decision 2 + v4 belief decision).

    `grid_origin`, when provided, is injected into the persisted
    `config_snapshot` JSON under the top-level key `grid_origin`. The
    dispatcher in `evaluation.py` assembles it as
    `{"real_model": str, "R": int, "C": int, "effective_min_search_calls": int}`
    so the resulting .db is self-describing about the grid cell it covers.
    When omitted (legacy / non-grid path) the key is NOT written, preserving
    byte-level compatibility with v4 single-cell snapshots.
    """
    if grid_origin is not None:
        config_snapshot = {**config_snapshot, "grid_origin": grid_origin}

    conn.execute(
        """
        INSERT INTO run_meta (
            run_id, model, sampling_n,
            config_snapshot, filters_snapshot,
            source_db_hash, metadata_hash, prompt_templates_hash,
            training_cutoff, started_at, finished_at,
            reflection_protocol_text, reflection_protocol_hash,
            belief_protocol_text, belief_protocol_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            model                    = excluded.model,
            sampling_n               = excluded.sampling_n,
            config_snapshot          = excluded.config_snapshot,
            filters_snapshot         = excluded.filters_snapshot,
            source_db_hash           = excluded.source_db_hash,
            metadata_hash            = excluded.metadata_hash,
            prompt_templates_hash    = excluded.prompt_templates_hash,
            training_cutoff          = excluded.training_cutoff,
            reflection_protocol_text = excluded.reflection_protocol_text,
            reflection_protocol_hash = excluded.reflection_protocol_hash,
            belief_protocol_text     = excluded.belief_protocol_text,
            belief_protocol_hash     = excluded.belief_protocol_hash
        """,
        (
            run_id,
            model,
            sampling_n,
            json.dumps(config_snapshot, ensure_ascii=False, sort_keys=True),
            json.dumps(filters_snapshot, ensure_ascii=False, sort_keys=True),
            source_db_hash,
            metadata_hash,
            prompt_templates_hash,
            training_cutoff,
            started_at or utcnow_iso(),
            reflection_protocol_text,
            reflection_protocol_hash,
            belief_protocol_text,
            belief_protocol_hash,
        ),
    )


def finish_run_meta(
    conn: sqlite3.Connection,
    run_id: str,
    finished_at: str | None = None,
) -> None:
    conn.execute(
        "UPDATE run_meta SET finished_at = ? WHERE run_id = ?",
        (finished_at or utcnow_iso(), run_id),
    )


def load_completed_samples(
    conn: sqlite3.Connection,
    sampling_n: int,
    *,
    retain_model_refusals: bool = False,
) -> set[tuple[str, int]]:
    """Return {(question_id, sample_idx)} already accounted for.

    A sample counts as accounted-for if its `s{i}_created_at` is set AND its
    `s{i}_error` is either NULL (normal completion) or 'skipped_training_cutoff'
    (actively filtered). Opting into refusal retention also accounts for
    'content_policy', without treating it as a prediction. Other errors retry.
    """
    done: set[tuple[str, int]] = set()
    for i in range(sampling_n):
        created = sample_col(i, "created_at")
        err = sample_col(i, "error")
        rows = conn.execute(
            f"""
            SELECT question_id FROM run_results
            WHERE {created} IS NOT NULL
              AND ({err} IS NULL OR {err} = 'skipped_training_cutoff'
                   OR (? AND {err} = 'content_policy'))
            """, (retain_model_refusals,)
        ).fetchall()
        for r in rows:
            done.add((r["question_id"], i))
    return done


# ---------- Async writer ----------

_SENTINEL = object()


class AsyncWriter:
    """Per-DB single-consumer async writer that batches UPSERTs.

    Each producer hands in a dict shaped like:
        {
            "question_id": str,
            "sample_idx": int,
            "user_prompt": str | None,
            "final_answer_letters": str | None,
            "final_answer_raw": str | None,
            "correct": int | None,
            "parse_ok": int,
            "tool_calls_count": int,
            "react_steps": int,
            "prompt_tokens": int,
            "completion_tokens": int,
            "reasoning_tokens": int,
            "latency_ms": int,
            "messages_trace": str | None,
            "search_calls": str | None,
            "error": str | None,
            "created_at": str,
            # v3 observability columns:
            "finish_reason": str | None,
            "nudges_used": int,
            "step_metrics": str | None,
            "response_id": str | None,
            "system_fingerprint": str | None,
            "service_tier": str | None,
            # v4 belief columns:
            "belief_final": str | None,
            "belief_trace": str | None,
            "belief_parse_ok": int,
            # v5 harness-resilience column:
            "final_answer_retry_used": int,
        }

    The writer looks up `sample_idx` and UPSERTs the matching `s{i}_*` columns
    on the `question_id` row. `user_prompt` is written once (COALESCE) so the
    first sample that lands for a question persists the canonical rendered
    prompt; later samples don't overwrite it. Every per-sample key listed
    above MUST be present — a missing key raises `KeyError` at flush time.
    """

    FLUSH_INTERVAL_S = 1.0

    def __init__(self, conn: sqlite3.Connection, sampling_n: int, batch: int) -> None:
        self._conn = conn
        self._sampling_n = sampling_n
        self._batch = max(1, int(batch))
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._started = False
        self._sql_cache: dict[int, str] = {}

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._task = asyncio.create_task(self._run(), name="db-writer")

    async def enqueue_result(self, row: dict[str, Any]) -> None:
        if "question_id" not in row or "sample_idx" not in row:
            raise ValueError("writer row requires question_id and sample_idx")
        await self._queue.put(row)

    async def drain(self) -> None:
        await self._queue.join()

    async def close(self) -> None:
        if not self._started or self._task is None:
            return
        await self._queue.put(_SENTINEL)
        await self._task
        self._task = None
        self._started = False

    async def _run(self) -> None:
        pending: list[dict[str, Any]] = []
        first_enqueued_at: float | None = None
        while True:
            timeout: float | None
            if pending and first_enqueued_at is not None:
                timeout = max(0.0, self.FLUSH_INTERVAL_S - (time.monotonic() - first_enqueued_at))
            else:
                timeout = None
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                await self._flush(pending)
                for _ in range(len(pending)):
                    self._queue.task_done()
                pending.clear()
                first_enqueued_at = None
                continue

            if item is _SENTINEL:
                self._queue.task_done()
                if pending:
                    await self._flush(pending)
                    for _ in range(len(pending)):
                        self._queue.task_done()
                    pending.clear()
                return

            pending.append(item)
            if first_enqueued_at is None:
                first_enqueued_at = time.monotonic()

            if len(pending) >= self._batch:
                await self._flush(pending)
                for _ in range(len(pending)):
                    self._queue.task_done()
                pending.clear()
                first_enqueued_at = None

    def _sql_for(self, sample_idx: int) -> str:
        cached = self._sql_cache.get(sample_idx)
        if cached is not None:
            return cached
        if not (0 <= sample_idx < self._sampling_n):
            raise ValueError(
                f"sample_idx {sample_idx} out of range for SAMPLING_N={self._sampling_n}"
            )
        sample_cols = [sample_col(sample_idx, name) for name in PER_SAMPLE_FIELD_NAMES]
        insert_cols = ["question_id", "user_prompt", *sample_cols]
        placeholders = ",".join("?" * len(insert_cols))
        update_parts = [
            "user_prompt = COALESCE(run_results.user_prompt, excluded.user_prompt)"
        ]
        update_parts.extend(f"{col} = excluded.{col}" for col in sample_cols)
        sql = (
            "INSERT INTO run_results ("
            + ",".join(insert_cols)
            + ") VALUES ("
            + placeholders
            + ") ON CONFLICT(question_id) DO UPDATE SET "
            + ", ".join(update_parts)
        )
        self._sql_cache[sample_idx] = sql
        return sql

    async def _flush(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        await asyncio.to_thread(self._insert_rows_sync, rows)

    def _insert_rows_sync(self, rows: Iterable[dict[str, Any]]) -> None:
        # Group rows by sample_idx so each SQL stays stable within executemany.
        # Per-sample keys are accessed via `row[name]` (not `.get`): missing
        # field MUST raise KeyError so a producer that forgets to populate a
        # new column fails loudly instead of silently writing NULL.
        buckets: dict[int, list[tuple[Any, ...]]] = {}
        for row in rows:
            sample_idx = int(row["sample_idx"])
            qid = row["question_id"]
            payload = [qid, row.get("user_prompt")]
            payload.extend(row[name] for name in PER_SAMPLE_FIELD_NAMES)
            buckets.setdefault(sample_idx, []).append(tuple(payload))

        self._conn.execute("BEGIN")
        try:
            for sample_idx, batch in buckets.items():
                self._conn.executemany(self._sql_for(sample_idx), batch)
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise


# Standalone helper: synchronous upsert used by tests and by the runner's
# best-effort cutoff pre-seeding before the async writer starts.
def upsert_sample_sync(
    conn: sqlite3.Connection,
    sampling_n: int,
    row: dict[str, Any],
) -> None:
    writer_like = AsyncWriter(conn, sampling_n=sampling_n, batch=1)
    writer_like._insert_rows_sync([row])
