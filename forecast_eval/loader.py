from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from .db import utcnow_iso
from .prompts import DEFAULT_PROMPT_TEMPLATES, REQUIRED_PROMPT_TEMPLATE_KEYS
from .types import QFilter, Question


def _default_features() -> dict[str, Any]:
    return {
        "prompt_reconstruction": dict(DEFAULT_PROMPT_TEMPLATES),
    }


def _source_has_table(src_conn: sqlite3.Connection, table: str) -> bool:
    row = src_conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _connect_source(source_db: str | Path) -> sqlite3.Connection:
    uri = Path(source_db).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _read_features_json(src_conn: sqlite3.Connection) -> dict[str, Any]:
    if not _source_has_table(src_conn, "dataset_metadata"):
        return _default_features()
    row = src_conn.execute("SELECT features_json FROM dataset_metadata").fetchone()
    if row is None:
        raise ValueError(
            "dataset_metadata is empty; source DB is not a valid forecast eval dataset"
        )
    return json.loads(row["features_json"])


def sync_prompt_templates(
    source_db: str | Path,
    results_conn: sqlite3.Connection,
) -> dict[str, str]:
    """Flatten source prompt templates into
    `results.db.prompt_templates` and return it as a dict for in-memory use.

    String fields are stored verbatim; nested (dict/list) values are JSON-serialised
    into the same key so downstream readers can `json.loads` on demand.
    """
    src = _connect_source(source_db)
    try:
        features = _read_features_json(src)
    finally:
        src.close()

    reconstruction = features.get("prompt_reconstruction")
    if not isinstance(reconstruction, dict):
        raise ValueError("features_json.prompt_reconstruction missing or malformed")

    flat: dict[str, str] = {}
    for key, value in reconstruction.items():
        if isinstance(value, str):
            flat[key] = value
        else:
            flat[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)

    missing = [k for k in REQUIRED_PROMPT_TEMPLATE_KEYS if k not in flat]
    if missing:
        raise ValueError(f"prompt_reconstruction missing required keys: {missing}")
    unexpected = [k for k in flat if k not in REQUIRED_PROMPT_TEMPLATE_KEYS]
    if unexpected:
        raise ValueError(f"prompt_reconstruction contains unknown keys: {unexpected}")

    now = utcnow_iso()
    results_conn.executemany(
        "INSERT INTO prompt_templates (key, value, imported_at) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, imported_at = excluded.imported_at",
        [(k, v, now) for k, v in flat.items()],
    )
    return flat


_SQL_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def sync_questions(
    source_db: str | Path,
    results_conn: sqlite3.Connection,
    filters: QFilter,
    table: str = "test_cases",
) -> list[Question]:
    """Copy the filtered question rows from `<source_db>.<table>` into
    `results.db.questions`.

    `table` must be a bare SQLite identifier — Settings already validates this,
    but we re-check here so direct callers (tests, ad-hoc scripts) can't slip
    user-controlled SQL into the FROM clause.
    """
    if not _SQL_IDENT_RE.match(table):
        raise ValueError(
            f"sync_questions: table {table!r} must match [A-Za-z_][A-Za-z0-9_]*"
        )
    src = _connect_source(source_db)
    try:
        where, params = filters.apply_sql()
        rows = src.execute(
            f"SELECT id, choice_type, question_type, event, options, answer, end_time "
            f"FROM {table} WHERE {where} "
            f"ORDER BY end_time, id",
            params,
        ).fetchall()
    finally:
        src.close()

    now = utcnow_iso()
    results_conn.executemany(
        "INSERT INTO questions (id, choice_type, question_type, event, options, answer, end_time, imported_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET "
        "choice_type=excluded.choice_type, question_type=excluded.question_type, "
        "event=excluded.event, options=excluded.options, answer=excluded.answer, "
        "end_time=excluded.end_time, imported_at=excluded.imported_at",
        [
            (
                r["id"],
                r["choice_type"],
                r["question_type"],
                r["event"],
                r["options"],
                r["answer"],
                r["end_time"],
                now,
            )
            for r in rows
        ],
    )

    return [
        Question(
            id=r["id"],
            choice_type=r["choice_type"],
            question_type=r["question_type"],
            event=r["event"],
            options=r["options"],
            answer=r["answer"],
            end_time=r["end_time"],
        )
        for r in rows
    ]


def load_raw_features_json(source_db: str | Path) -> str:
    """Return the raw `features_json` string for metadata hashing."""
    src = _connect_source(source_db)
    try:
        if not _source_has_table(src, "dataset_metadata"):
            return json.dumps(_default_features(), ensure_ascii=False, sort_keys=True)
        row = src.execute("SELECT features_json FROM dataset_metadata").fetchone()
    finally:
        src.close()
    if row is None:
        raise ValueError("dataset_metadata is empty")
    return row["features_json"]
