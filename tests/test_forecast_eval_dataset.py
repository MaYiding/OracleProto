from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from pathlib import Path

import pytest

from forecast_eval import db as dbmod
from forecast_eval import loader
from forecast_eval.config import Settings
from forecast_eval.prompts import DEFAULT_PROMPT_TEMPLATES, render_user_prompt
from forecast_eval.types import Question
from scripts.build_forecast_eval_set import build_rows, validate_database


REPO = Path(__file__).resolve().parents[1]
SOURCE_DB = REPO / "forecast_eval_set_example.db"
TRUNCATED_MONTH_DATE_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d$"
)


def test_bundled_dataset_schema_and_rows_are_parseable() -> None:
    report = validate_database(
        SOURCE_DB,
        table="test_cases",
        min_end_date="2026-03-18",
        min_rows=301,
    )
    assert report.row_count > 300
    assert report.bad_rows == []

    conn = sqlite3.connect(f"file:{SOURCE_DB}?mode=ro", uri=True)
    try:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    finally:
        conn.close()

    assert "test_cases" in tables
    assert "forecast_eval_set_example" not in tables
    assert "dataset_metadata" not in tables


def test_bundled_dataset_text_fields_are_prompt_safe() -> None:
    conn = sqlite3.connect(f"file:{SOURCE_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT id, event, options FROM test_cases").fetchall()
    finally:
        conn.close()

    dirty: list[tuple[str, str, str]] = []
    forbidden_fragments = (
        "IMPORTANT:",
        "Do not use any other format",
        "\\boxed",
        "Your final answer MUST",
    )
    for row in rows:
        fields = [("event", row["event"])]
        fields.extend(
            (f"option[{i}]", option)
            for i, option in enumerate(json.loads(row["options"]))
        )
        for field, value in fields:
            if value != value.strip():
                dirty.append((row["id"], field, "outer whitespace"))
            if any(ch in value for ch in "\r\n\t"):
                dirty.append((row["id"], field, "control whitespace"))
            if any(fragment in value for fragment in forbidden_fragments):
                dirty.append((row["id"], field, "prompt instruction fragment"))
            if "\\" in value:
                dirty.append((row["id"], field, "backslash escape marker"))
            if any(unicodedata.category(ch) in {"Cc", "Cf"} for ch in value):
                dirty.append((row["id"], field, "unicode control/format char"))
            if value.endswith(('"', "'", "\\")):
                dirty.append((row["id"], field, f"trailing boundary char: {value!r}"))
            if "  " in value:
                dirty.append((row["id"], field, "repeated spaces"))
            if TRUNCATED_MONTH_DATE_RE.search(value):
                dirty.append((row["id"], field, f"truncated month date: {value!r}"))

    assert dirty == []


def test_source_prompt_closing_quote_is_not_kept_as_option_text() -> None:
    rows = build_rows(
        [
            {
                "id": "q1",
                "level": 1,
                "end_time": "2026-03-19 00:00:00",
                "title": "Clean sample",
                "ground_truth": "['B']",
                "prompt": (
                    'You are an agent that can predict future events. The event to be predicted: '
                    '"Clean sample (resolved around 2026-03-19 (GMT+8)). \n'
                    "A.  the outcome be Alpha\n"
                    'B.  the outcome be Beta"\n'
                    "IMPORTANT: Your final answer MUST end with this exact format:\n"
                ),
            }
        ],
        min_end_date="2026-03-18",
    )

    assert json.loads(rows[0]["options"]) == ["Alpha", "Beta"]


def test_source_rows_with_truncated_event_dates_are_dropped() -> None:
    rows = build_rows(
        [
            {
                "id": "q_truncated",
                "level": 1,
                "end_time": "2026-03-19 00:00:00",
                "title": "UEFA Champions League Playoff Prop Bets, Mar 10th-Mar 1",
                "ground_truth": "['B']",
                "prompt": (
                    'You are an agent that can predict future events. The event to be predicted: '
                    '"UEFA Champions League Playoff Prop Bets, Mar 10th-Mar 1 '
                    '(resolved around 2026-03-19 (GMT+8)). \n'
                    "A.  the outcome be Alpha\n"
                    'B.  the outcome be Beta"\n'
                    "IMPORTANT: Your final answer MUST end with this exact format:\n"
                ),
            }
        ],
        min_end_date="2026-03-18",
    )

    assert rows == []


def test_default_source_table_tracks_bundled_dataset(monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "sk-or-v1-TEST_ABCDEFGH")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-TEST_ABCDEFGH")
    monkeypatch.setenv("MODELS", "openai/gpt-4o-mini")
    monkeypatch.setenv("MODEL_TRAINING_CUTOFFS", "")

    settings = Settings(_env_file=None)

    assert settings.SOURCE_DB == "./forecast_eval_set_example.db"
    assert settings.SOURCE_TABLE == "test_cases"


def test_source_db_without_dataset_metadata_uses_default_prompt_templates(
    tmp_path: Path,
) -> None:
    src = tmp_path / "source.db"
    conn = sqlite3.connect(src)
    conn.execute(
        """
        CREATE TABLE test_cases (
            id TEXT PRIMARY KEY,
            choice_type TEXT NOT NULL,
            question_type TEXT NOT NULL,
            event TEXT NOT NULL,
            options TEXT NOT NULL,
            answer TEXT NOT NULL,
            end_time TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO test_cases VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "q1",
            "single",
            "yes_no",
            "Will the event happen?",
            json.dumps(["Yes", "No"]),
            "A",
            "2026-03-19",
        ),
    )
    conn.commit()
    conn.close()

    results_conn = dbmod.connect(tmp_path / "results.db")
    dbmod.init_schema(results_conn, sampling_n=1)

    templates = loader.sync_prompt_templates(src, results_conn)
    raw_features = loader.load_raw_features_json(src)
    features = json.loads(raw_features)

    assert features["prompt_reconstruction"] == templates
    rendered = render_user_prompt(
        Question(
            id="q1",
            choice_type="single",
            question_type="yes_no",
            event="Will the event happen?",
            options=json.dumps(["Yes", "No"]),
            answer="A",
            end_time="2026-03-19",
        ),
        templates,
    )
    assert "Will the event happen?" in rendered


def test_source_prompt_templates_reject_unknown_keys(tmp_path: Path) -> None:
    src = tmp_path / "source.db"
    conn = sqlite3.connect(src)
    conn.execute("CREATE TABLE dataset_metadata (features_json TEXT NOT NULL)")
    features = {
        "prompt_reconstruction": {
            **DEFAULT_PROMPT_TEMPLATES,
            "prompt_template": "{agent_role}\n{output_format}",
        }
    }
    conn.execute(
        "INSERT INTO dataset_metadata VALUES (?)",
        (json.dumps(features, ensure_ascii=False),),
    )
    conn.commit()
    conn.close()

    results_conn = dbmod.connect(tmp_path / "results.db")
    dbmod.init_schema(results_conn, sampling_n=1)

    with pytest.raises(ValueError, match="unknown keys"):
        loader.sync_prompt_templates(src, results_conn)
