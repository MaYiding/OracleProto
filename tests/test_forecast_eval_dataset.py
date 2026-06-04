from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from forecast_eval import db as dbmod
from forecast_eval import loader
from forecast_eval.config import Settings
from forecast_eval.prompts import render_user_prompt
from forecast_eval.types import Question
from scripts.build_forecast_eval_set import validate_database


REPO = Path(__file__).resolve().parents[1]
SOURCE_DB = REPO / "forecast_eval_set_example.db"


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
