"""End-to-end smoke test: 3 questions × 1 model × 1 sample against mocked APIs.

Mocks the OpenRouter chat/completions and Tavily /search endpoints via respx,
then drives the full runner against a RUNS_ROOT/{run_id}/db/<slug>.db layout.
Verifies the wide `run_results` table contains the expected per-sample fields,
that `messages_trace` is valid JSON, and that Tavily was called with the
information-barrier-injected end_date.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import httpx
import pytest
import respx

from forecast_eval import db as dbmod
from forecast_eval import loader, runner
from forecast_eval.config import Settings
from forecast_eval.types import QFilter


SOURCE_DB = Path(__file__).resolve().parents[1] / "forecast_eval_set_example.db"
OPENROUTER_URL = re.compile(r"https://openrouter\.ai/api/v1/chat/completions")
TAVILY_URL = re.compile(r"https://api\.tavily\.com/search")


def _make_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("LLM_API_KEY", "sk-or-v1-TEST_ABCDEFGH")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-TEST_ABCDEFGH")
    monkeypatch.setenv("MODELS", "openai/gpt-4o-mini")
    monkeypatch.setenv("MODEL_TRAINING_CUTOFFS", "")
    monkeypatch.setenv("SAMPLING_N", "1")
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "2")
    monkeypatch.setenv("SEARCH_MAX_CONCURRENCY", "2")
    monkeypatch.setenv("REACT_MAX_STEPS", "3")
    monkeypatch.setenv("REACT_MAX_SEARCH_CALLS", "2")
    monkeypatch.setenv("LLM_BACKOFF_NETWORK_S", "0")
    monkeypatch.setenv("LLM_BACKOFF_RATE_LIMIT_S", "0")
    monkeypatch.setenv("LLM_BACKOFF_SERVER_5XX_S", "0")
    monkeypatch.setenv("SEARCH_BACKOFF_S", "0")
    monkeypatch.setenv("SEARCH_RETRY_MAX", "1")
    monkeypatch.setenv("SOURCE_DB", str(SOURCE_DB))
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("DB_COMMIT_BATCH", "2")
    monkeypatch.setenv("WRITE_MESSAGES_TRACE", "true")
    monkeypatch.setenv("TAVILY_END_DATE_OFFSET_DAYS", "-1")
    # Pin Tavily raw_content off so the smoke test stays focused on plumbing
    # rather than tracking the default in .env.example
    monkeypatch.setenv("TAVILY_INCLUDE_RAW_CONTENT", "false")
    # force-final-answer-near-limit-v1 is on by default; the smoke test runs the
    # full plumbing with minimal step count, so turn the new mechanism off to keep
    # user_prompt / messages_trace byte-identical to v5.1 (the new mechanism has
    # dedicated test_force_final_* tests).
    monkeypatch.setenv("REACT_BUDGET_AWARENESS_PROTOCOL", "false")
    monkeypatch.setenv("REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT", "false")
    return Settings(_env_file=None)


def _llm_body_with_tool_call() -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "openai/gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": json.dumps({"query": "latest info"}),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


def _llm_final_body() -> dict:
    return {
        "id": "chatcmpl-2",
        "object": "chat.completion",
        "model": "openai/gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Based on evidence... final: \\boxed{No}",
                },
            }
        ],
        "usage": {"prompt_tokens": 150, "completion_tokens": 30},
    }


def _tavily_body() -> dict:
    return {
        "answer": "Recent reporting suggests No.",
        "results": [
            {
                "title": "News Report",
                "url": "https://example.com/a",
                "content": "Article summary",
                "published_date": "2026-01-10",
            }
        ],
    }


@respx.mock
async def test_smoke_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(tmp_path, monkeypatch)

    llm_responses = [
        httpx.Response(200, json=_llm_body_with_tool_call()),
        httpx.Response(200, json=_llm_final_body()),
    ] * 10  # enough for 3 questions × 2 LLM turns each

    respx.post(OPENROUTER_URL).mock(side_effect=llm_responses)
    tavily_route = respx.post(TAVILY_URL).mock(
        return_value=httpx.Response(200, json=_tavily_body())
    )

    # Create run dir manually, mimicking what evaluation.py would do.
    run_id = "20260424-120000-abcd"
    run_dir = settings.run_dir(run_id)
    (run_dir / "db").mkdir(parents=True, exist_ok=True)

    # Pull 3 yes_no questions from the source DB
    src = sqlite3.connect(f"file:{SOURCE_DB}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    rows = src.execute(
        "SELECT id, choice_type, question_type, event, options, answer, end_time "
        f"FROM {settings.SOURCE_TABLE} WHERE question_type='yes_no' ORDER BY end_time LIMIT 3"
    ).fetchall()
    src.close()
    ids = tuple(r["id"] for r in rows)

    # Filter the source DB down to exactly those 3 questions via a fresh loader.
    # Even single-value .env now goes through the virtual slug pipeline; we
    # mirror evaluation.py here by composing one virtual slug for the (real,
    # R[0], C[0]) cell and downcasting Settings to a cell-local sub-view.
    real_model = settings.MODELS[0]
    R = int(settings.TAVILY_MAX_RESULTS[0])
    C = int(settings.REACT_MAX_SEARCH_CALLS[0])
    model = dbmod.compose_virtual_slug(real_model, R, C)
    cell_settings = settings.model_copy(
        update={
            "MODELS": [model],
            "TAVILY_MAX_RESULTS": R,
            "REACT_MAX_SEARCH_CALLS": C,
        }
    )
    slug = dbmod.model_slug_safe(model)
    model_db = run_dir / "db" / f"{slug}.db"
    conn = dbmod.connect(model_db)
    dbmod.init_schema(conn, settings.SAMPLING_N)
    templates = loader.sync_prompt_templates(SOURCE_DB, conn)

    # Re-load just the 3 targeted questions
    src = sqlite3.connect(f"file:{SOURCE_DB}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(ids))
    questions_rows = src.execute(
        f"SELECT id, choice_type, question_type, event, options, answer, end_time "
        f"FROM {settings.SOURCE_TABLE} WHERE id IN ({placeholders}) ORDER BY end_time",
        ids,
    ).fetchall()
    src.close()

    now = dbmod.utcnow_iso()
    conn.executemany(
        "INSERT OR REPLACE INTO questions (id, choice_type, question_type, event, options, answer, end_time, imported_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        [
            (r["id"], r["choice_type"], r["question_type"], r["event"], r["options"], r["answer"], r["end_time"], now)
            for r in questions_rows
        ],
    )

    from forecast_eval.types import Question
    questions = [
        Question(
            id=r["id"],
            choice_type=r["choice_type"],
            question_type=r["question_type"],
            event=r["event"],
            options=r["options"],
            answer=r["answer"],
            end_time=r["end_time"],
        )
        for r in questions_rows
    ]

    dbmod.register_run_meta(
        conn,
        run_id=run_id,
        model=model,
        sampling_n=cell_settings.SAMPLING_N,
        filters_snapshot={"question_types": ["yes_no"], "question_count": len(questions)},
        config_snapshot=dbmod.snapshot_settings(cell_settings),
        source_db_hash=dbmod.compute_source_db_hash(SOURCE_DB),
        metadata_hash=dbmod.compute_metadata_hash(loader.load_raw_features_json(SOURCE_DB)),
        prompt_templates_hash=dbmod.compute_prompt_templates_hash(templates),
        grid_origin={
            "real_model": real_model,
            "R": R,
            "C": C,
            "effective_min_search_calls": min(cell_settings.REACT_MIN_SEARCH_CALLS, C),
        },
    )

    stats = await runner.run(
        settings=cell_settings,
        filters=QFilter(question_types=frozenset({"yes_no"})),
        questions=questions,
        templates=templates,
        run_id=run_id,
        conns={model: conn},
    )

    assert stats.planned == 3
    assert stats.done == 3
    assert stats.errors == {}

    # Every written row must have the expected shape (wide table: s0_*)
    written = conn.execute(
        "SELECT * FROM run_results ORDER BY question_id"
    ).fetchall()
    assert len(written) == 3
    for row in written:
        assert row["s0_error"] is None
        assert row["s0_parse_ok"] == 1
        assert row["s0_final_answer_letters"] is not None
        assert row["s0_messages_trace"] is not None
        assert row["s0_search_calls"] is not None
        assert row["user_prompt"]
        msgs = json.loads(row["s0_messages_trace"])
        assert isinstance(msgs, list) and msgs[0]["role"] == "user"
        calls = json.loads(row["s0_search_calls"])
        assert calls and all("end_date" in c for c in calls)
        assert row["s0_latency_ms"] >= 0
        assert (row["s0_prompt_tokens"] or 0) + (row["s0_completion_tokens"] or 0) > 0

    # Tavily must have been called with the correct end_date (= end_time + OFFSET)
    assert tavily_route.called
    from datetime import date, timedelta
    expected_end_dates = {
        (date.fromisoformat(q.end_time) + timedelta(days=-1)).isoformat()
        for q in questions
    }
    observed_end_dates = {
        json.loads(call.request.content.decode("utf-8"))["end_date"]
        for call in tavily_route.calls
    }
    assert observed_end_dates == expected_end_dates
    assert all(
        json.loads(call.request.content.decode("utf-8"))["include_raw_content"] is False
        for call in tavily_route.calls
    )

    # run_meta.finished_at set on normal completion
    r = conn.execute("SELECT finished_at FROM run_meta WHERE run_id=?", (run_id,)).fetchone()
    assert r["finished_at"] is not None
    conn.close()
