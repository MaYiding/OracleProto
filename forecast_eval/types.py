from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Question:
    """One row of the source-DB question table (default
    `test_cases`, configurable via `SOURCE_TABLE`) after loader sync.

    `options` is the raw JSON string; callers that need the Python list MUST
    `json.loads(q.options)` themselves so downstream comparisons stay byte-for-byte
    stable with what is stored in the DB.
    """

    id: str
    choice_type: str           # 'single' | 'multi'
    question_type: str         # 'yes_no' | 'binary_named' | 'multiple_choice'
    event: str
    options: str               # JSON array string
    answer: str                # e.g. 'A' or 'A, B'
    end_time: str              # YYYY-MM-DD


@dataclass(frozen=True)
class QFilter:
    """CLI-derived filters. None means "do not filter on this dimension"."""

    question_types: frozenset[str] | None = None
    choice_types: frozenset[str] | None = None

    def apply_sql(self) -> tuple[str, list[Any]]:
        """Build a parameterised WHERE clause fragment + params."""
        clauses: list[str] = []
        params: list[Any] = []
        if self.question_types:
            placeholders = ",".join("?" * len(self.question_types))
            clauses.append(f"question_type IN ({placeholders})")
            params.extend(sorted(self.question_types))
        if self.choice_types:
            placeholders = ",".join("?" * len(self.choice_types))
            clauses.append(f"choice_type IN ({placeholders})")
            params.extend(sorted(self.choice_types))
        return (" AND ".join(clauses) if clauses else "1=1"), params

    def snapshot(self) -> dict[str, Any]:
        return {
            "question_types": sorted(self.question_types) if self.question_types else None,
            "choice_types": sorted(self.choice_types) if self.choice_types else None,
        }


@dataclass
class SampleResult:
    """One row about to be enqueued to `run_results`.

    Field names match `run_results` columns 1:1 so `to_row()` is trivial.
    All call sites use keyword arguments, so appending new fields without
    defaults is safe — any future positional caller will fail fast.
    """

    run_id: str
    question_id: str
    model: str
    sample_idx: int

    final_answer_letters: str | None
    final_answer_raw: str | None
    correct: int | None
    parse_ok: int

    tool_calls_count: int
    react_steps: int
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    latency_ms: int

    user_prompt: str | None
    messages_trace: str | None
    search_calls: str | None
    error: str | None
    created_at: str

    finish_reason: str | None
    nudges_used: int
    step_metrics: str | None
    response_id: str | None
    system_fingerprint: str | None
    service_tier: str | None

    # Belief observability (BELIEF_PROTOCOL). All None / 0 when the
    # protocol is disabled or when belief parsing failed for every step.
    belief_final: str | None
    belief_trace: str | None
    belief_parse_ok: int

    # Harness-resilience indicator: 1 iff the ReAct loop exited with an empty
    # `final_raw` and the no-tool bail-out retry kicked in. 0 otherwise. The
    # field has a default so legacy callers (cutoff / error-row builders)
    # don't need to pass it explicitly.
    final_answer_retry_used: int = 0

    def to_row(self) -> dict[str, Any]:
        """Shape matching `forecast_eval.db.AsyncWriter.enqueue_result`.

        `run_id` and `model` are bound to the target DB (run_meta row) and are
        intentionally omitted from the per-sample row so there's no possibility
        of mis-routing a sample into the wrong model's file.
        """
        return {
            "question_id": self.question_id,
            "sample_idx": self.sample_idx,
            "user_prompt": self.user_prompt,
            "final_answer_letters": self.final_answer_letters,
            "final_answer_raw": self.final_answer_raw,
            "correct": self.correct,
            "parse_ok": self.parse_ok,
            "tool_calls_count": self.tool_calls_count,
            "react_steps": self.react_steps,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "latency_ms": self.latency_ms,
            "messages_trace": self.messages_trace,
            "search_calls": self.search_calls,
            "error": self.error,
            "created_at": self.created_at,
            "finish_reason": self.finish_reason,
            "nudges_used": self.nudges_used,
            "step_metrics": self.step_metrics,
            "response_id": self.response_id,
            "system_fingerprint": self.system_fingerprint,
            "service_tier": self.service_tier,
            "belief_final": self.belief_final,
            "belief_trace": self.belief_trace,
            "belief_parse_ok": self.belief_parse_ok,
            "final_answer_retry_used": self.final_answer_retry_used,
        }


@dataclass
class SearchCall:
    """Audit entry appended to `run_results.search_calls`."""

    query: str
    end_date: str
    n_results: int
    published_dates: list[str | None] = field(default_factory=list)
    error_kind: str | None = None
