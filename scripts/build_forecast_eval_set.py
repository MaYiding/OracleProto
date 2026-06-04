"""Build `forecast_eval_set_example.db` from futurex-ai/Futurex-Past.

Run from the repo root:

    uv run --with pandas --with pyarrow python scripts/build_forecast_eval_set.py

The script reads the latest parquet shard from Hugging Face, keeps level 1/2
rows whose resolution date is on or after the configured cutoff, reconstructs
the seven-column OracleProto question schema, validates every row, and atomically
replaces the destination database.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sqlite3
import sys
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from forecast_eval.parser import parse_answer, parse_gt
from forecast_eval.prompts import (
    DEFAULT_PROMPT_TEMPLATES,
    index_to_letter,
    letter_to_index,
    render_user_prompt,
)
from forecast_eval.types import Question


DEFAULT_SOURCE_PARQUET = (
    "https://huggingface.co/datasets/futurex-ai/Futurex-Past/resolve/main/"
    "data/train-00000-of-00001.parquet"
)
DEFAULT_OUTPUT_DB = REPO / "forecast_eval_set_example.db"
DEFAULT_TABLE = "test_cases"
DEFAULT_MIN_END_DATE = "2026-03-18"
DEFAULT_MIN_ROWS = 301
ALLOWED_LEVELS = frozenset({1, 2})

_OPTION_LINE_RE = re.compile(r"^\s*(?P<label>.)\.\s+(?P<text>.+?)\s*$")
_EVENT_RE = re.compile(
    r'The event to be predicted:\s*"(?P<event>[\s\S]*?)\s*'
    r"\(resolved around\s+[^()]+?\s+\(GMT\+8\)\)\.",
)
_SPACE_RE = re.compile(r"\s+")
_OUTCOME_PREFIX_RE = re.compile(
    r"^(?:the\s+)?outcome\s+(?:be|is|will\s+be)\s+",
    re.IGNORECASE,
)
_PROMPT_INSTRUCTION_FRAGMENTS = (
    "IMPORTANT:",
    "Do not use any other format",
    "\\boxed",
    "Your final answer MUST",
)
_TRUNCATED_MONTH_DATE_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d$"
)


@dataclass(frozen=True)
class ValidationIssue:
    row_id: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    row_count: int
    bad_rows: list[ValidationIssue]
    by_question_type: dict[str, int]
    by_choice_type: dict[str, int]
    min_end_time: str | None
    max_end_time: str | None


class DatasetBuildError(RuntimeError):
    pass


def _normalise_space(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()


def _normalise_for_match(text: str) -> str:
    return _normalise_space(text).casefold()


def _normalise_date(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().date().isoformat()

    text = str(value).strip()
    if not text:
        raise DatasetBuildError("empty end_time")
    slash_match = re.match(r"^(\d{4})/(\d{1,2})/(\d{1,2})(?:\s|$)", text)
    if slash_match:
        y, m, d = slash_match.groups()
        return date(int(y), int(m), int(d)).isoformat()
    if len(text) >= 10:
        head = text[:10]
        try:
            return date.fromisoformat(head).isoformat()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError as exc:
        raise DatasetBuildError(f"cannot parse end_time {text!r}") from exc


def _get(record: Mapping[str, Any], *names: str) -> Any:
    by_lower = {str(k).lower(): k for k in record.keys()}
    for name in names:
        key = by_lower.get(name.lower())
        if key is not None:
            return record[key]
    raise DatasetBuildError(f"missing source column; tried {', '.join(names)}")


def _clean_option_text(text: str) -> str:
    cleaned = _normalise_space(text)
    cleaned = _OUTCOME_PREFIX_RE.sub("", cleaned)
    cleaned = cleaned.rstrip("\"'\\")
    return _normalise_space(cleaned)


def _clean_event_text(text: str) -> str:
    return _normalise_space(text).rstrip("\"'\\")


def _extract_event_from_prompt(prompt: str) -> str:
    match = _EVENT_RE.search(prompt)
    if not match:
        raise DatasetBuildError("cannot extract event from prompt")
    return _normalise_space(match.group("event"))


def _parse_options_from_prompt(prompt: str) -> list[str]:
    options: list[str] = []
    for line in prompt.splitlines():
        match = _OPTION_LINE_RE.match(line)
        if not match:
            continue
        expected = index_to_letter(len(options))
        label = match.group("label")
        if label != expected:
            continue
        option = _clean_option_text(match.group("text"))
        if not option:
            raise DatasetBuildError(f"empty option for label {label}")
        options.append(option)
    return options


def _parse_boxed_options(prompt: str) -> list[str]:
    values = [v.strip() for v in re.findall(r"\\boxed\{([^}]*)\}", prompt) if v.strip()]
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _parse_ground_truth(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        values = list(value)
    else:
        text = str(value).strip()
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            if text.startswith("[") and text.endswith("]"):
                inner = text[1:-1].strip()
                values = [p.strip().strip("'\"") for p in inner.split(",") if p.strip()]
            else:
                values = [text]
        else:
            values = list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]

    cleaned = [_normalise_space(str(v).strip().strip("'\"")) for v in values]
    cleaned = [v for v in cleaned if v]
    if not cleaned:
        raise DatasetBuildError("empty ground_truth")
    return cleaned


def _is_letter_for_options(value: str, options: Sequence[str]) -> bool:
    if len(value) != 1:
        return False
    idx = letter_to_index(value)
    return 0 <= idx < len(options)


def _answer_csv(letters: Iterable[str]) -> str:
    ordered = sorted(set(letters), key=letter_to_index)
    if not ordered:
        raise DatasetBuildError("empty answer letter set")
    return ", ".join(ordered)


def _truth_to_letters(truth_values: Sequence[str], options: Sequence[str]) -> list[str]:
    by_option = {_normalise_for_match(option): index_to_letter(i) for i, option in enumerate(options)}
    letters: list[str] = []
    for value in truth_values:
        if _is_letter_for_options(value, options):
            letters.append(value)
            continue
        matched = by_option.get(_normalise_for_match(value))
        if matched is None:
            raise DatasetBuildError(f"ground_truth {value!r} does not match any option")
        letters.append(matched)
    return sorted(set(letters), key=letter_to_index)


def _truth_to_yes_no_letter(truth_values: Sequence[str]) -> str:
    if len(truth_values) != 1:
        raise DatasetBuildError("yes_no question must have exactly one ground_truth value")
    value = truth_values[0].casefold()
    if value == "yes":
        return "A"
    if value == "no":
        return "B"
    raise DatasetBuildError(f"yes_no ground_truth must be Yes or No, got {truth_values[0]!r}")


def _convert_source_record(
    record: Mapping[str, Any],
    *,
    min_end_date: str,
) -> dict[str, str] | None:
    level = int(_get(record, "level", "Level"))
    end_time = _normalise_date(_get(record, "end_time", "endTime"))
    if level not in ALLOWED_LEVELS or end_time < min_end_date:
        return None

    prompt = str(_get(record, "prompt"))
    event_raw = str(_get(record, "title", "event", "question")).strip()
    event = _clean_event_text(event_raw) if event_raw else _extract_event_from_prompt(prompt)
    if not event:
        raise DatasetBuildError("empty event")

    row_id = _normalise_space(str(_get(record, "id")))
    if not row_id:
        raise DatasetBuildError("empty id")

    truth_values = _parse_ground_truth(_get(record, "ground_truth", "answer"))
    parsed_options = _parse_options_from_prompt(prompt)

    if parsed_options:
        question_type = "multiple_choice"
        options = parsed_options
        answer_letters = _truth_to_letters(truth_values, options)
    else:
        boxed_options = _parse_boxed_options(prompt)
        if {v.casefold() for v in truth_values}.issubset({"yes", "no"}):
            question_type = "yes_no"
            options = ["Yes", "No"]
            answer_letters = [_truth_to_yes_no_letter(truth_values)]
        elif len(boxed_options) == 2:
            question_type = "binary_named"
            options = boxed_options
            answer_letters = _truth_to_letters(truth_values, options)
        else:
            raise DatasetBuildError("no options block and not a yes_no or binary_named prompt")

    if len(options) < 2:
        raise DatasetBuildError("question must have at least two options")
    if question_type in {"yes_no", "binary_named"} and len(answer_letters) != 1:
        raise DatasetBuildError(f"{question_type} question must have one answer")

    choice_type = "multi" if len(answer_letters) > 1 else "single"
    return {
        "id": row_id,
        "choice_type": choice_type,
        "question_type": question_type,
        "event": event,
        "options": json.dumps(options, ensure_ascii=False),
        "answer": _answer_csv(answer_letters),
        "end_time": end_time,
    }


def _question_from_row(row: Mapping[str, Any]) -> Question:
    return Question(
        id=str(row["id"]),
        choice_type=str(row["choice_type"]),
        question_type=str(row["question_type"]),
        event=str(row["event"]),
        options=str(row["options"]),
        answer=str(row["answer"]),
        end_time=str(row["end_time"]),
    )


def _payload_for_answer(q: Question) -> str:
    gt = parse_gt(q.answer)
    options = json.loads(q.options)
    if q.question_type == "yes_no":
        if gt == frozenset({"A"}):
            return "Yes"
        if gt == frozenset({"B"}):
            return "No"
        raise DatasetBuildError("yes_no answer must be A or B")
    if q.question_type == "binary_named":
        if len(gt) != 1:
            raise DatasetBuildError("binary_named answer must contain one letter")
        idx = letter_to_index(next(iter(gt)))
        return options[idx]
    return ", ".join(sorted(gt, key=letter_to_index))


def _validate_question(row: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("id", "choice_type", "question_type", "event", "options", "answer", "end_time"):
        if field not in row or row[field] in (None, ""):
            errors.append(f"{field} is required")
    if errors:
        return errors

    try:
        q = _question_from_row(row)
        options = json.loads(q.options)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return [f"cannot decode row: {exc}"]

    if q.choice_type not in {"single", "multi"}:
        errors.append(f"invalid choice_type {q.choice_type!r}")
    if q.question_type not in {"yes_no", "binary_named", "multiple_choice"}:
        errors.append(f"invalid question_type {q.question_type!r}")
    if not isinstance(options, list) or len(options) < 2:
        errors.append("options must be a JSON array with at least two entries")
    elif any(not isinstance(option, str) or not option.strip() for option in options):
        errors.append("all options must be non-empty strings")

    try:
        end_time = _normalise_date(q.end_time)
    except DatasetBuildError as exc:
        errors.append(str(exc))
    else:
        if end_time != q.end_time:
            errors.append(f"end_time must be YYYY-MM-DD, got {q.end_time!r}")

    try:
        gt = parse_gt(q.answer)
    except ValueError as exc:
        errors.append(str(exc))
        gt = frozenset()

    if options and gt:
        expected_letters = {index_to_letter(i) for i in range(len(options))}
        extra = gt - expected_letters
        if extra:
            errors.append(f"answer letters outside option range: {sorted(extra)}")
    if q.choice_type == "single" and len(gt) != 1:
        errors.append("single choice_type must have exactly one answer letter")
    if q.choice_type == "multi" and len(gt) <= 1:
        errors.append("multi choice_type must have at least two answer letters")
    if q.question_type == "yes_no" and options != ["Yes", "No"]:
        errors.append("yes_no options must be [\"Yes\", \"No\"]")
    if q.question_type == "binary_named" and len(options) != 2:
        errors.append("binary_named must have exactly two options")

    text_fields = [("event", q.event)]
    text_fields.extend((f"option[{i}]", option) for i, option in enumerate(options))
    for field, value in text_fields:
        if value != value.strip():
            errors.append(f"{field} has outer whitespace")
        if any(ch in value for ch in "\r\n\t"):
            errors.append(f"{field} contains control whitespace")
        if any(fragment in value for fragment in _PROMPT_INSTRUCTION_FRAGMENTS):
            errors.append(f"{field} contains prompt instruction text")
        if "\\" in value:
            errors.append(f"{field} contains a backslash escape marker")
        if any(unicodedata.category(ch) in {"Cc", "Cf"} for ch in value):
            errors.append(f"{field} contains Unicode control or format characters")
        if value.endswith(('"', "'", "\\")):
            errors.append(f"{field} ends with a prompt boundary character")
        if "  " in value:
            errors.append(f"{field} contains repeated spaces")
        if _TRUNCATED_MONTH_DATE_RE.search(value):
            errors.append(f"{field} appears to end with a truncated month date")

    try:
        rendered = render_user_prompt(q, DEFAULT_PROMPT_TEMPLATES)
        parsed = parse_answer(f"\\boxed{{{_payload_for_answer(q)}}}", q)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"render/parser round-trip failed: {exc}")
    else:
        if q.event not in rendered:
            errors.append("rendered prompt does not contain event")
        if parsed != gt:
            errors.append(f"parser round-trip mismatch: {parsed} != {gt}")

    return errors


def validate_database(
    db_path: str | Path,
    *,
    table: str = DEFAULT_TABLE,
    min_end_date: str = DEFAULT_MIN_END_DATE,
    min_rows: int = DEFAULT_MIN_ROWS,
) -> ValidationReport:
    path = Path(db_path)
    if not path.exists():
        return ValidationReport(
            row_count=0,
            bad_rows=[ValidationIssue("__database__", f"missing database: {path}")],
            by_question_type={},
            by_choice_type={},
            min_end_time=None,
            max_end_time=None,
        )

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    bad_rows: list[ValidationIssue] = []
    try:
        tables = {
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if table not in tables:
            bad_rows.append(ValidationIssue("__database__", f"missing table {table!r}"))
            return ValidationReport(0, bad_rows, {}, {}, None, None)

        rows = conn.execute(
            f"SELECT id, choice_type, question_type, event, options, answer, end_time "
            f"FROM {table} ORDER BY end_time, id"
        ).fetchall()
    finally:
        conn.close()

    by_question_type: dict[str, int] = {}
    by_choice_type: dict[str, int] = {}
    end_times: list[str] = []

    seen_ids: set[str] = set()
    for row in rows:
        row_dict = dict(row)
        row_id = str(row_dict.get("id", ""))
        if row_id in seen_ids:
            bad_rows.append(ValidationIssue(row_id, "duplicate id"))
        seen_ids.add(row_id)
        by_question_type[row_dict["question_type"]] = by_question_type.get(row_dict["question_type"], 0) + 1
        by_choice_type[row_dict["choice_type"]] = by_choice_type.get(row_dict["choice_type"], 0) + 1
        end_times.append(row_dict["end_time"])
        if row_dict["end_time"] < min_end_date:
            bad_rows.append(
                ValidationIssue(row_id, f"end_time {row_dict['end_time']} < {min_end_date}")
            )
        for message in _validate_question(row_dict):
            bad_rows.append(ValidationIssue(row_id, message))

    if len(rows) < min_rows:
        bad_rows.append(
            ValidationIssue("__database__", f"row_count {len(rows)} < required {min_rows}")
        )

    return ValidationReport(
        row_count=len(rows),
        bad_rows=bad_rows,
        by_question_type=dict(sorted(by_question_type.items())),
        by_choice_type=dict(sorted(by_choice_type.items())),
        min_end_time=min(end_times) if end_times else None,
        max_end_time=max(end_times) if end_times else None,
    )


def _load_source_records(source: str) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise DatasetBuildError(
            "pandas and pyarrow are required to read the source parquet; run "
            "`uv run --with pandas --with pyarrow python scripts/build_forecast_eval_set.py`"
        ) from exc

    frame = pd.read_parquet(source)
    return list(frame.to_dict("records"))


def build_rows(
    source_records: Iterable[Mapping[str, Any]],
    *,
    min_end_date: str = DEFAULT_MIN_END_DATE,
    allow_drop_bad: bool = True,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    bad: list[ValidationIssue] = []
    for i, record in enumerate(source_records):
        row_id = str(record.get("id", f"row_{i}"))
        try:
            converted = _convert_source_record(record, min_end_date=min_end_date)
        except (DatasetBuildError, TypeError, ValueError) as exc:
            bad.append(ValidationIssue(row_id, str(exc)))
            continue
        if converted is not None:
            validation_errors = _validate_question(converted)
            if validation_errors:
                bad.append(ValidationIssue(row_id, "; ".join(validation_errors)))
                continue
            rows.append(converted)

    if bad and not allow_drop_bad:
        lines = "\n".join(f"  {issue.row_id}: {issue.message}" for issue in bad[:30])
        suffix = "" if len(bad) <= 30 else f"\n  ... {len(bad) - 30} more"
        raise DatasetBuildError(f"{len(bad)} source rows could not be parsed:\n{lines}{suffix}")
    if bad:
        lines = "\n".join(f"  {issue.row_id}: {issue.message}" for issue in bad[:10])
        suffix = "" if len(bad) <= 10 else f"\n  ... {len(bad) - 10} more"
        print(f"dropped {len(bad)} source rows:\n{lines}{suffix}", file=sys.stderr)

    rows.sort(key=lambda row: (row["end_time"], row["id"]))
    return rows


def _write_database(rows: Sequence[Mapping[str, str]], db_path: Path, *, table: str) -> None:
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.executescript(
            f"""
            CREATE TABLE {table} (
                id            TEXT PRIMARY KEY,
                choice_type   TEXT NOT NULL CHECK (choice_type IN ('single','multi')),
                question_type TEXT NOT NULL CHECK (
                    question_type IN ('yes_no','binary_named','multiple_choice')
                ),
                event         TEXT NOT NULL,
                options       TEXT NOT NULL,
                answer        TEXT NOT NULL,
                end_time      TEXT NOT NULL
            );
            CREATE INDEX idx_{table}_choice_type ON {table}(choice_type);
            CREATE INDEX idx_{table}_question_type ON {table}(question_type);
            CREATE INDEX idx_{table}_end_time ON {table}(end_time);
            """
        )
        conn.executemany(
            f"INSERT INTO {table} "
            f"(id, choice_type, question_type, event, options, answer, end_time) "
            f"VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["id"],
                    row["choice_type"],
                    row["question_type"],
                    row["event"],
                    row["options"],
                    row["answer"],
                    row["end_time"],
                )
                for row in rows
            ],
        )
        conn.commit()
    finally:
        conn.close()


def build_database(
    *,
    source: str = DEFAULT_SOURCE_PARQUET,
    output: Path = DEFAULT_OUTPUT_DB,
    table: str = DEFAULT_TABLE,
    min_end_date: str = DEFAULT_MIN_END_DATE,
    min_rows: int = DEFAULT_MIN_ROWS,
) -> ValidationReport:
    source_records = _load_source_records(source)
    rows = build_rows(source_records, min_end_date=min_end_date)
    tmp = output.with_suffix(output.suffix + ".tmp")
    _write_database(rows, tmp, table=table)
    report = validate_database(tmp, table=table, min_end_date=min_end_date, min_rows=min_rows)
    if report.bad_rows:
        tmp.unlink(missing_ok=True)
        lines = "\n".join(
            f"  {issue.row_id}: {issue.message}" for issue in report.bad_rows[:30]
        )
        suffix = "" if len(report.bad_rows) <= 30 else f"\n  ... {len(report.bad_rows) - 30} more"
        raise DatasetBuildError(f"generated database failed validation:\n{lines}{suffix}")
    tmp.replace(output)
    return report


def _print_report(report: ValidationReport, output: Path) -> None:
    print("=== verification ===")
    print(f"row_count: {report.row_count}")
    print(f"by_question_type: {report.by_question_type}")
    print(f"by_choice_type: {report.by_choice_type}")
    print(f"end_time: min={report.min_end_time}, max={report.max_end_time}")
    size = output.stat().st_size
    sha = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"file: {output.name}, size={size}B, sha256={sha}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE_PARQUET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DB)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--min-end-date", default=DEFAULT_MIN_END_DATE)
    parser.add_argument("--min-rows", type=int, default=DEFAULT_MIN_ROWS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_database(
            source=args.source,
            output=args.output,
            table=args.table,
            min_end_date=args.min_end_date,
            min_rows=args.min_rows,
        )
    except DatasetBuildError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _print_report(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
