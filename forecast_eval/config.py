from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


_PLACEHOLDER_TOKENS = {"REPLACE_ME", "CHANGEME", "PUT_YOUR_KEY_HERE"}
_RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")
_SQL_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Composite difficulty buckets. The four-bucket nested key set comes from
# factorising the question-type axis (yes_no / binary_named / multiple_choice)
# by the within-mc answering-mode axis (single / multi); see
# ``forecast_eval.analysis.composite.bucket_of`` for the (question_type,
# choice_type) -> bucket mapping. A frozenset is sufficient for validation;
# an enum class is not needed.
COMPOSITE_BUCKETS: frozenset[str] = frozenset(
    {"yes_no", "binary_named", "mc_single", "mc_multi"}
)


def _parse_csv(raw: str | list[Any] | None) -> list[str]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _parse_csv_int(raw: str | list[Any] | None) -> list[int]:
    return [int(x) for x in _parse_csv(raw)]


def _parse_weights_dict(
    raw: str | dict[str, Any] | None, *, allowed_buckets: frozenset[str], field_name: str
) -> dict[str, float]:
    """Parse a weight mapping of the form ``"yes_no=0.15,binary_named=0.15,multiple_choice=0.70"``.

    Validation: bucket names must be in ``allowed_buckets``, weights must parse as float, weights >= 0.
    Missing buckets are treated as 0 (i.e., "the bucket does not participate in synthesis"),
    but at least one bucket weight must be > 0.
    """
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        items = [(str(k).strip(), v) for k, v in raw.items()]
    else:
        items = []
        for pair in str(raw).split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(
                    f"{field_name} entry must be 'bucket=weight', got: {pair!r}"
                )
            bucket, value = pair.split("=", 1)
            items.append((bucket.strip(), value.strip()))
    out: dict[str, float] = {}
    for bucket, value in items:
        if not bucket:
            raise ValueError(f"{field_name} has an empty bucket name")
        if bucket not in allowed_buckets:
            raise ValueError(
                f"{field_name} bucket {bucket!r} not in {sorted(allowed_buckets)}"
            )
        try:
            w = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"{field_name}[{bucket}] is not a valid float: {value!r}"
            ) from e
        if w < 0:
            raise ValueError(
                f"{field_name}[{bucket}] must be >= 0; got {w}"
            )
        out[bucket] = w
    if out and not any(w > 0 for w in out.values()):
        raise ValueError(
            f"{field_name} requires at least one weight > 0 (otherwise composite "
            "score has no defined denominator)"
        )
    return out


def _parse_overrides_dict(
    raw: str | dict[str, Any] | None,
    *,
    allowed_buckets: frozenset[str],
    field_name: str,
) -> dict[str, dict[str, float]]:
    """Parse per-metric weight overrides of the form
    ``"fss=yes_no=0.05,binary_named=0.05,multiple_choice=0.90;cohen_kappa=..."``.

    Semicolons separate different metrics, commas separate buckets within the same metric;
    each segment follows :func:`_parse_weights_dict` semantics (including the at-least-one > 0 check).
    Misspelled metric names (not in the known-metric whitelist) are raised at runtime by
    ``forecast_eval.analysis.composite``; this function does not perform that layer of validation,
    to avoid the config module reverse-importing analysis.
    """
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        # dict passed in directly: values can be a "k=v,k=v" string or an already-parsed dict
        out: dict[str, dict[str, float]] = {}
        for metric, sub in raw.items():
            metric_name = str(metric).strip()
            if not metric_name:
                raise ValueError(f"{field_name} has an empty metric name")
            sub_field = f"{field_name}[{metric_name}]"
            out[metric_name] = _parse_weights_dict(
                sub, allowed_buckets=allowed_buckets, field_name=sub_field
            )
        return out
    out = {}
    for segment in str(raw).split(";"):
        segment = segment.strip()
        if not segment:
            continue
        if "=" not in segment:
            raise ValueError(
                f"{field_name} segment must be 'metric=bucket=w,bucket=w'; got: {segment!r}"
            )
        metric_name, rest = segment.split("=", 1)
        metric_name = metric_name.strip()
        if not metric_name:
            raise ValueError(f"{field_name} has an empty metric name in segment: {segment!r}")
        sub_field = f"{field_name}[{metric_name}]"
        out[metric_name] = _parse_weights_dict(
            rest, allowed_buckets=allowed_buckets, field_name=sub_field
        )
    return out


_MAX_TOKENS_PARAM_ALLOWED = {"max_tokens", "max_completion_tokens"}
_OMIT_SAMPLING_FIELDS_ALLOWED = {"temperature", "top_p"}


def _parse_max_tokens_param(raw: str | dict[str, Any] | None) -> dict[str, str]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        items = [(str(k), str(v)) for k, v in raw.items()]
    else:
        items = []
        for pair in str(raw).split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(
                    f"MODEL_MAX_TOKENS_PARAM entry must be 'model=param_name', got: {pair!r}"
                )
            model_slug, name = pair.split("=", 1)
            items.append((model_slug.strip(), name.strip()))
    out: dict[str, str] = {}
    for model_slug, name in items:
        if not model_slug:
            raise ValueError("MODEL_MAX_TOKENS_PARAM has an empty model slug")
        if name not in _MAX_TOKENS_PARAM_ALLOWED:
            raise ValueError(
                f"MODEL_MAX_TOKENS_PARAM[{model_slug}]={name!r} must be one of "
                f"{sorted(_MAX_TOKENS_PARAM_ALLOWED)}"
            )
        out[model_slug] = name
    return out


def _parse_omit_sampling_fields(
    raw: str | dict[str, Any] | None,
) -> dict[str, list[str]]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        items: list[tuple[str, str]] = []
        for k, v in raw.items():
            if isinstance(v, (list, tuple, set)):
                items.extend((str(k), str(x)) for x in v)
            else:
                items.append((str(k), str(v)))
    else:
        items = []
        for pair in str(raw).split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(
                    f"MODEL_OMIT_SAMPLING_FIELDS entry must be 'model=temperature' "
                    f"or 'model=top_p', got: {pair!r}"
                )
            model_slug, name = pair.split("=", 1)
            items.append((model_slug.strip(), name.strip()))
    out: dict[str, list[str]] = {}
    for model_slug, name in items:
        if not model_slug:
            raise ValueError("MODEL_OMIT_SAMPLING_FIELDS has an empty model slug")
        if name not in _OMIT_SAMPLING_FIELDS_ALLOWED:
            raise ValueError(
                f"MODEL_OMIT_SAMPLING_FIELDS[{model_slug}]={name!r} must be one of "
                f"{sorted(_OMIT_SAMPLING_FIELDS_ALLOWED)}"
            )
        bucket = out.setdefault(model_slug, [])
        if name not in bucket:
            bucket.append(name)
    return out


def _parse_cutoffs(raw: str | dict[str, Any] | None) -> dict[str, date]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        items = [(str(k), v) for k, v in raw.items()]
    else:
        items = []
        for pair in str(raw).split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(
                    f"MODEL_TRAINING_CUTOFFS entry must be 'model=YYYY-MM-DD', got: {pair!r}"
                )
            model_slug, d = pair.split("=", 1)
            items.append((model_slug.strip(), d.strip()))
    out: dict[str, date] = {}
    for model_slug, value in items:
        if not model_slug:
            raise ValueError("MODEL_TRAINING_CUTOFFS has an empty model slug")
        if isinstance(value, date):
            out[model_slug] = value
        else:
            try:
                out[model_slug] = date.fromisoformat(str(value))
            except ValueError as e:
                raise ValueError(
                    f"MODEL_TRAINING_CUTOFFS[{model_slug}] is not a valid YYYY-MM-DD: {value!r}"
                ) from e
    return out


class Settings(BaseSettings):
    """Runtime configuration loaded once from `.env`.

    All tunable parameters live here; CLI only owns --question-type/--choice-type.
    Instances MUST be constructed at process start; failures raise ValueError to
    abort before any LLM / Tavily call is made.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # LLM (any OpenAI-compatible endpoint)
    LLM_API_KEY: str
    LLM_BASE_URL: str = "https://openrouter.ai/api/v1"
    MODELS: Annotated[list[str], NoDecode] = Field(default_factory=list)
    MODEL_TRAINING_CUTOFFS: Annotated[dict[str, date], NoDecode] = Field(default_factory=dict)
    LLM_MAX_TOKENS: int = 12000
    # Some providers (e.g., OpenAI official o-series / GPT-5 on /v1/chat/completions)
    # have deprecated `max_tokens` in favor of `max_completion_tokens`. This overrides per
    # model slug the field name actually used in the request body; undeclared models still default to `max_tokens`.
    # Format: "<model_slug>=max_completion_tokens" with multiple entries comma-separated.
    MODEL_MAX_TOKENS_PARAM: Annotated[dict[str, str], NoDecode] = Field(default_factory=dict)
    LLM_TIMEOUT_S: int = 240
    LLM_TEMPERATURE: float = 0.7
    LLM_TOP_P: float = 1.0
    # Per-model omission of individual sampling fields. Use when a non-reasoning model rejects
    # `temperature` and `top_p` together (e.g., providers returning 400 with "cannot both be
    # specified") but still accepts custom sampling on the kept field. Format:
    # "<model_slug>=temperature" or "<model_slug>=top_p" with multiple entries comma-separated;
    # repeating the same slug accumulates fields, e.g. "foo=temperature,foo=top_p" omits both
    # (in which case prefer LLM_REASONING_MODEL_PATTERNS instead). Allowed values: temperature, top_p.
    # Reasoning models matched by LLM_REASONING_MODEL_PATTERNS already omit both fields and ignore
    # this map entirely.
    MODEL_OMIT_SAMPLING_FIELDS: Annotated[dict[str, list[str]], NoDecode] = Field(default_factory=dict)
    LLM_MAX_CONCURRENCY: int = 5
    LLM_RETRY_MAX: int = 5
    LLM_BACKOFF_NETWORK_S: Annotated[list[int], NoDecode] = Field(default_factory=lambda: [2, 5, 15, 30, 60])
    LLM_BACKOFF_RATE_LIMIT_S: Annotated[list[int], NoDecode] = Field(default_factory=lambda: [10, 30, 60, 120, 300])
    LLM_BACKOFF_SERVER_5XX_S: Annotated[list[int], NoDecode] = Field(default_factory=lambda: [5, 15, 30, 60, 120])
    # List of reasoning-model slug substrings: matched models omit temperature / top_p in calls
    # (reasoning models such as o-series / deepseek-r1 / qwq do not accept custom sampling parameters and return 400)
    LLM_REASONING_MODEL_PATTERNS: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["o1", "o3", "o4", "r1", "qwq"]
    )

    # Web search master switch: when False, the ReAct loop runs with no tool
    # schema at all — Tavily is never hit and TAVILY_API_KEY becomes optional.
    ENABLE_WEB_SEARCH: bool = True

    # Tavily - see field comments in .env.example for details
    # Supports a single key or CSV multi-key (`TAVILY_API_KEY=tvly-aaa,tvly-bbb`). With multiple keys,
    # forecast_eval.tavily_keys.TavilyKeyPool performs least-used scheduling + failure circuit breaker,
    # and all grid cells within the same process share a single pool instance via module-level cache
    # (cache key = tuple(TAVILY_API_KEY)), so usage counts accumulate across cells rather than per-cell.
    TAVILY_API_KEY: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # Seconds to temporarily blacklist a single key when it hits 429 / quota-exceeded. 0 = no blacklist
    # (rely on acquire order alone). 401/403 goes through permanent blacklist, unaffected by this parameter.
    TAVILY_KEY_COOLDOWN_S: float = 60.0
    # Grid-scannable: list of cell-local R values, parsed from CSV in .env.
    # Single-value envs (`TAVILY_MAX_RESULTS=5`) parse to `[5]`; multi-value
    # (`TAVILY_MAX_RESULTS=5,10`) drives the (R, C) cartesian dispatcher in
    # evaluation.py. Per-cell sub-views downcast this to a single int via
    # `model_copy(update={"TAVILY_MAX_RESULTS": R})` before being handed to
    # `tavily_search`, so the Tavily request body always sees a single int.
    TAVILY_MAX_RESULTS: Annotated[list[int], NoDecode] = Field(default_factory=lambda: [5])
    # basic | advanced (Tavily official search_depth)
    TAVILY_SEARCH_DEPTH: str = "basic"
    # false | markdown | text (legacy bool 'true' compatibility maps to 'markdown')
    TAVILY_INCLUDE_RAW_CONTENT: str = "markdown"
    # Single-result raw_content truncation length; 0 = no truncation
    TAVILY_RAW_CONTENT_MAX_CHARS: int = 8000
    # false | basic | advanced (Tavily internal LLM quick answer, off by default to avoid polluting evaluation)
    TAVILY_INCLUDE_ANSWER: str = "false"
    TAVILY_END_DATE_OFFSET_DAYS: int = -1
    SEARCH_MAX_CONCURRENCY: int = 5
    SEARCH_RETRY_MAX: int = 3
    SEARCH_BACKOFF_S: Annotated[list[int], NoDecode] = Field(default_factory=lambda: [2, 5, 15])

    # ReAct
    REACT_MAX_STEPS: int = 6
    # Grid-scannable: list of cell-local C values, parsed from CSV. Same shape
    # contract as TAVILY_MAX_RESULTS — dispatcher derives per-cell sub-views
    # carrying a single int; runner / react never see the list form.
    REACT_MAX_SEARCH_CALLS: Annotated[list[int], NoDecode] = Field(default_factory=lambda: [4])
    # Master switch for reflection protocol: when enabled, multi-step reasoning scaffolding is appended
    # at the end of the user prompt, significantly improving search + reflection depth.
    # Not written to prompt_templates (so prompt_templates_hash remains unchanged), but the actual user message
    # text lands in each sample's user_prompt field and is also recorded in run_meta via config_snapshot.
    REACT_REFLECTION_PROTOCOL: bool = True
    # Soft minimum search count: when the LLM tries to give a final answer but cumulative web_search < min,
    # inject a user nudge asking it to continue retrieval. 0 = disabled (default; rely on reflection protocol
    # to drive naturally). >0 = enable fallback floor.
    REACT_MIN_SEARCH_CALLS: int = 0
    # Max number of nudges to inject, preventing infinite nudge loops between LLM and system. REACT_MAX_STEPS is still the hard ceiling.
    REACT_MAX_NUDGES: int = 2
    # Harness-resilience switches.
    # REACT_FINAL_ANSWER_RETRY=True: when the loop ends normally but final_raw=="", call LLM once more with tools=[].
    #   Default False - superseded by force-final-answer-near-limit-v1 (last-step hard switch to tools=[] within the loop),
    #   kept as an optional out-of-loop emergency backstop. Enabling costs one extra LLM step (react_steps + 1).
    # REACT_BUDGET_EXCEEDED_DROP_TOOLS=True (default): once cumulative web_search >= REACT_MAX_SEARCH_CALLS,
    #   subsequent rounds call the LLM with tools=[], forcing it to output content only (rather than repeatedly hitting budget exceeded).
    REACT_FINAL_ANSWER_RETRY: bool = False
    REACT_BUDGET_EXCEEDED_DROP_TOOLS: bool = True

    # force-final-answer-near-limit-v1: front-loads "running out of steps" at the prompt layer,
    # complementary to REACT_FINAL_ANSWER_RETRY (post-hoc backstop).
    # REACT_BUDGET_AWARENESS_PROTOCOL=True: append a budget-awareness section at the end of the user prompt,
    #   informing the model of total steps + total searches, guiding it to reserve the last step for the \boxed{...} answer.
    # REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT=True: actively inject user messages as the loop nears the limit:
    #   - Step (N - REACT_FORCE_FINAL_ANSWER_LOOKAHEAD) through second-to-last: inject soft reminder, tool_calls still allowed.
    #   - Last step: inject hard wrap-up text + tools=[], forcing content-only output of \boxed{...}.
    # REACT_FORCE_FINAL_ANSWER_LOOKAHEAD: int = how many steps before the limit to start intervening (>= 1, <= REACT_MAX_STEPS).
    #   = 1: hard switch only on the last step, no soft reminder. = 2 (default): soft reminder at second-to-last + hard switch on last.
    REACT_BUDGET_AWARENESS_PROTOCOL: bool = True
    REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT: bool = True
    REACT_FORCE_FINAL_ANSWER_LOOKAHEAD: int = 2

    # Grid search anchors (optional): when .env is configured with multi-value R / C, the paper main figure
    # pins R = GRID_DEFAULT_R to draw the BI vs C curve; similarly GRID_DEFAULT_C controls the C anchor for
    # the BI vs R curve. When unset, plot_analysis.py defaults to the first value of each list.
    # Must belong to the corresponding list (validated at startup).
    GRID_DEFAULT_R: int | None = None
    GRID_DEFAULT_C: int | None = None

    # Master switch for the structured confidence protocol. When enabled, a belief protocol section
    # is appended at the end of the user prompt (positioned after the reflection protocol), requiring the
    # LLM to emit a <belief>...</belief> JSON before \\boxed{...}, carrying probabilities / confidence /
    # key_evidence / counterevidence / decision_rule. Independent of the reflection protocol; does not enter
    # prompt_templates_hash; fingerprinted via run_meta.belief_protocol_text/_hash.
    # Off by default to preserve v3 behavior byte-for-byte; pilot parse rate on candidate models before enabling.
    BELIEF_PROTOCOL: bool = False

    # -------- Search leak filter (Stage 2 detector, search-leak-filter-v1) --------
    # Master switch: when True, every result from tavily_search is audited by an independent LLM (detector)
    # before return; verdict=drop removes the entire item. When False, byte-for-byte revert to pre-proposal behavior.
    # Enabled by default (default-strict). Requires LEAK_DETECTOR_API_KEY / LEAK_DETECTOR_MODEL to be filled in,
    # and ENABLE_WEB_SEARCH must also be True (otherwise the detector path is dead code).
    ENABLE_SEARCH_LEAK_FILTER: bool = True
    # Detector's own OpenAI-compatible endpoint config, fully decoupled from LLM_*, allowing the detector
    # to use a more advanced model than the evaluated models for strict judgment.
    LEAK_DETECTOR_API_KEY: str = ""
    # When empty, leak_filter back-fills LLM_BASE_URL during client initialization.
    LEAK_DETECTOR_BASE_URL: str = ""
    # Detector model slug. ":online" suffix is not allowed (provider-native browsing protection).
    LEAK_DETECTOR_MODEL: str = ""
    LEAK_DETECTOR_TIMEOUT_S: int = 60
    LEAK_DETECTOR_TEMPERATURE: float = 0.0
    LEAK_DETECTOR_MAX_TOKENS: int = 512
    LEAK_DETECTOR_RETRY_MAX: int = 3
    LEAK_DETECTOR_BACKOFF_S: Annotated[list[int], NoDecode] = Field(default_factory=lambda: [2, 5, 15])
    # drop = remove failed items (default, fail-closed); keep = pass through failed items (escape hatch).
    LEAK_DETECTOR_FAIL_ACTION: str = "drop"
    LEAK_DETECTOR_CONCURRENCY: int = 5
    # Manual version label; sha256(prompt_template) is auto-hashed; this is a human-readable label.
    LEAK_DETECTOR_PROMPT_VERSION: str = "v1"

    # -------- Composite difficulty coefficients --------
    # The composite difficulty coefficient factorises along two orthogonal
    # axes: a question-type prior (0.15, 0.15, 0.70) over
    # {yes_no, binary_named, multiple_choice} and a within-mc answering-mode
    # split (0.40, 0.60) over {single, multi}. Multiplying the two yields the
    # four scoring buckets {yes_no, binary_named, mc_single, mc_multi} with
    # weights (0.15, 0.15, 0.28, 0.42). For all metrics (all columns of
    # per_model_summary) the analysis layer applies "compute per bucket ->
    # synthesise by weight -> drop missing buckets and renormalise", writing
    # runs/{run_id}/analysis/per_model_composite.csv. The legacy
    # per_model_summary.csv keeps its "all questions mixed" semantics.
    COMPOSITE_WEIGHTS: Annotated[dict[str, float], NoDecode] = Field(
        default_factory=lambda: {
            "yes_no": 0.15,
            "binary_named": 0.15,
            "mc_single": 0.28,
            "mc_multi": 0.42,
        }
    )
    # Per-metric overrides: of the form ``"fss=yes_no=0.05,binary_named=0.05,
    # mc_single=0.20,mc_multi=0.70;cohen_kappa=..."``. Defaults fall back to
    # the global defaults; misspelled metric names (not in _SUMMARY_FIELDS +
    # probability-family whitelist) are raised at runtime by
    # analysis.composite.
    COMPOSITE_WEIGHT_OVERRIDES: Annotated[
        dict[str, dict[str, float]], NoDecode
    ] = Field(default_factory=dict)

    # Sampling / Run
    SAMPLING_N: int = 5
    RUN_ID: str = ""
    RESUME: bool = True

    # Database
    SOURCE_DB: str = "./forecast_eval_set_example.db"
    # Question table inside SOURCE_DB. The bundled example DB ships with
    # `test_cases`; bring-your-own datasets can point at any other
    # table name as long as it has the same 7-column schema (see FRAME.md §2.1).
    SOURCE_TABLE: str = "test_cases"
    # Every evaluation gets its own folder at RUNS_ROOT/{run_id}/, containing one
    # SQLite file per model under db/, plus analysis/ (post-run statistics) and
    # logs/. The old single-file RESULTS_DB layout is gone — see FRAME.md §5/§6.
    RUNS_ROOT: str = "./runs"
    DB_COMMIT_BATCH: int = 10
    WRITE_MESSAGES_TRACE: bool = True

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "./logs"

    @field_validator("MODELS", mode="before")
    @classmethod
    def _parse_models(cls, v: Any) -> list[str]:
        return _parse_csv(v)

    @field_validator("TAVILY_API_KEY", mode="before")
    @classmethod
    def _parse_tavily_keys(cls, v: Any) -> list[str]:
        # Single value (`tvly-xxx`) parses to a length-1 list, CSV multi-value expands. Empty returns []
        # so _post_validate uniformly decides whether to require non-empty based on ENABLE_WEB_SEARCH.
        return _parse_csv(v)

    @field_validator(
        "LLM_BACKOFF_NETWORK_S",
        "LLM_BACKOFF_RATE_LIMIT_S",
        "LLM_BACKOFF_SERVER_5XX_S",
        "SEARCH_BACKOFF_S",
        "LEAK_DETECTOR_BACKOFF_S",
        mode="before",
    )
    @classmethod
    def _parse_backoffs(cls, v: Any) -> list[int]:
        if isinstance(v, list):
            return [int(x) for x in v]
        return _parse_csv_int(v)

    @field_validator(
        "TAVILY_MAX_RESULTS",
        "REACT_MAX_SEARCH_CALLS",
        mode="before",
    )
    @classmethod
    def _parse_grid_int_list(cls, v: Any) -> list[int]:
        # Multi-value grid axis: CSV in .env, list/int passthrough for tests.
        # Single-value envs degrade to a length-1 list (back-compat with v4
        # `TAVILY_MAX_RESULTS=5`); empty values raise so we never silently
        # swallow a misconfigured .env into a no-op dispatcher.
        if isinstance(v, list):
            parsed = [int(x) for x in v]
        elif isinstance(v, int) and not isinstance(v, bool):
            parsed = [int(v)]
        else:
            parsed = _parse_csv_int(v)
        if not parsed:
            raise ValueError(
                "TAVILY_MAX_RESULTS / REACT_MAX_SEARCH_CALLS must be a non-empty CSV "
                "of positive integers (e.g. '5' or '5,10')"
            )
        for n in parsed:
            if n <= 0:
                raise ValueError(
                    f"TAVILY_MAX_RESULTS / REACT_MAX_SEARCH_CALLS values must be > 0; got {n}"
                )
        return parsed

    @field_validator("LLM_REASONING_MODEL_PATTERNS", mode="before")
    @classmethod
    def _parse_reasoning_patterns(cls, v: Any) -> list[str]:
        return _parse_csv(v)

    @field_validator("GRID_DEFAULT_R", "GRID_DEFAULT_C", mode="before")
    @classmethod
    def _parse_grid_default(cls, v: Any) -> Any:
        # `.env.example` documents that leaving these blank means "take the
        # first list element". A literal empty string from the env file would
        # otherwise fail pydantic's int parser, so coerce "" → None here.
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @field_validator("MODEL_TRAINING_CUTOFFS", mode="before")
    @classmethod
    def _parse_training_cutoffs(cls, v: Any) -> dict[str, date]:
        return _parse_cutoffs(v)

    @field_validator("MODEL_MAX_TOKENS_PARAM", mode="before")
    @classmethod
    def _parse_max_tokens_param_field(cls, v: Any) -> dict[str, str]:
        return _parse_max_tokens_param(v)

    @field_validator("MODEL_OMIT_SAMPLING_FIELDS", mode="before")
    @classmethod
    def _parse_omit_sampling_fields_field(cls, v: Any) -> dict[str, list[str]]:
        return _parse_omit_sampling_fields(v)

    @field_validator("COMPOSITE_WEIGHTS", mode="before")
    @classmethod
    def _parse_composite_weights(cls, v: Any) -> dict[str, float]:
        # Empty string is treated as "use default dict" - pydantic field default_factory already
        # provides the complete default weights; we only parse when the user explicitly sets a value.
        if v is None or v == "":
            return {
                "yes_no": 0.15,
                "binary_named": 0.15,
                "mc_single": 0.28,
                "mc_multi": 0.42,
            }
        return _parse_weights_dict(
            v,
            allowed_buckets=COMPOSITE_BUCKETS,
            field_name="COMPOSITE_WEIGHTS",
        )

    @field_validator("COMPOSITE_WEIGHT_OVERRIDES", mode="before")
    @classmethod
    def _parse_composite_overrides(
        cls, v: Any
    ) -> dict[str, dict[str, float]]:
        return _parse_overrides_dict(
            v,
            allowed_buckets=COMPOSITE_BUCKETS,
            field_name="COMPOSITE_WEIGHT_OVERRIDES",
        )

    @field_validator("TAVILY_INCLUDE_RAW_CONTENT", mode="before")
    @classmethod
    def _parse_include_raw_content(cls, v: Any) -> str:
        # Compatibility for legacy bool values: True -> "markdown", False -> "false"
        if isinstance(v, bool):
            return "markdown" if v else "false"
        s = str(v).strip().lower()
        if s == "true":
            return "markdown"
        if s in ("false", "markdown", "text"):
            return s
        raise ValueError(
            f"TAVILY_INCLUDE_RAW_CONTENT must be one of false|markdown|text "
            f"(or legacy bool); got {v!r}"
        )

    @field_validator("TAVILY_INCLUDE_ANSWER", mode="before")
    @classmethod
    def _parse_include_answer(cls, v: Any) -> str:
        if isinstance(v, bool):
            return "basic" if v else "false"
        s = str(v).strip().lower()
        if s == "true":
            return "basic"
        if s in ("false", "basic", "advanced"):
            return s
        raise ValueError(
            f"TAVILY_INCLUDE_ANSWER must be one of false|basic|advanced; got {v!r}"
        )

    @field_validator("TAVILY_SEARCH_DEPTH", mode="before")
    @classmethod
    def _parse_search_depth(cls, v: Any) -> str:
        s = str(v).strip().lower()
        if s in ("basic", "advanced"):
            return s
        raise ValueError(
            f"TAVILY_SEARCH_DEPTH must be one of basic|advanced; got {v!r}"
        )

    @field_validator("RUN_ID")
    @classmethod
    def _validate_run_id(cls, v: str) -> str:
        if v and not _RUN_ID_RE.match(v):
            raise ValueError(
                f"RUN_ID {v!r} does not match YYYYMMDD-HHMMSS-xxxx (4 hex) format"
            )
        return v

    @field_validator("SOURCE_TABLE")
    @classmethod
    def _validate_source_table(cls, v: str) -> str:
        # SOURCE_TABLE is interpolated into SQL — restrict to a safe identifier
        # so misconfiguration cannot turn into injection.
        if not _SQL_IDENT_RE.match(v):
            raise ValueError(
                f"SOURCE_TABLE {v!r} must match [A-Za-z_][A-Za-z0-9_]* (SQLite identifier)"
            )
        return v

    @model_validator(mode="after")
    def _post_validate(self) -> "Settings":
        if not self.MODELS:
            raise ValueError("MODELS must not be empty")
        for slug in self.MODELS:
            if slug.endswith(":online"):
                raise ValueError(
                    f"MODELS entry {slug!r} must not end with ':online' — "
                    "provider-native browsing is not allowed (see information-barrier spec)"
                )
            # Defensive: real_model slugs containing `::` would collide with the
            # virtual slug encoding `{real}::r{R}::c{C}` and break round-tripping
            # in `db.parse_virtual_slug`. Reject early with a clear message.
            if "::" in slug:
                raise ValueError(
                    f"MODELS entry {slug!r} must not contain '::' — that delimiter is "
                    "reserved for grid-search virtual slugs (compose_virtual_slug)"
                )
        # LLM_API_KEY is a plain string; TAVILY_API_KEY is a list[str]
        # (CSV multi-key support), so each placeholder must be validated individually.
        if not self.LLM_API_KEY:
            raise ValueError("LLM_API_KEY must not be empty")
        if any(tok in self.LLM_API_KEY for tok in _PLACEHOLDER_TOKENS):
            raise ValueError(
                "LLM_API_KEY still holds a placeholder token; fill your real key in .env"
            )
        if self.ENABLE_WEB_SEARCH:
            if not self.TAVILY_API_KEY:
                raise ValueError(
                    "TAVILY_API_KEY must not be empty when ENABLE_WEB_SEARCH=true "
                    "(provide one or more comma-separated keys)"
                )
            for idx, key in enumerate(self.TAVILY_API_KEY):
                if not key:
                    raise ValueError(f"TAVILY_API_KEY[{idx}] is empty")
                if any(tok in key for tok in _PLACEHOLDER_TOKENS):
                    raise ValueError(
                        f"TAVILY_API_KEY[{idx}] still holds a placeholder token; "
                        "fill real keys in .env"
                    )
        if self.TAVILY_KEY_COOLDOWN_S < 0:
            raise ValueError(
                f"TAVILY_KEY_COOLDOWN_S must be >= 0; got {self.TAVILY_KEY_COOLDOWN_S}"
            )
        if self.SAMPLING_N < 1:
            raise ValueError("SAMPLING_N must be >= 1")
        if self.LLM_MAX_CONCURRENCY < 1 or self.SEARCH_MAX_CONCURRENCY < 1:
            raise ValueError("concurrency settings must be >= 1")
        if self.REACT_MAX_STEPS < 1:
            raise ValueError("REACT_MAX_STEPS must be >= 1")
        # REACT_MAX_SEARCH_CALLS is a list (multi-value grid axis); each entry
        # must be a non-negative C value. The field_validator already requires
        # > 0 at parse time, but keep this as a belt-and-braces guard for tests
        # that construct Settings via overrides.
        for c in self.REACT_MAX_SEARCH_CALLS:
            if c < 0:
                raise ValueError(
                    f"REACT_MAX_SEARCH_CALLS contains a negative value: {c}"
                )
        for r in self.TAVILY_MAX_RESULTS:
            if r <= 0:
                raise ValueError(
                    f"TAVILY_MAX_RESULTS must be > 0 per cell; got {r}"
                )
        if self.REACT_MIN_SEARCH_CALLS < 0:
            raise ValueError("REACT_MIN_SEARCH_CALLS must be >= 0 (0 = disabled)")
        # MIN is a single int but C is a list. Hard error only when MIN exceeds
        # the smallest C in the grid (then no cell could honor the floor). When
        # MIN exceeds *some* but not all cells, the dispatcher silently clamps
        # `effective_min = min(MIN, C)` per cell — see DESIGN.md decision 4.
        if self.REACT_MAX_SEARCH_CALLS and self.REACT_MIN_SEARCH_CALLS > min(self.REACT_MAX_SEARCH_CALLS):
            raise ValueError(
                "REACT_MIN_SEARCH_CALLS must not exceed min(REACT_MAX_SEARCH_CALLS) "
                f"(min_floor={self.REACT_MIN_SEARCH_CALLS}, c_list={self.REACT_MAX_SEARCH_CALLS})"
            )
        if self.REACT_MAX_NUDGES < 0:
            raise ValueError("REACT_MAX_NUDGES must be >= 0")
        # Harness-resilience switches: pydantic has already validated the bool type; this is just
        # a defensive type guard to prevent tests from injecting arbitrary values via model_copy(update={...}).
        if not isinstance(self.REACT_FINAL_ANSWER_RETRY, bool):
            raise ValueError(
                f"REACT_FINAL_ANSWER_RETRY must be bool; got {type(self.REACT_FINAL_ANSWER_RETRY).__name__}"
            )
        if not isinstance(self.REACT_BUDGET_EXCEEDED_DROP_TOOLS, bool):
            raise ValueError(
                f"REACT_BUDGET_EXCEEDED_DROP_TOOLS must be bool; got "
                f"{type(self.REACT_BUDGET_EXCEEDED_DROP_TOOLS).__name__}"
            )
        # force-final-answer-near-limit-v1 triple-set validation.
        if not isinstance(self.REACT_BUDGET_AWARENESS_PROTOCOL, bool):
            raise ValueError(
                f"REACT_BUDGET_AWARENESS_PROTOCOL must be bool; got "
                f"{type(self.REACT_BUDGET_AWARENESS_PROTOCOL).__name__}"
            )
        if not isinstance(self.REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT, bool):
            raise ValueError(
                f"REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT must be bool; got "
                f"{type(self.REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT).__name__}"
            )
        if self.REACT_FORCE_FINAL_ANSWER_LOOKAHEAD < 1:
            raise ValueError(
                "REACT_FORCE_FINAL_ANSWER_LOOKAHEAD must be >= 1 "
                f"(0 = no near-limit window; disable via REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT=false instead); "
                f"got {self.REACT_FORCE_FINAL_ANSWER_LOOKAHEAD}"
            )
        if self.REACT_FORCE_FINAL_ANSWER_LOOKAHEAD > self.REACT_MAX_STEPS:
            raise ValueError(
                f"REACT_FORCE_FINAL_ANSWER_LOOKAHEAD={self.REACT_FORCE_FINAL_ANSWER_LOOKAHEAD} "
                f"must not exceed REACT_MAX_STEPS={self.REACT_MAX_STEPS} "
                "(otherwise every step would be treated as near-limit, defeating the purpose)"
            )
        if self.TAVILY_RAW_CONTENT_MAX_CHARS < 0:
            raise ValueError("TAVILY_RAW_CONTENT_MAX_CHARS must be >= 0 (0 = no truncation)")
        # Optional grid anchors: when set, must be one of the configured cells.
        if self.GRID_DEFAULT_R is not None and self.GRID_DEFAULT_R not in self.TAVILY_MAX_RESULTS:
            raise ValueError(
                f"GRID_DEFAULT_R={self.GRID_DEFAULT_R} not in TAVILY_MAX_RESULTS "
                f"={self.TAVILY_MAX_RESULTS}; pick one of the configured cells"
            )
        if self.GRID_DEFAULT_C is not None and self.GRID_DEFAULT_C not in self.REACT_MAX_SEARCH_CALLS:
            raise ValueError(
                f"GRID_DEFAULT_C={self.GRID_DEFAULT_C} not in REACT_MAX_SEARCH_CALLS "
                f"={self.REACT_MAX_SEARCH_CALLS}; pick one of the configured cells"
            )
        # search-leak-filter-v1 startup validation
        if self.LEAK_DETECTOR_FAIL_ACTION not in ("drop", "keep"):
            raise ValueError(
                f"LEAK_DETECTOR_FAIL_ACTION must be one of {{drop, keep}}; "
                f"got {self.LEAK_DETECTOR_FAIL_ACTION!r}"
            )
        if self.LEAK_DETECTOR_CONCURRENCY < 1:
            raise ValueError(
                f"LEAK_DETECTOR_CONCURRENCY must be >= 1; got {self.LEAK_DETECTOR_CONCURRENCY}"
            )
        if self.LEAK_DETECTOR_RETRY_MAX < 0:
            raise ValueError(
                f"LEAK_DETECTOR_RETRY_MAX must be >= 0; got {self.LEAK_DETECTOR_RETRY_MAX}"
            )
        if self.LEAK_DETECTOR_TIMEOUT_S <= 0:
            raise ValueError(
                f"LEAK_DETECTOR_TIMEOUT_S must be > 0; got {self.LEAK_DETECTOR_TIMEOUT_S}"
            )
        if self.LEAK_DETECTOR_TEMPERATURE < 0:
            raise ValueError(
                f"LEAK_DETECTOR_TEMPERATURE must be >= 0; got {self.LEAK_DETECTOR_TEMPERATURE}"
            )
        if self.LEAK_DETECTOR_MAX_TOKENS < 1:
            raise ValueError(
                f"LEAK_DETECTOR_MAX_TOKENS must be >= 1; got {self.LEAK_DETECTOR_MAX_TOKENS}"
            )
        if self.LEAK_DETECTOR_MODEL.endswith(":online"):
            raise ValueError(
                f"LEAK_DETECTOR_MODEL {self.LEAK_DETECTOR_MODEL!r} must not end with "
                "':online' — provider-native browsing is not allowed"
            )
        if self.ENABLE_SEARCH_LEAK_FILTER:
            if not self.ENABLE_WEB_SEARCH:
                raise ValueError(
                    "ENABLE_SEARCH_LEAK_FILTER requires ENABLE_WEB_SEARCH=true; "
                    "the leak filter only runs against tavily_search results"
                )
            if not self.LEAK_DETECTOR_API_KEY:
                raise ValueError(
                    "LEAK_DETECTOR_API_KEY must not be empty when "
                    "ENABLE_SEARCH_LEAK_FILTER=true"
                )
            if any(tok in self.LEAK_DETECTOR_API_KEY for tok in _PLACEHOLDER_TOKENS):
                raise ValueError(
                    "LEAK_DETECTOR_API_KEY still holds a placeholder token; "
                    "fill your real detector key in .env"
                )
            if not self.LEAK_DETECTOR_MODEL:
                raise ValueError(
                    "LEAK_DETECTOR_MODEL must not be empty when "
                    "ENABLE_SEARCH_LEAK_FILTER=true"
                )
            if any(tok in self.LEAK_DETECTOR_MODEL for tok in _PLACEHOLDER_TOKENS):
                raise ValueError(
                    "LEAK_DETECTOR_MODEL still holds a placeholder token; "
                    "fill a real model slug in .env"
                )
        # Composite difficulty post-validation: field_validator (mode=before)
        # does not handle dicts produced directly by default_factory, nor
        # values injected via model_copy(update=...); we run through again
        # here to guard against in-code-constructed Settings.
        for bucket, weight in self.COMPOSITE_WEIGHTS.items():
            if bucket not in COMPOSITE_BUCKETS:
                raise ValueError(
                    f"COMPOSITE_WEIGHTS bucket {bucket!r} not in "
                    f"{sorted(COMPOSITE_BUCKETS)}"
                )
            if weight < 0:
                raise ValueError(
                    f"COMPOSITE_WEIGHTS[{bucket}] must be >= 0; got {weight}"
                )
        if not any(w > 0 for w in self.COMPOSITE_WEIGHTS.values()):
            raise ValueError(
                "COMPOSITE_WEIGHTS requires at least one weight > 0 "
                "(otherwise composite score has no defined denominator)"
            )
        for metric, sub in self.COMPOSITE_WEIGHT_OVERRIDES.items():
            if not metric:
                raise ValueError(
                    "COMPOSITE_WEIGHT_OVERRIDES has an empty metric name"
                )
            for bucket, weight in sub.items():
                if bucket not in COMPOSITE_BUCKETS:
                    raise ValueError(
                        f"COMPOSITE_WEIGHT_OVERRIDES[{metric}] bucket "
                        f"{bucket!r} not in {sorted(COMPOSITE_BUCKETS)}"
                    )
                if weight < 0:
                    raise ValueError(
                        f"COMPOSITE_WEIGHT_OVERRIDES[{metric}][{bucket}] "
                        f"must be >= 0; got {weight}"
                    )
            if sub and not any(w > 0 for w in sub.values()):
                raise ValueError(
                    f"COMPOSITE_WEIGHT_OVERRIDES[{metric}] requires at "
                    "least one weight > 0"
                )
        return self

    def __repr__(self) -> str:
        redacted = {
            k: ("<redacted>" if k.endswith("_API_KEY") else v)
            for k, v in self.model_dump().items()
        }
        return f"Settings({redacted!r})"

    __str__ = __repr__

    def source_db_path(self) -> Path:
        return Path(self.SOURCE_DB).expanduser().resolve()

    def runs_root_path(self) -> Path:
        return Path(self.RUNS_ROOT).expanduser().resolve()

    def run_dir(self, run_id: str) -> Path:
        return self.runs_root_path() / run_id

    def log_dir_path(self) -> Path:
        return Path(self.LOG_DIR).expanduser().resolve()


def load_settings(**overrides: Any) -> Settings:
    """Construct a Settings instance; `overrides` take precedence over env."""
    return Settings(**overrides)
