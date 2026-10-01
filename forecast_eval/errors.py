from __future__ import annotations

import asyncio
import json
from enum import StrEnum
from typing import Any, Optional

import httpx
from openai import APIConnectionError


class ErrorKind(StrEnum):
    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    SERVER_5XX = "server_5xx"
    AUTH = "auth"
    BAD_REQUEST = "bad_request"
    CONTENT_POLICY = "content_policy"
    UNKNOWN = "unknown"


class AuthError(Exception):
    """Raised once by `llm.chat` when the API key is invalid / forbidden.

    Runner catches this at the top level, cancels all in-flight tasks, flushes
    the writer, and exits with a non-zero status. Content of the run_results
    rows depends on runner policy.
    """


class CollectionBlockedError(Exception):
    """Collection cannot preserve its retrieval or persistence contract."""


class AnalysisContractError(ValueError):
    """Stored observations or analysis inputs violate the scoring contract."""


class AnalysisIncompleteError(AnalysisContractError):
    """A declared evaluation panel has missing or failed sample slots."""


# Content-policy needles for HTTP 400 bodies.
# `_body_matches` runs case-insensitive substring matching on `_error_body`,
# so needles MUST be lowercase ASCII. Update only here when a new provider's
# rejection vocabulary appears.
#
# Coverage:
# - English (OpenAI / Anthropic style): content_policy / content_filter / safety
# - Aliyun DashScope (qwen* via dashscope.aliyuncs.com): data_inspection_failed,
#   "inappropriate content" (the canonical message body)
CONTENT_POLICY_NEEDLES: tuple[str, ...] = (
    "content_policy",
    "content filter",
    "content_filter",
    "safety",
    "content_policy_violation",
    "data_inspection_failed",
    "inappropriate content",
    "sensitive",
)


def _status_code(exc: BaseException) -> Optional[int]:
    """Try to pull an HTTP status code out of an httpx / openai exception."""
    resp = getattr(exc, "response", None)
    if resp is not None:
        code = getattr(resp, "status_code", None)
        if isinstance(code, int):
            return code
    # openai SDK raises openai.APIStatusError with .status_code
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    return None


def _error_body(exc: BaseException) -> str:
    """Best-effort string representation of the server-sent error body."""
    resp = getattr(exc, "response", None)
    if resp is not None:
        text = getattr(resp, "text", None)
        if isinstance(text, str) and text:
            return text
    body = getattr(exc, "body", None)
    if body is not None:
        try:
            return json.dumps(body, ensure_ascii=False)
        except TypeError:
            return str(body)
    return str(exc)


def _body_matches(exc: BaseException, needles: tuple[str, ...]) -> bool:
    body = _error_body(exc).lower()
    return any(n in body for n in needles)


def classify(exc: BaseException) -> ErrorKind:
    """Map an outgoing-HTTP exception to a coarse ErrorKind for retry decisions.

    The network family includes httpx transport failures and the OpenAI SDK
    connection/timeout wrappers. A transport retry keeps the current model
    context, including search results already obtained for this sample.
    """
    if isinstance(
        exc,
        (
            APIConnectionError,
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.ConnectTimeout,
            httpx.WriteTimeout,
            httpx.WriteError,
            httpx.PoolTimeout,
            httpx.RemoteProtocolError,
        ),
    ):
        return ErrorKind.NETWORK
    if isinstance(exc, asyncio.TimeoutError):
        return ErrorKind.NETWORK

    code = _status_code(exc)
    if code is not None:
        if code in (401, 403):
            return ErrorKind.AUTH
        if code == 429:
            return ErrorKind.RATE_LIMIT
        if 500 <= code <= 599:
            return ErrorKind.SERVER_5XX
        if code == 400:
            # CONTENT_POLICY MUST be checked before BAD_REQUEST. Some bodies
            # carry both `data_inspection_failed` (provider-side moderation,
            # we should not retry but ALSO should not silently look like a
            # bug-in-our-request) and a generic "invalid request" wrapper.
            # Spec llm-integration §"Content policy: no retry" pins priority.
            if _body_matches(exc, CONTENT_POLICY_NEEDLES):
                return ErrorKind.CONTENT_POLICY
            if _body_matches(exc, ("model_not_found", "invalid_request", "invalid request", "invalid model")):
                return ErrorKind.BAD_REQUEST
            return ErrorKind.BAD_REQUEST

    return ErrorKind.UNKNOWN


def should_retry(kind: ErrorKind) -> bool:
    return kind in (ErrorKind.NETWORK, ErrorKind.RATE_LIMIT, ErrorKind.SERVER_5XX)


def _sequence_for(kind: ErrorKind, settings: Any) -> list[int]:
    if kind is ErrorKind.NETWORK:
        return list(settings.LLM_BACKOFF_NETWORK_S)
    if kind is ErrorKind.RATE_LIMIT:
        return list(settings.LLM_BACKOFF_RATE_LIMIT_S)
    if kind is ErrorKind.SERVER_5XX:
        return list(settings.LLM_BACKOFF_SERVER_5XX_S)
    return []


def backoff_seconds(
    kind: ErrorKind,
    attempt: int,
    settings: Any,
    retry_after: Optional[float] = None,
) -> Optional[float]:
    """Return how long to wait before attempt `attempt` (1-indexed), or None.

    None signals the caller "don't retry further". For RATE_LIMIT we honour an
    explicit Retry-After first; for every other retryable kind we index into
    the corresponding Settings sequence by `attempt-1`.
    """
    if not should_retry(kind):
        return None
    if kind is ErrorKind.RATE_LIMIT and retry_after is not None:
        return max(0.0, float(retry_after))
    sequence = _sequence_for(kind, settings)
    idx = attempt - 1
    if 0 <= idx < len(sequence):
        return float(sequence[idx])
    return None


def parse_retry_after(headers: Any) -> Optional[float]:
    """Extract a Retry-After header value in seconds (HTTP-date form is ignored)."""
    if headers is None:
        return None
    get = getattr(headers, "get", None)
    if get is None:
        return None
    raw = get("Retry-After") or get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
