from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from forecast_eval.errors import (
    AuthError,
    CONTENT_POLICY_NEEDLES,
    ErrorKind,
    backoff_seconds,
    classify,
    parse_retry_after,
    should_retry,
)


@dataclass
class _StubSettings:
    LLM_BACKOFF_NETWORK_S: list[int]
    LLM_BACKOFF_RATE_LIMIT_S: list[int]
    LLM_BACKOFF_SERVER_5XX_S: list[int]


def _settings() -> _StubSettings:
    return _StubSettings(
        LLM_BACKOFF_NETWORK_S=[2, 5, 15, 30, 60],
        LLM_BACKOFF_RATE_LIMIT_S=[10, 30, 60, 120, 300],
        LLM_BACKOFF_SERVER_5XX_S=[5, 15, 30, 60, 120],
    )


def _http_error(status: int, body: str = "") -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.com/x")
    response = httpx.Response(status_code=status, request=request, text=body)
    return httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)


def test_network_errors_map_to_network() -> None:
    req = httpx.Request("GET", "https://example.com")
    assert classify(httpx.ConnectError("boom", request=req)) is ErrorKind.NETWORK
    assert classify(httpx.ReadTimeout("slow", request=req)) is ErrorKind.NETWORK
    assert classify(httpx.ConnectTimeout("slow", request=req)) is ErrorKind.NETWORK
    assert classify(asyncio.TimeoutError()) is ErrorKind.NETWORK
    # v5.1 (harness-resilience): the new transient-network family. These were
    # previously dropping into UNKNOWN, which never retries — see the spec
    # llm-integration §"network/timeout error tiered backoff" RemoteProtocolError scenario.
    assert classify(httpx.WriteTimeout("slow", request=req)) is ErrorKind.NETWORK
    assert classify(httpx.WriteError("write blew up", request=req)) is ErrorKind.NETWORK
    assert classify(httpx.PoolTimeout("pool exhausted")) is ErrorKind.NETWORK
    assert (
        classify(httpx.RemoteProtocolError("Server disconnected without sending a response."))
        is ErrorKind.NETWORK
    )


def test_content_policy_aliyun_data_inspection_failed() -> None:
    """Aliyun DashScope returns 400 with `data_inspection_failed` for content
    moderation rejections. v5.0 mis-classified these as BAD_REQUEST."""
    body = (
        '{"error":{"message":"Input data may contain inappropriate content",'
        '"type":"invalid_request_error","code":"data_inspection_failed"}}'
    )
    assert classify(_http_error(400, body)) is ErrorKind.CONTENT_POLICY


def test_content_policy_sensitive_token() -> None:
    """A 400 body whose message just mentions `sensitive` content should still
    classify as CONTENT_POLICY rather than falling through to BAD_REQUEST."""
    body = '{"error":{"message":"Request rejected: sensitive content detected"}}'
    assert classify(_http_error(400, body)) is ErrorKind.CONTENT_POLICY


def test_bad_request_invalid_request_still_falls_through() -> None:
    """Plain `invalid_request` body without any content-policy needle stays
    BAD_REQUEST (priority is content-policy first, bad-request as fallback)."""
    body = '{"error":{"message":"invalid request: missing field model"}}'
    assert classify(_http_error(400, body)) is ErrorKind.BAD_REQUEST


def test_content_policy_needles_constant_includes_aliyun() -> None:
    """The constant is the single source of truth — readers grep this list to
    add new providers, so guard that the Aliyun-style entries actually landed."""
    assert "data_inspection_failed" in CONTENT_POLICY_NEEDLES
    assert "inappropriate content" in CONTENT_POLICY_NEEDLES
    assert "sensitive" in CONTENT_POLICY_NEEDLES


def test_status_codes_classification() -> None:
    assert classify(_http_error(401)) is ErrorKind.AUTH
    assert classify(_http_error(403)) is ErrorKind.AUTH
    assert classify(_http_error(429)) is ErrorKind.RATE_LIMIT
    for code in (500, 502, 503, 504):
        assert classify(_http_error(code)) is ErrorKind.SERVER_5XX
    assert classify(_http_error(400, '{"error": {"code": "model_not_found"}}')) is ErrorKind.BAD_REQUEST
    assert classify(_http_error(400, '{"error": {"code": "content_policy_violation"}}')) is ErrorKind.CONTENT_POLICY
    assert classify(_http_error(418)) is ErrorKind.UNKNOWN


def test_should_retry_rules() -> None:
    assert should_retry(ErrorKind.NETWORK)
    assert should_retry(ErrorKind.RATE_LIMIT)
    assert should_retry(ErrorKind.SERVER_5XX)
    assert not should_retry(ErrorKind.AUTH)
    assert not should_retry(ErrorKind.BAD_REQUEST)
    assert not should_retry(ErrorKind.CONTENT_POLICY)
    assert not should_retry(ErrorKind.UNKNOWN)


def test_backoff_sequences() -> None:
    s = _settings()
    assert backoff_seconds(ErrorKind.NETWORK, 1, s) == 2.0
    assert backoff_seconds(ErrorKind.NETWORK, 5, s) == 60.0
    # exhausted -> None
    assert backoff_seconds(ErrorKind.NETWORK, 6, s) is None


def test_rate_limit_retry_after_wins() -> None:
    s = _settings()
    assert backoff_seconds(ErrorKind.RATE_LIMIT, 1, s, retry_after=30) == 30.0
    # no header -> fall back to sequence
    assert backoff_seconds(ErrorKind.RATE_LIMIT, 1, s) == 10.0
    assert backoff_seconds(ErrorKind.RATE_LIMIT, 6, s) is None


def test_auth_error_type_exists() -> None:
    with pytest.raises(AuthError):
        raise AuthError("nope")


def test_parse_retry_after_numeric() -> None:
    class H:
        def __init__(self, d: dict[str, str]) -> None:
            self._d = d

        def get(self, k: str) -> Any:
            return self._d.get(k)

    assert parse_retry_after(H({"Retry-After": "15"})) == 15.0
    assert parse_retry_after(H({"retry-after": "7.5"})) == 7.5
    assert parse_retry_after(H({})) is None
    assert parse_retry_after(H({"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"})) is None
    assert parse_retry_after(None) is None


@pytest.mark.parametrize("message", [
    "Tool call validation failed: parameters for tool web_search did not match schema: missing properties: 'query'",
    "Tool choice is none, but model called a tool (tid: fixture)",
])
def test_generated_tool_failure_is_not_a_transport_retry(message):
    import json
    body = json.dumps({"error": {"code": "tool_use_failed", "message": message}})
    assert classify(_http_error(400, body)) is ErrorKind.TOOL_USE_FAILED
    assert not should_retry(ErrorKind.TOOL_USE_FAILED)
    assert classify(_http_error(401, body)) is ErrorKind.AUTH
    assert classify(_http_error(429, body)) is ErrorKind.RATE_LIMIT
    assert classify(_http_error(503, body)) is ErrorKind.SERVER_5XX
    assert classify(_http_error(400, body.replace("tool_use_failed", "invalid_request"))) is ErrorKind.BAD_REQUEST
    assert classify(_http_error(400, '{"error":{"code":"tool_use_failed","message":"Invalid tools schema"}}')) is ErrorKind.BAD_REQUEST
